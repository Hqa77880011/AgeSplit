import argparse
import csv
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def json_ready(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


def age_integral(now, duration, stamp, threshold):
    age = now - stamp
    return (
        age * duration + duration * duration / 2,
        max(0.0, duration - max(0.0, threshold - age)),
    )


def generate(args, cfg):
    rng = np.random.default_rng(args.seed)
    path = Path(args.data)
    path.mkdir(parents=True, exist_ok=True)
    rows, t, bad = ([], 0.0, False)
    while t <= cfg["duration_s"]:
        rows.append(
            [t, cfg["bad_bytes_s"] if bad else cfg["good_bytes_s"], cfg["down_bytes_s"]]
        )
        mean = (
            cfg["bad_burst_s"]
            if bad
            else cfg["bad_burst_s"] * (1 - cfg["bad_fraction"]) / cfg["bad_fraction"]
        )
        t += rng.exponential(mean)
        bad = not bad
    with (path / "network.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "up_bytes_s", "down_bytes_s"])
        writer.writerows(rows)
    with (path / "frames.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capture", "deadline", "image_id"])
        for i, t in enumerate(np.arange(0, cfg["duration_s"], 1 / cfg["fps"])):
            writer.writerow([t, t + cfg["frame_deadline_s"], i])


def replay(frames, network, routes, cfg, policy):
    dt, end = (cfg["tick_s"], cfg["duration_s"])
    caps, refill = (
        np.array(cfg["burst_credits"], float),
        np.array(cfg["refill_rates"], float),
    )
    credits = caps.copy()
    queues = [[] for _ in range(4)]
    active = [None] * 4
    jobs, waiting, completed, logs = ([], [], [], [])
    spent = np.zeros(3)
    area = violation = debt = 0.0
    now, stamp, cursor = (0.0, -cfg["initial_age_s"], 0)
    controller = 0.0
    peak_memory = 0.0
    expired_work = 0.0
    net_index = 0
    while now < end - 1e-10:
        step = min(dt, end - now)
        credits = np.minimum(caps, credits + refill * step)
        while net_index + 1 < len(network) and network[net_index + 1, 0] <= now:
            net_index += 1
        actual_rates = network[net_index, 1:]
        observed_time = max(0.0, now - cfg["telemetry_delay_s"])
        observed_index = max(
            0, np.searchsorted(network[:, 0], observed_time, side="right") - 1
        )
        estimate_rates = np.maximum(
            network[observed_index, 1:], cfg["estimate_rate_floor"]
        )
        while cursor < len(frames) and frames[cursor]["capture"] <= now + 1e-12:
            if len(waiting) + len(jobs) < cfg["queue_limit"]:
                waiting.append(dict(frames[cursor]))
            else:
                logs.append(
                    {"image_id": frames[cursor]["image_id"], "status": "queue-drop"}
                )
            cursor += 1
        begin = time.perf_counter()
        waiting = [f for f in waiting if f["deadline"] >= now]
        for job in list(jobs):
            stage = job["stage"]
            minimum = sum(
                (
                    job["remaining"][s]
                    / (
                        cfg["physical_max_rates"][0]
                        if s == 1
                        else cfg["physical_max_rates"][1] if s == 3 else 1.0
                    )
                    for s in range(stage, 4)
                )
            )
            if policy != "no-cancel" and now + minimum > job["deadline"]:
                if stage in (0, 2) and active[stage] is job:
                    job["discard"] = True
                else:
                    for q in queues:
                        if job in q:
                            q.remove(job)
                    if active[stage] is job:
                        active[stage] = None
                    credits = np.minimum(caps, credits + job["reserved"])
                    expired_work += job["used"][1] + job["used"][2]
                    jobs.remove(job)
        memory = sum((j["memory"] for j in jobs))
        candidates = []
        allowed = [r for r in routes if policy != "local" or r["up_bytes"] == 0]
        if policy in ["fixed-cut", "fixed-gain"]:
            allowed = [routes[cfg["fixed_route"]]]
        for frame in waiting:
            for route in allowed:
                demand = np.array(
                    [
                        route["up_bytes"] + route["down_bytes"] + cfg["control_bytes"],
                        route["local_s"],
                        route["edge_s"],
                    ]
                )
                if (
                    np.any(demand > credits)
                    or memory + route["memory_bytes"] > cfg["memory_bytes"]
                ):
                    continue
                stages = np.array(
                    [
                        route["local_s"],
                        route["up_bytes"],
                        route["edge_s"],
                        route["down_bytes"],
                    ],
                    float,
                )
                delay = 0.0
                for s in range(4):
                    ahead = sum((j["remaining"][s] for j in queues[s]))
                    if active[s] is not None:
                        ahead += active[s]["remaining"][s]
                    rate = (
                        estimate_rates[0]
                        if s == 1
                        else estimate_rates[1] if s == 3 else 1.0
                    )
                    delay += (stages[s] + ahead) / rate + (
                        cfg["link_delay_s"] if s in (1, 3) and stages[s] else 0.0
                    )
                delay += cfg["admission_reserve_s"]
                if now + delay + cfg["clock_uncertainty_s"] > frame["deadline"]:
                    continue
                horizon = cfg["horizon_s"]
                gain = max(0.0, horizon - delay) * max(0.0, frame["capture"] - stamp)
                idle_v = max(
                    0.0, horizon - max(delay, cfg["aoi_threshold_s"] - (now - stamp))
                )
                new_v = max(
                    0.0,
                    horizon
                    - max(delay, cfg["aoi_threshold_s"] - (now - frame["capture"])),
                )
                gv = max(0.0, idle_v - new_v) if frame["capture"] > stamp else 0.0
                score = (
                    cfg["quality_weight"] * route["quality"]
                    + cfg["age_weight"] * gain / horizon**2
                )
                score += (
                    (
                        cfg["violation_weight"]
                        + (0 if policy == "no-debt" else debt / horizon)
                    )
                    * gv
                    / horizon
                )
                score -= np.dot(cfg["resource_prices"], demand / caps)
                if policy == "latency":
                    score = 1.0 / max(delay, 1e-09)
                if policy == "edf":
                    score = 1.0 / max(frame["deadline"] - now, 1e-09) + 1e-06 / max(
                        delay, 1e-09
                    )
                if policy == "fixed-cut":
                    score = 1.0 / (1.0 + frame["capture"])
                candidates.append(
                    (
                        score,
                        -frame["capture"],
                        route["name"],
                        frame,
                        route,
                        demand,
                        stages,
                    )
                )
        if candidates:
            score, _, _, frame, route, demand, stages = max(
                candidates, key=lambda c: c[:3]
            )
            if score > 0:
                waiting.remove(frame)
                credits -= demand
                job = dict(
                    frame,
                    remaining=stages,
                    reserved=demand.copy(),
                    used=np.zeros(3),
                    stage=0,
                    route=route["name"],
                    memory=route["memory_bytes"],
                    discard=False,
                    ready=now,
                )
                job["reserved"][0] -= cfg["control_bytes"]
                spent[0] += cfg["control_bytes"]
                job["used"][0] += cfg["control_bytes"]
                jobs.append(job)
                first = next((s for s in range(4) if stages[s] > 0), 3)
                job["stage"] = first
                queues[first].append(job)
        controller += time.perf_counter() - begin
        peak_memory = max(peak_memory, sum((j["memory"] for j in jobs)))
        updates = []
        for s in range(4):
            if active[s] is None:
                ready = [j for j in queues[s] if j["ready"] <= now]
                if ready:
                    job = ready[0]
                    queues[s].remove(job)
                    active[s] = job
            job = active[s]
            if job is None:
                continue
            rate = actual_rates[0] if s == 1 else actual_rates[1] if s == 3 else 1.0
            work = min(job["remaining"][s], max(0.0, rate) * step)
            job["remaining"][s] -= work
            resource = 0 if s in (1, 3) else 1 if s == 0 else 2
            job["reserved"][resource] -= work
            job["used"][resource] += work
            spent[resource] += work
            if job["remaining"][s] <= 1e-10:
                active[s] = None
                if job["discard"]:
                    credits = np.minimum(caps, credits + job["reserved"])
                    jobs.remove(job)
                    expired_work += job["used"][1] + job["used"][2]
                elif not any(job["remaining"][s + 1 :] > 1e-10):
                    finish = now + step
                    completed.append(
                        {
                            "image_id": job["image_id"],
                            "capture": job["capture"],
                            "finish": finish,
                            "route": job["route"],
                            "fresh": finish <= job["deadline"] + 1e-12,
                        }
                    )
                    if finish <= job["deadline"] + 1e-12:
                        updates.append(job["capture"])
                    jobs.remove(job)
                else:
                    next_stage = next(
                        (j for j in range(s + 1, 4) if job["remaining"][j] > 1e-10)
                    )
                    job["stage"] = next_stage
                    job["ready"] = (
                        now
                        + step
                        + (cfg["link_delay_s"] if next_stage in (1, 3) else 0.0)
                    )
                    queues[next_stage].append(job)
        da, dv = age_integral(now, step, stamp, cfg["aoi_threshold_s"])
        area += da
        violation += dv
        debt = max(0.0, debt + dv - cfg["violation_target"] * step)
        now += step
        if updates:
            stamp = max(stamp, max(updates))
    ages = [r["finish"] - r["capture"] for r in completed]
    fresh = sum((r["fresh"] for r in completed))
    metrics = {
        "policy": policy,
        "mean_aoi_s": area / end,
        "violation_fraction": violation / end,
        "fresh_coverage": fresh / len(frames),
        "missed_fraction": 1 - fresh / len(frames),
        "completion_age_p95_s": float(np.quantile(ages, 0.95)) if ages else None,
        "bytes": spent[0],
        "local_s": spent[1],
        "edge_s": spent[2],
        "controller_ms_per_s": controller * 1000 / end,
        "peak_tensor_bytes": peak_memory,
        "wasted_compute_s": expired_work,
        "reservation_remaining": sum((j["reserved"].sum() for j in jobs)),
    }
    return (metrics, completed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "run", "compare"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="outputs")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument(
        "--policy",
        choices=[
            "agesplit",
            "latency",
            "edf",
            "fixed-cut",
            "fixed-gain",
            "local",
            "no-debt",
            "no-cancel",
        ],
        default="agesplit",
    )
    p.add_argument("--coco-reference")
    p.add_argument("--detections")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "prepare":
        generate(args, cfg)
        return
    with (Path(args.data) / "frames.csv").open() as f:
        frames = [
            {
                "capture": float(r["capture"]),
                "deadline": float(r["deadline"]),
                "image_id": int(r["image_id"]),
            }
            for r in csv.DictReader(f)
        ]
    frames.sort(key=lambda r: r["capture"])
    if not frames:
        raise ValueError("No captured frames")
    network = np.genfromtxt(
        Path(args.data) / "network.csv", delimiter=",", skip_header=1, ndmin=2
    )
    routes = cfg["routes"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    policies = (
        [args.policy]
        if args.command == "run"
        else [
            "agesplit",
            "latency",
            "edf",
            "fixed-cut",
            "fixed-gain",
            "local",
            "no-debt",
            "no-cancel",
        ]
    )
    summaries = []
    for policy in policies:
        metrics, delivery = replay(frames, network, routes, cfg, policy)
        (out / f"{policy}_delivery.json").write_text(
            json.dumps(json_ready(delivery), indent=2, allow_nan=False)
        )
        if args.coco_reference and args.detections:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            all_predictions = json.loads(Path(args.detections).read_text())
            predictions = []
            for row in delivery:
                if row["fresh"]:
                    predictions.extend(
                        all_predictions.get(row["route"], {}).get(
                            str(row["image_id"]), []
                        )
                    )
            truth = COCO(args.coco_reference)
            if predictions:
                evaluator = COCOeval(truth, truth.loadRes(predictions), "bbox")
                evaluator.params.imgIds = [r["image_id"] for r in frames]
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
                metrics["fresh_ap"] = float(evaluator.stats[0])
            else:
                metrics["fresh_ap"] = 0.0
        summaries.append(metrics)
    reference = next((r for r in summaries if r["policy"] == "latency"), None)
    if reference:
        denominator = (
            1000 * reference["local_s"]
            + 200 * reference["edge_s"]
            + 160 * reference["bytes"] / 1000000.0
        )
        for row in summaries:
            row["work_proxy"] = (
                (
                    1000 * row["local_s"]
                    + 200 * row["edge_s"]
                    + 160 * row["bytes"] / 1000000.0
                )
                / denominator
                if denominator
                else None
            )
    (out / "metrics.json").write_text(
        json.dumps(json_ready(summaries), indent=2, allow_nan=False)
    )
    plt.scatter(
        [r["fresh_coverage"] for r in summaries], [r["mean_aoi_s"] for r in summaries]
    )
    for r in summaries:
        plt.annotate(r["policy"], (r["fresh_coverage"], r["mean_aoi_s"]))
    plt.xlabel("Fresh coverage over all captures")
    plt.ylabel("Mean receiver age (s)")
    plt.tight_layout()
    plt.savefig(out / "service.png")
    plt.close()


if __name__ == "__main__":
    main()
