# AgeSplit

Implementation of AoI-constrained split-inference scheduling over bursty edge networks.

Install with `python -m pip install -e .`, then follow the replay workflow:

```bash
python -m agesplit prepare --seed 11
python -m agesplit compare --output outputs/comparison
python -m agesplit run --policy agesplit --output outputs/agesplit
```

`prepare` writes arrival timestamps and a two-state service-opportunity trace. It does not generate model predictions. Replace `frames.csv` with measured captures (`capture,deadline,image_id`, seconds) and `network.csv` with piecewise-constant uplink/downlink rates (`time,up_bytes_s,down_bytes_s`). Time starts at zero; rows are chronological. Profiles in `config.json` describe legal cuts including all crossing tensors, encoding/decoding compute, return bytes, peak buffer bytes and development-only quality. Replace example profiles with measurements from your detector. COCO and VisDrone provide the paper's detection workloads: [COCO](https://cocodataset.org/) and [VisDrone](https://github.com/VisDrone/VisDrone-Dataset). Separate development/test sequences and network seeds before selecting weights.

The online controller sees current or delayed telemetry. Actual transfers consume the shared exogenous trace and maintain policy-specific queues. The event loop schedules local, uplink, edge and downlink stages, reserving the complete route against replenished byte/local/edge credits. Expired waiting suffixes refund unused reservations; launched compute completes and stays charged. The frame/cut score combines one-refresh age-area gain, violation-duration gain, debt, quality and resource prices. Comparisons retain the same route library and include latency, EDF, fixed-cut, fixed-cut gain, local-only, no-debt and no-cancellation controls.

`tick_s` sets the execution resolution and rounds stage completion to ticks. `horizon_s`, `aoi_threshold_s`, frame deadlines and initial age are distinct. `burst_credits` and `refill_rates` follow byte/local-second/edge-second order; `physical_max_rates` supplies cancellation lower bounds and must bound the actual traces. Memory is reserved per admitted route. The included serial stage executor does not switch an admitted frame to a different cut. `fixed-cut` serves oldest admitted captures on a fixed route; `fixed-gain` retains the receiver-aware score while restricting route choice.

Each run writes delivery records and actual resource/receiver metrics. Mean AoI and violation duration integrate the full observation interval, including its initial and final gaps; coverage divides by every capture. Controller CPU time is measured in the replay process; it is reported separately from profiled device compute. The work proxy converts measured occupied seconds to milliseconds and uses MB with the paper's weights. `service.png` displays the service trade-off.

For full-stream detection AP, install `pycocotools` and pass `--coco-reference instances.json --detections route_predictions.json`. Predictions have shape `{route_name: {image_id: [COCO detection dictionaries]}}`. Only timely deliveries contribute detections, while every captured image remains in the COCO evaluation set. This evaluates your actual detector outputs; route quality estimates are never converted into AP. Multi-stream hardware contention and two-refresh optimization require separately supplied execution traces or a scheduler extension and are not labeled as the included controls.
