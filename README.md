# AgeSplit

Implementation of AoI-constrained split-inference scheduling over bursty edge networks.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

`prepare` creates seeded frame arrivals and bursty-network replay tapes from `config.json`. Route latency, bytes, memory and quality are declared simulation inputs. Actual detector mAP additionally requires `--coco-reference` and `--detections` inputs; generated route quality is not detector accuracy.

```bash
python -m agesplit prepare --config config.json --data data --seed 11
```

## Training and inference

```bash
python -m agesplit run --config config.json --data data --policy agesplit --output outputs/agesplit --seed 11
```

## Evaluation

```bash
python -m agesplit compare --config config.json --data data --output outputs/comparison --seed 11
```

## Outputs

The data folder contains generated replay arrays. Run/comparison folders contain per-policy delivery JSON, `metrics.json` and `service.png`.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
