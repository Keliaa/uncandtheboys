# RETROFIT AI: Uncandtheboys

This is our entry for the **RETROFIT AI: Make Existing Machines Intelligent** challenge (Since AI Hackathon 2026, Turku). It is an advisory assistant for operators of cooling-water pumps. It reads the telemetry a legacy or lightly retrofitted pump already produces, spots developing machine faults and sensor faults early, explains the evidence, states its confidence and recommends what a human should check. **It never controls the machine.**

The core of the entry is **B4**, a causal detector that uses physics and operating context. It is in [physics-context-detector/](physics-context-detector/README.md).

## Deliverables

The deliverables listed in [original-challenge/START_HERE.md](original-challenge/START_HERE.md#what-you-deliver), and where to find each one:

| # | Deliverable | Location |
|---|---|---|
| 1 | Working prototype: replay → detect → evidence and uncertainty → recommendation | [operator-demo/](operator-demo/README.md) (B4 operator demo). Also [pump-assist/](pump-assist/README.md) and [pump-assist-v2/](pump-assist-v2/README.md) |
| 2 | **Alerts for the 3 test pumps** (PUMP-10, PUMP-11, PUMP-12) | **[`physics-context-detector/results/b4_test_predictions.csv`](physics-context-detector/results/b4_test_predictions.csv)** |
| 3 | Code and run instructions | This README, plus the README in each experiment directory |
| 3 | Comparison against the provided baselines | [physics-context-detector/RESULTS.md](physics-context-detector/RESULTS.md#b1-vs-b4), [baseline-reproduction/](baseline-reproduction/README.md) |
| 3 | Honest limitations | [physics-context-detector/RESULTS.md](physics-context-detector/RESULTS.md#honest-limitations) |
| 3 | Demo / presentation | [RETROFIT AI DEMO](https://customer-vitality-swarm.lovable.app/) |

### Test-pump alerts (B4)

[`physics-context-detector/results/b4_test_predictions.csv`](physics-context-detector/results/b4_test_predictions.csv) is the scored submission file. It holds 26 alerts and uses the exact column format of [`submission_template/predictions_template.csv`](original-challenge/submission_template/predictions_template.csv):

```
asset_id,alert_time_utc,event_class,fault_type,confidence,evidence,recommended_action
```

| Pump | MACHINE alerts | SENSOR alerts |
|---|---|---|
| PUMP-10 | 3 | 4 |
| PUMP-11 | 1 | 2 |
| PUMP-12 | 16 | 0 |

The file was written once by `run_b4.py --final` using the frozen parameters in [`b4_frozen_params.json`](physics-context-detector/results/b4_frozen_params.json). The parameters were tuned on the training pumps only. The test pumps were never labelled by hand, inspected or used for tuning. The validation-pump alerts are in [`b4_validation_predictions.csv`](physics-context-detector/results/b4_validation_predictions.csv).

## Headline results

These scores come from the official, unmodified scorer (`original-challenge/src/retrofit_sim/scoring.py`):

| Method | Train composite | Validation composite | Stress fleets (32 unseen pumps) |
|---|---|---|---|
| B1 (provided baseline) | 58.2 | 78.0 | 52.3 |
| **B4 (ours)** | **92.3** | **95.7** | **86.8** |

B4 beats B1 on every component of the composite score. [RESULTS.md](physics-context-detector/RESULTS.md) has the full breakdown, a per-event table, a tier ablation, leave-one-pump-out checks and limitations.

## Repository map

| Directory | Contents |
|---|---|
| [original-challenge/](original-challenge/README_FIRST.md) | Exact tracked contents of the initial participant package (commit `8ceb2a1`). Reference only; do not edit. |
| [data/](data/README.md) | Shared `full/` and `starter/` datasets. Read only; excluded from Git. |
| [baseline-reproduction/](baseline-reproduction/README.md) | B1 evaluation and reproduction of the official B1–B3 outputs. |
| [physics-context-detector/](physics-context-detector/README.md) | **B4 detector**, development helpers, causality tests and frozen predictions. |
| [operator-demo/](operator-demo/README.md) | Presenter-controlled operator demo of B4 (macOS/Linux and Windows). |
| [pump-assist/](pump-assist/README.md) | Worklist implementation (ingest, detect, evidence, recommend, app, reports) and its results. |
| [pump-assist-v2/](pump-assist-v2/README.md) | The same system with ideas from B4 added and tested one at a time. |
| [docs/](docs/REPOSITORY_LAYOUT.md) | Repository layout, provenance and team plans. |

## Quick start

Use Python 3.10 or newer. Run all commands from the repository root (`Uncandtheboys`). The first dependency install needs network access.

### macOS / Linux

```sh
python -m venv .venv && source .venv/bin/activate
python -m pip install -r original-challenge/requirements.txt pytest

python baseline-reproduction/scripts/eval.py --b1
python physics-context-detector/scripts/run_b4.py           # add --final to regenerate test predictions
python physics-context-detector/scripts/eval.py physics-context-detector/results/b4_trainval_predictions.csv --compare-b1
python -m pytest physics-context-detector/tests
```

### Windows PowerShell

On Windows the installed interpreter is usually `python`. `python3` may point to a Windows app alias. The commands below call the virtualenv interpreter directly, so you don't need to activate it or change the execution policy.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\original-challenge\requirements.txt pytest

.\.venv\Scripts\python.exe .\baseline-reproduction\scripts\eval.py --b1
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\run_b4.py
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\eval.py .\physics-context-detector\results\b4_trainval_predictions.csv --compare-b1
.\.venv\Scripts\python.exe -m pytest .\physics-context-detector\tests
```

On later runs, skip creating the virtualenv and installing dependencies.

### Operator demo

The demo has its own dependencies and virtualenv. See [operator-demo/README.md](operator-demo/README.md). The Windows instructions are in [its Windows PowerShell section](operator-demo/README.md#windows-powershell).

## Conventions

- Experiments read `data/full` regardless of the working directory. Shared datasets are never an output target.
- Predictions, caches, generated stress fleets and other outputs stay inside the experiment that produced them. B4 `--out` only accepts paths inside its own experiment directory.
- Stress fleets are generated locally (see the [B4 README](physics-context-detector/README.md)). They are excluded from Git, as are `data/` and `.venv`.
- All data is synthetic and comes from the documented challenge simulator. The system gives advice only and sends no commands to devices.
