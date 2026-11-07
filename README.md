# RETROFIT AI experiments

- [Original challenge](original-challenge/README_FIRST.md): exact tracked contents of the initial participant-package commit `8ceb2a1`. Do not edit these reference files.
- [Shared data](data/README.md): existing `full/` and `starter/` datasets, read only for experiments.
- [Baseline reproduction](baseline-reproduction/README.md): B1 evaluation and official B1–B3 reproduction outputs.
- [Physics/context detector](physics-context-detector/README.md): B4 detector, development helpers, causality tests and frozen predictions.
- [Operator demo](operator-demo/README.md): presentation setup and launch commands for macOS/Linux and Windows PowerShell.
- [Pump assist](pump-assist/README.md): worklist implementation (ingest, detect, evidence, recommend, app, reports) and its results.
- [Pump assist v2](pump-assist-v2/README.md): the same system with ideas taken from the B4 detector, tried one at a time.
- [Repository layout and provenance](docs/REPOSITORY_LAYOUT.md).

## macOS / Linux

Install dependencies with `python -m pip install -r original-challenge/requirements.txt` and `python -m pip install pytest` for tests. Commands below run from the repository root:

```sh
python baseline-reproduction/scripts/eval.py --b1
python physics-context-detector/scripts/run_b4.py
python physics-context-detector/scripts/eval.py physics-context-detector/results/b4_trainval_predictions.csv --compare-b1
python -m pytest physics-context-detector/tests
```

## Windows PowerShell

Run from the repository root (`Uncandtheboys`). Use Python 3.10 or newer; on Windows the installed interpreter is usually `python`, while `python3` may point to a Windows app alias. These commands use the virtualenv directly, so activation and execution-policy changes are unnecessary. Dependency installation needs network access on the first run.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\original-challenge\requirements.txt
.\.venv\Scripts\python.exe .\baseline-reproduction\scripts\eval.py --b1
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\run_b4.py
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\eval.py .\physics-context-detector\results\b4_trainval_predictions.csv --compare-b1
.\.venv\Scripts\python.exe -m pytest .\physics-context-detector\tests
```

On later runs, skip virtualenv creation and dependency installation. For the presentation, use the separate [operator-demo Windows instructions](operator-demo/README.md#windows-powershell); the demo has its own dependencies and virtualenv.

Experiments read `data/full` independently of the working directory. Predictions, caches, generated stress fleets and other outputs belong inside the experiment that produced them. B4 `--out` accepts only paths within its experiment directory. Shared datasets and the local `.venv` are excluded from Git. Stress fleets are generated locally; see the B4 README.
