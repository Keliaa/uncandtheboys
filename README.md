# RETROFIT AI experiments

- [Original challenge](original-challenge/README_FIRST.md): exact tracked contents of the initial participant-package commit `8ceb2a1`. Do not edit these reference files.
- [Shared data](data/README.md): existing `full/` and `starter/` datasets, read only for experiments.
- [Baseline reproduction](baseline-reproduction/README.md): B1 evaluation and official B1–B3 reproduction outputs.
- [Physics/context detector](physics-context-detector/README.md): B4 detector, development helpers, causality tests and frozen predictions.
- [Repository layout and provenance](docs/REPOSITORY_LAYOUT.md).

Install dependencies with `python -m pip install -r original-challenge/requirements.txt` and `python -m pip install pytest` for tests. Commands below run from the repository root:

```sh
python baseline-reproduction/scripts/eval.py --b1
python physics-context-detector/scripts/run_b4.py
python physics-context-detector/scripts/eval.py physics-context-detector/results/b4_trainval_predictions.csv --compare-b1
python -m pytest physics-context-detector/tests
```

Experiments read `data/full` independently of the working directory. Predictions, caches, generated stress fleets and other outputs belong inside the experiment that produced them. B4 `--out` accepts only paths within its experiment directory. Shared datasets and the local `.venv` are excluded from Git. Stress fleets are generated locally; see the B4 README.
