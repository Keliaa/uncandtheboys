# Physics/context detector (B4)

Causal physics/context detector developed from commits `e39105d` through `aab874a`. Its incremental variants and rejected trials are recorded in [EXPERIMENTS.md](EXPERIMENTS.md); they are development stages of this detector. [RESULTS.md](RESULTS.md) contains scores and analysis. The implementation is `b4.py`; the unchanged simulator/scorer is imported from `../original-challenge/src`.

From the repository root:

```sh
python physics-context-detector/scripts/run_b4.py
python physics-context-detector/scripts/eval.py physics-context-detector/results/b4_trainval_predictions.csv --compare-b1 --events-table --lopo
python -m pytest physics-context-detector/tests
python physics-context-detector/scripts/tier_ablation.py
```

`--final` also generates test predictions using frozen parameters. Inputs default to `../data/full` (read only); outputs default to this experiment's `results/`. Custom `--out` paths must resolve inside this experiment. The local evaluator keeps its own B1 comparison cache.

To create optional stress fleets, run from this directory (repeat for seeds 101â€“104):

```sh
python ../original-challenge/make_dataset.py --tier stress --seed 101 --stress-assets 8 --days 7 --output build/stress_101
python scripts/run_b4.py --stress 101
```

Generated stress inputs and predictions stay in this experiment's `build/`. The shared `data/` directory is never an output target.
