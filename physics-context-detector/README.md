# Physics/context detector (B4)

An isolated, opt-in [learned diagnosis experiment](LEARNED_DIAGNOSIS.md) trains on
simulated stress fleets and relabels B4 alerts without changing detection. Its
runner and outputs are separate from the frozen results below.

Causal physics/context detector developed from commits `e39105d` through `aab874a`. Its incremental variants and rejected trials are recorded in [EXPERIMENTS.md](EXPERIMENTS.md); they are development stages of this detector. [RESULTS.md](RESULTS.md) contains scores and analysis. The implementation is `b4.py`; the unchanged simulator/scorer is imported from `../original-challenge/src`.

From the repository root:

```sh
python physics-context-detector/scripts/run_b4.py
python physics-context-detector/scripts/eval.py physics-context-detector/results/b4_trainval_predictions.csv --compare-b1 --events-table --lopo
python -m pytest physics-context-detector/tests
python physics-context-detector/scripts/tier_ablation.py
```

Windows PowerShell, from the repository root after the [Windows setup](../README.md#windows-powershell):

```powershell
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\run_b4.py
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\eval.py .\physics-context-detector\results\b4_trainval_predictions.csv --compare-b1 --events-table --lopo
.\.venv\Scripts\python.exe -m pytest .\physics-context-detector\tests
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\tier_ablation.py
```

`--final` also generates test predictions using frozen parameters. Inputs default to `../data/full` (read only); outputs default to this experiment's `results/`. Custom `--out` paths must resolve inside this experiment. The local evaluator keeps its own B1 comparison cache.

To create optional stress fleets, run from this directory (repeat for seeds 101â€“104):

```sh
python ../original-challenge/make_dataset.py --tier stress --seed 101 --stress-assets 8 --days 7 --output build/stress_101
python scripts/run_b4.py --stress 101
```

Windows PowerShell, starting from the repository root (repeat for seeds 101–104):

```powershell
cd .\physics-context-detector
..\.venv\Scripts\python.exe ..\original-challenge\make_dataset.py --tier stress --seed 101 --stress-assets 8 --days 7 --output build/stress_101
..\.venv\Scripts\python.exe .\scripts\run_b4.py --stress 101
```

Generated stress inputs and predictions stay in this experiment's `build/`. The shared `data/` directory is never an output target.
