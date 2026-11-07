# Baseline reproduction

B1 reference evaluation introduced in commit `ae14121`. The official B1–B3 reproduction outputs are in `results/`; `b1_trainval_predictions.csv` is the B1 train/validation cache. Original baseline code is imported from `../original-challenge/src` without modification.

From the repository root:

```sh
python baseline-reproduction/scripts/eval.py --b1 --events-table --lopo
```

Windows PowerShell, from the repository root after the [Windows setup](../README.md#windows-powershell):

```powershell
.\.venv\Scripts\python.exe .\baseline-reproduction\scripts\eval.py --b1 --events-table --lopo
```

Input: `../data/full` (read only). All generated files stay in this experiment's `results/` directory. The evaluator is local to this experiment so it does not depend on the B4 experiment.
