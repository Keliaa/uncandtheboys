# Pump assist

Implementation of the practical worklist ([docs/elias/practical_worklist.html](../docs/elias/practical_worklist.html)):
ingest, detect, evidence, recommend, operator app, static reports and scoring. Everything this experiment writes stays in
this folder. It reads `../data/full` and imports the read-only reference code in `../original-challenge/src`.

Splits are kept: settings are chosen on train pumps (PUMP-01 to 07), validation (08, 09) is scored once, test (10 to 12) is only
run with frozen settings and never scored or tuned. `assist/splits.py` has the guards (`require_split`, `splits=` arguments).

```
assist/      code: alert record (S01), config loader (S02), ingest (I01, Q01), detect (D01, D02, D04, D06),
             evidence (E01, E02), recommend (R01, R02), pipeline.py (frozen end-to-end system), report.py (U02)
app/         Streamlit operator app (U01)
configs/     machine file (channels, symptom table, action table, physics) and generated per-asset files
scripts/     tuning and scoring scripts (tune_*.py choose settings on train only), c01_score.py, c02_test_file.py, make_reports.py
tests/       unit tests
results/     predictions, grids, results.md (C01), reports/ (one HTML file per pump), test_predictions.csv (C02)
```

Install: `python -m pip install -r original-challenge/requirements.txt -r pump-assist/requirements.txt`. Commands run from the repository root:

```sh
python -m pytest pump-assist/tests
python pump-assist/scripts/c01_score.py          # results table with baselines B0-B3
python pump-assist/scripts/c02_test_file.py      # frozen system on PUMP-10..12
python pump-assist/scripts/make_reports.py       # open pump-assist/results/reports/index.html
streamlit run pump-assist/app/main.py
```

## Results (official scorer, composite)

| System | Train | Validation |
|---|---|---|
| B1 baseline | 58.2 | 78.0 |
| Frozen system (D01 + D04 + Q01 + D06 + E01/E02 with rule gate + R01) | 61.8 | 81.9 |

Full table: `results/results.md`. Validation has 9 events on 2 pumps, so small differences are not meaningful. Known issues: E02
probabilities are overconfident (0.98 stated vs 0.65 correct on 28 alerts); D04 raises many alerts on PUMP-12 because its
nameplate sits about 7 % below the pump curve; D02 is implemented but off in the frozen system (train +2.8, validation neutral).
