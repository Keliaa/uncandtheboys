# Pump assist v2 (ideas from the B4 detector)

Starting point: a copy of `../pump-assist` at its final commit on `exp/practical-worklist` (frozen system 64.1 train / 85.5 validation). This folder tries ideas from `../physics-context-detector` one at a time; see [IDEAS_FROM_B4.md](IDEAS_FROM_B4.md). Run commands from the repository root with `pump-assist-v2/` instead of `pump-assist/`. Results below this heading describe the original system.

# Pump assist

Implementation of the practical worklist ([docs/elias/practical_worklist.html](../docs/elias/practical_worklist.html)):
ingest, detect, evidence, recommend, operator app, static reports and scoring. Everything this experiment writes stays in
this folder. It reads `../data/full` and imports the read-only reference code in `../original-challenge/src`.

Splits are kept: settings are chosen on train pumps (PUMP-01 to 07), validation (08, 09) is scored once, test (10 to 12) is only
run with frozen settings and never scored or tuned. `assist/splits.py` has the guards (`require_split`, `splits=` arguments).

```
assist/      code: alert record (S01), config loader (S02), ingest (I01, I02 replay, Q01), detect (D01, D02, D03, D04, D06, V01, V02, V03, T01 baseline variants, T02 after-maintenance steps, E03 sensor-vs-machine; E04 tree for comparison; D08 change times; D05 mode clusters and D07 autoencoder, not adopted),
             evidence (E01, E02), recommend (R01, R02), pipeline.py (frozen end-to-end system), report.py (U02)
app/         Streamlit operator app (U01) with a replay-demo mode (C03)
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
# C03 demo (replay of PUMP-08 -> alerts -> scripted decisions, then choose "Replay demo" in the app): python pump-assist/scripts/c03_demo.py
# T03: python original-challenge/make_dataset.py --tier stress --seed 101 --stress-assets 8 --days 7 --output pump-assist/build/stress_101 (seeds 101-103), then pump-assist/scripts/eval_t03.py
streamlit run pump-assist/app/main.py
```

## Results (official scorer, composite)

| System | Train | Validation |
|---|---|---|
| B1 baseline | 58.2 | 78.0 |
| Frozen system (D01 + D04 + Q01 + D06 + E01/E02 with rule gate + R01 + T02) | 63.1 | 85.5 |

Full table: `results/results.md`. Stress fleets (T03, seeds 101-103, mean 57.9 vs B1 52.1): `results/t03_stress.md`. Fewer sensors (T04): `results/t04_tiers.csv`, blind spots in `results/t04_blind_spots.md`. Leave-one-site-out transfer test (T05): `results/t05_sites.md` (held-out mean 56.2 vs 67.3 when the site was seen, B1 55.1). Validation has 9 events on 2 pumps, so small differences are not meaningful. Known issues: some alerts show evidence that belongs to a concurrent fault (PUMP-08 on 11 Sep is typed BEARING_WEAR, which is a real labelled event, while the listed evidence is the strainer clogging). Two attempted fixes lowered validation, so they are not in the system: vetoing BEARING_WEAR when the snapshot and envelope show no defect (train 64.1 -> 61.2, validation 85.5 -> 80.7, bearing events typed correctly 2/4 -> 0/4, snapshots lag the onset) and requiring the strongest-deviating channel to belong to the type (validation 85.5 -> 80.7). E02
probabilities are overconfident (0.98 stated vs 0.65 correct on 28 alerts); D04 raises many alerts on PUMP-12 because its
nameplate sits about 7 % below the pump curve; D02 is implemented but off in the frozen system (train +2.8, validation neutral).
