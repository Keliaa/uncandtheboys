# Experiments log (B4 detector)

Selection metric: mean composite over train and validation (`scripts/eval.py`). Validation has only 9 events, so
differences under ~3 points are treated as noise. Test pumps (PUMP-10/11/12) are never inspected or scored.

| # | idea | change | train comp | val comp | mean | type recall (tr / va) | kept? | why |
|---|---|---|---|---|---|---|---|---|
| 0 | B1 reference | `b1_threshold(k=6, persist=15)` as tuned by run_baselines.py | 58.2 | 78.0 | 68.1 | 0.27 / 0.44 | ref | reproduced official numbers exactly |
| 1 | B4 v1 skeleton | physics speed residual (VFD_TEMP), causal context ridge residuals, per-family detectors, stability-gated diagnosis | 65.6 | 73.3 | 69.4 | 0.58 / 0.78 | kept | type recall doubled; FA 0.39/0.71 per day |
| 2 | segment resets + sub-episodes | EWMA reset at stops (no stale scores after repairs), thermal settling gate, early-calibration scale inflation, basin level regressor, per-type sub-episodes (clog -> cavitation) | 70.9 | 88.7 | 79.8 | 0.73 / 1.00 | kept | stress 4 seeds 71.2; FA 0.57/0.36 still high |
| 3 | alert management | quiet time counted only while steadily running; machine episodes close after 12 h steady quiet; after the first alert only escalation types (cavitation, air) may alert within an episode; sensor tracks close after 4 h | 77.5 | 95.9 | 86.7 | 0.73 / 1.00 | kept | FA+excess 0.14/0.07 per day; stress 82.4 |
| 4 | physics signature diagnosis | steady-state physics model per asset gives expected residual signature of clog / restriction / impeller wear at the current operating point; hydraulic residual vector classified by best fit vs single-instrument explanation (chi2 margin 4); escalation (cavitation/air) diagnosed on the change on top of an ongoing fault; timing-based drift vs step; learned natural constant-run length for wireless stuck detection; no single-channel drift calls on winding/fluid temperature | 78.9 | 94.1 | 86.5 | 0.69 / 0.89 | kept | stress 83.7 (v3: 82.4), FA 0.12/day; more general than hand rules |
| 5 | pre-existing faults (absolute physics) | measured vs steady-state physics (assets.csv curve, documented control law) with fleet-level uncertainty (fixed speed: current 1.2 %, pressure 1.2 %; VFD 2.5-4 %), classified with the same fault signatures, own episode track | 82.7 | 94.1 | 88.4 | 0.77 / 0.89 | kept | catches pre-existing impeller wear on PUMP-03/05 at start; no new false alerts on 32 stress pumps |
| 6 | derived-channel aware sensor test | single-instrument hypothesis for pd/ps/q includes its implied change of the derived head ratio | 86.1 | 94.1 | 90.1 | 0.81 / 0.89 | kept | catches PUMP-02 suction drift; stress 85.1 |
