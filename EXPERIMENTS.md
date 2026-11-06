# Experiments log (B4 detector)

Selection metric: mean composite over train and validation (`scripts/eval.py`). Validation has only 9 events, so
differences under ~3 points are treated as noise. Test pumps (PUMP-10/11/12) are never inspected or scored.

| # | idea | change | train comp | val comp | mean | type recall (tr / va) | kept? | why |
|---|---|---|---|---|---|---|---|---|
| 0 | B1 reference | `b1_threshold(k=6, persist=15)` as tuned by run_baselines.py | 58.2 | 78.0 | 68.1 | 0.27 / 0.44 | ref | reproduced official numbers exactly |
| 1 | B4 v1 skeleton | physics speed residual (VFD_TEMP), causal context ridge residuals, per-family detectors, stability-gated diagnosis | 65.6 | 73.3 | 69.4 | 0.58 / 0.78 | kept | type recall doubled; FA 0.39/0.71 per day |
| 2 | segment resets + sub-episodes | EWMA reset at stops (no stale scores after repairs), thermal settling gate, early-calibration scale inflation, basin level regressor, per-type sub-episodes (clog -> cavitation) | 70.9 | 88.7 | 79.8 | 0.73 / 1.00 | kept | stress 4 seeds 71.2; FA 0.57/0.36 still high |
| 3 | alert management | quiet time counted only while steadily running; machine episodes close after 12 h steady quiet; after the first alert only escalation types (cavitation, air) may alert within an episode; sensor tracks close after 4 h | 77.5 | 95.9 | 86.7 | 0.73 / 1.00 | kept | FA+excess 0.14/0.07 per day; stress 82.4 |
