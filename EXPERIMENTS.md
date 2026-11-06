# Experiments log (B4 detector)

Selection metric: mean composite over train and validation (`scripts/eval.py`). Validation has only 9 events, so
differences under ~3 points are treated as noise. Test pumps (PUMP-10/11/12) are never inspected or scored.

| # | idea | change | train comp | val comp | mean | type recall (tr / va) | kept? | why |
|---|---|---|---|---|---|---|---|---|
| 0 | B1 reference | `b1_threshold(k=6, persist=15)` as tuned by run_baselines.py | 58.2 | 78.0 | 68.1 | 0.27 / 0.44 | ref | reproduced official numbers exactly |
