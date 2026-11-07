# Leave-one-pump-out and shadow comparison

Pumps PUMP-01 to 09 (the labelled ones). Each pump is held out in turn; B4 picks its stability window N on the other 8 pumps, v2 picks its limits
(D01 k, D04 head limit) and trains its naive Bayes model on the other 8 pumps' events. The held-out pump is scored alone with the official scorer.
B4's other parameters were tuned on the train pumps during its development, so its fold results are optimistic for pumps 01 to 07.

## Held-out composite per pump

| held_out | B4 | v2 without derived-channel diagnosis | v2 with derived-channel diagnosis |
|---|---|---|---|
| PUMP-01 | 93.4 | 60.5 | 60.5 |
| PUMP-02 | 98.7 | 37.9 | 37.9 |
| PUMP-03 | 99.7 | 69.3 | 69.3 |
| PUMP-04 | 100.0 | 57.4 | 57.4 |
| PUMP-05 | 100.0 | 100.0 | 100.0 |
| PUMP-06 | 86.2 | 44.0 | 44.0 |
| PUMP-07 | 74.5 | 90.8 | 90.8 |
| PUMP-08 | 99.2 | 93.0 | 93.0 |
| PUMP-09 | 93.0 | 79.6 | 79.6 |
| mean | 93.9 | 70.3 | 70.3 |

## Warning against the conventional alarm deadline (all 9 held-out pumps)

| system | events | detected | type_correct | events_with_deadline | alerted_before_deadline | median_lead_h | mean_timeliness |
|---|---|---|---|---|---|---|---|
| B4 | 35 | 34 | 33 | 11 | 5 | -0.1 | 0.93 |
| v2 without derived-channel diagnosis | 35 | 29 | 15 | 8 | 6 | 6.2 | 0.87 |
| v2 with derived-channel diagnosis | 35 | 29 | 15 | 8 | 6 | 6.2 | 0.87 |

`alerted_before_deadline` counts events whose first alert came before the time a conventional limit would have been reached; most events have no
deadline. Timeliness can be high even when an alert comes after the deadline, so the lead time is the figure to watch.

## Shared stress fleets (seeds 101 to 103, frozen settings, 8 unseen pumps each)

| system | composite_score | event_recall | alert_precision | false_alerts_per_asset_day | timeliness | fault_type_recall | class_recall |
|---|---|---|---|---|---|---|---|
| B1 | 52.103 | 0.567 | 0.75 | 0.113 | 0.515 | 0.246 | 0.453 |
| B4 | 87.653 | 0.912 | 0.939 | 0.024 | 0.839 | 0.79 | 0.886 |
| v2 with derived | 58.934 | 0.673 | 0.727 | 0.107 | 0.601 | 0.329 | 0.491 |
| v2 without derived | 58.674 | 0.673 | 0.727 | 0.107 | 0.601 | 0.318 | 0.491 |

Source files: `lopo_b4.csv`, `lopo_v2.csv`, `lopo_*_events.csv`, `stress_compare.csv`.
