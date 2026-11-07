# B4 results: physics- and context-aware causal detector

B4 beats baseline B1 on every composite component, on train, on validation, and on 32 unseen simulated pumps.
All numbers come from the official scorer (`../original-challenge/src/retrofit_sim/scoring.py`). The B4 parameters were tuned on train.
Validation and the stress pumps were used only as checks.

## Reproduce (from this experiment directory)

```bash
python -m pip install -r ../original-challenge/requirements.txt
# Activate the repository virtualenv first, then cd physics-context-detector
python scripts/run_b4.py --final        # train+val predictions, scores vs B1, frozen test predictions
python scripts/eval.py results/b4_trainval_predictions.csv --compare-b1 --events-table --lopo
python -m pytest -q tests/test_b4_causality.py
python scripts/tier_ablation.py
```

Windows PowerShell, from this experiment directory after the [repository Windows setup](../README.md#windows-powershell):

```powershell
..\.venv\Scripts\python.exe .\scripts\run_b4.py --final
..\.venv\Scripts\python.exe .\scripts\eval.py .\results\b4_trainval_predictions.csv --compare-b1 --events-table --lopo
..\.venv\Scripts\python.exe -m pytest -q .\tests\test_b4_causality.py
..\.venv\Scripts\python.exe .\scripts\tier_ablation.py
```

The code and outputs are in these files:

* Code: `b4.py` and `scripts/run_b4.py`.
* Harness: `scripts/eval.py`.
* Causality test: `tests/test_b4_causality.py`.
* Submission files:
  * `results/b4_validation_predictions.csv`.
  * `results/b4_test_predictions.csv` (26 alerts, written once with the frozen parameters in `results/b4_frozen_params.json`).
* Development helpers: `scripts/sweep.py`, `scripts/diag.py` and `scripts/stress_events.py`.
* Experiment log, one row per experiment: `EXPERIMENTS.md`.

`baselines.py`, `scoring.py` and the data were not modified. The test pumps' telemetry was read only by the final
`--final` run. No test events or alerts were inspected, and nothing was tuned on them.

## B1 vs B4

| method | split | composite | F1 | recall | precision | timeliness | fault-type recall | class recall | false+excess / asset-day |
|---|---|---|---|---|---|---|---|---|---|
| B1 | train | 58.2 | 0.720 | 0.692 | 0.750 | 0.577 | 0.269 | 0.538 | 0.122 |
| B1 | validation | 78.0 | 0.889 | 0.889 | 0.889 | 0.863 | 0.444 | 0.889 | 0.071 |
| **B4** | train | **92.3** | 0.909 | 0.962 | 0.862 | 0.911 | 0.923 | 0.962 | 0.082 |
| **B4** | validation | **95.7** | 0.947 | 1.000 | 0.900 | 0.897 | 1.000 | 1.000 | 0.071 |

The mean composite over train and validation is 68.1 for B1 and **94.0** for B4.

B4 was also checked in three more ways:

* **Stress pumps.** These are 4 stress fleets (32 random pumps with unseen ratings, seasons and missing sensors, 94 labelled
  events), generated with `../original-challenge/make_dataset.py --tier stress --seed 101..104 --stress-assets 8 --days 7` and scored with
  the full private labels.

  | method | stress composite | fault-type recall | recall |
  |---|---|---|---|
  | B1 | 52.3 | 0.23 | 0.58 |
  | B4 | **86.8** | 0.75 | 0.91 |

* **Leave-one-pump-out** (choice of the stability window N). Each of the 9 labelled pumps was held out in turn and N was
  chosen on the other 8. N = 20 min was chosen for 8 of the 9 pumps, and the mean held-out composite was 93.3.
* **Validation has only 9 events.** Treat the validation number as indicative. The train and stress results are the
  stronger evidence.

## Per-event table (train + validation)

| split | event | true type | detected | predicted type | lead to deadline (h) | delay after detectable (h) |
|---|---|---|---|---|---|---|
| train | PUMP-01-E01 | SUCTION_STRAINER_CLOGGING | yes | SUCTION_STRAINER_CLOGGING | - | 16.1 |
| train | PUMP-01-E03 | COMMUNICATION_LOSS | yes | COMMUNICATION_LOSS | - | 0.1 |
| train | PUMP-01-E02 | SUCTION_STRAINER_CLOGGING | yes | SUCTION_STRAINER_CLOGGING | - | 10.0 |
| train | PUMP-01-E04 | SENSOR_STUCK | yes | SENSOR_STUCK | - | 0.5 |
| train | PUMP-02-E01 | BEARING_WEAR (pre-existing) | yes | BEARING_WEAR | - | 0.6 |
| train | PUMP-02-E03 | SENSOR_DRIFT | yes | SENSOR_DRIFT | - | 22.1 |
| train | PUMP-02-E02 | MISALIGNMENT | yes | MISALIGNMENT | - | 1.6 |
| train | PUMP-03-E03 | IMPELLER_WEAR (pre-existing) | yes | IMPELLER_WEAR | - | 0.6 |
| train | PUMP-03-E01 | DISCHARGE_RESTRICTION | yes | DISCHARGE_RESTRICTION | -0.5 | 0.7 |
| train | PUMP-03-E02 | SENSOR_CALIBRATION_SHIFT | yes | SENSOR_CALIBRATION_SHIFT | - | 7.5 |
| train | PUMP-04-E02 | BEARING_WEAR (pre-existing) | yes | BEARING_WEAR | 119.6 | 0.6 |
| train | PUMP-04-E01 | IMBALANCE | yes | IMBALANCE | 71.9 | 24.3 |
| train | PUMP-04-E03 | SENSOR_NOISE_SPIKES | yes | SENSOR_NOISE_SPIKES | - | 0.9 |
| train | PUMP-04-E04 | DISCHARGE_RESTRICTION | yes | DISCHARGE_RESTRICTION | 1.9 | 1.1 |
| train | PUMP-05-E01 | IMPELLER_WEAR (pre-existing) | yes | IMPELLER_WEAR | - | 0.6 |
| train | PUMP-05-E02 | SENSOR_STUCK | yes | SENSOR_STUCK | - | 0.5 |
| train | PUMP-06-E02 | MISALIGNMENT | yes | MISALIGNMENT | - | 1.2 |
| train | PUMP-06-E03 | SUCTION_STRAINER_CLOGGING | yes | SUCTION_STRAINER_CLOGGING | 20.6 | 10.7 |
| train | PUMP-06-E05 | CAVITATION | yes | CAVITATION | -0.2 | 0.1 |
| train | PUMP-06-E06 | CAVITATION | yes | CAVITATION | -0.3 | 0.2 |
| train | PUMP-06-E07 | CAVITATION | yes | CAVITATION | -0.3 | 0.2 |
| train | PUMP-06-E08 | CAVITATION | yes | CAVITATION | -0.3 | 0.2 |
| train | PUMP-06-E04 | SENSOR_DRIFT | yes | SENSOR_CALIBRATION_SHIFT (class right) | - | 4.3 |
| train | PUMP-07-E01 | BEARING_WEAR | yes | BEARING_WEAR | - | 6.5 |
| train | PUMP-07-E02 | SENSOR_CALIBRATION_SHIFT | yes | SENSOR_CALIBRATION_SHIFT | - | 0.4 |
| train | PUMP-07-E03 | AIR_ENTRAINMENT | **no** | - | - | - |
| val | PUMP-08-E02 | BEARING_WEAR (pre-existing) | yes | BEARING_WEAR | - | 20.9 |
| val | PUMP-08-E01 | SUCTION_STRAINER_CLOGGING | yes | SUCTION_STRAINER_CLOGGING | 20.5 | 16.5 |
| val | PUMP-08-E03 | COMMUNICATION_LOSS | yes | COMMUNICATION_LOSS | - | 0.1 |
| val | PUMP-08-E04 | SENSOR_NOISE_SPIKES | yes | SENSOR_NOISE_SPIKES | - | 0.4 |
| val | PUMP-09-E04 | SENSOR_DRIFT | yes | SENSOR_DRIFT | - | 38.8 |
| val | PUMP-09-E01 | MISALIGNMENT | yes | MISALIGNMENT | - | 0.7 |
| val | PUMP-09-E03 | SENSOR_STUCK | yes | SENSOR_STUCK | - | 2.1 |
| val | PUMP-09-E02 | DISCHARGE_RESTRICTION | yes | DISCHARGE_RESTRICTION | -0.1 | 0.7 |
| val | PUMP-09-E06 | AIR_ENTRAINMENT | yes | AIR_ENTRAINMENT | - | 0.5 |

B4 detects 34 of 35 events, and 33 of the 35 detecting alerts name the correct fault type. The negative lead times
(restriction and cavitation events) mean the conventional limit was reached within minutes of onset. Those events
still score full or near-full timeliness, because the alert came well before the end of the event window.

## Tier ablation (T2 pumps PUMP-02, 04, 07, 09; channels removed)

| simulated tier | composite | recall | fault-type recall | false+excess / asset-day |
|---|---|---|---|---|
| T2 full | 92.5 | 0.93 | 0.93 | 0.04 |
| T1 retrofit (no flow / winding / valve; wireless vibration every 5 min) | 83.5 | 0.87 | 0.73 | 0.07 |
| T0 legacy (current, discharge pressure, temperatures; + drive speed/power) | 42.1 | 0.33 | 0.27 | 0.04 |

The labels stay those of the T2 pumps, so the T0 run is also scored on bearing, misalignment and imbalance events that
no T0 instrument can reveal. The T0 number is therefore pessimistic.

The retrofit message is still clear. A low-cost wireless vibration sensor plus a suction transmitter (T1) recovers most
of the T2 performance. Without vibration (T0), the detector is limited to hydraulic and instrument faults.

## How it works

Every step is causal: an alert at time t uses only rows stamped <= t, plus `assets.csv`.

1. **Physics expectations.** The physics come from the pump curve and system curve in `assets.csv`, the documented control
   laws, and the profile constants from the public configuration. Two uses follow:
   * On VFD temperature-controlled pumps, the speed required by demand and sea temperature matches the measured speed
     within about 0.5 %. The speed residual therefore exposes clogging and restriction hours earlier than speed-band
     baselines.
   * Where flow is metered, the head ratio against the pump curve exposes impeller wear and air.
2. **Context model.** Each channel gets a small ridge regression on exogenous context:
   * Hydraulic channels use the physics-required speed, duty demand and basin level.
   * Temperature channels use lagged load, ambient and supply temperature.

   The regression is fitted on the pump's own steady minutes, expanding during the first 24 h and then frozen. Early
   fits are deliberately made more uncertain (wider residual scale), and thermal channels are gated for 1.5â€“2.5 h
   after each start.
3. **Diagnosis from fault signatures.** A steady-state physics model computes, at the current operating point, how
   each channel would move under a clog, a restriction or impeller wear (small and large magnitudes, because valves and
   drives saturate). The observed residual vector is classified by its best-fitting signature, or as one instrument
   alone. A single-instrument hypothesis must beat the best machine fault by a chiÂ² margin. It also accounts for derived
   channels: a drifting suction transmitter also moves the computed head.
4. **Pre-existing faults.** The pump is compared against absolute physics with fleet-level uncertainty: about 1.2 % for
   current on fixed-speed pumps, wider for VFD pumps. This catches the impeller wear present from the first hour on
   PUMP-03 and PUMP-05.
5. **Vibration.** The bearing track uses acceleration and crest factor, with a fleet crest prior. The vibration
   snapshots add three spectral checks:
   * BPFO envelope peak at least 4.5Ã— the noise floor: bearing defect. Healthy snapshots stay below 3.8 (524 checked).
   * 2Ã— line at least 0.55 mm/s and at least 0.9Ã— the 1Ã— line: misalignment.
   * Dominant 1Ã— line: imbalance.
6. **Alert management.** One alert is raised per developing problem, per detector:
   * The first alert waits until the diagnosed type has been stable for 20 min.
   * Once an episode has alerted, only cavitation or air can raise a new alert in it.
   * An episode ends only after 12 h of steady-running quiet.
   * The same type is not re-alerted within 24 h (12 h for instrument faults).
   * Alerts are suppressed during start/stop and settling.
   * Alerts on one asset are at least 6 min apart, so a scoring burst never holds two types.

Every alert carries evidence text with the deviation in engineering units, the operating context and, where available,
the snapshot finding. It also carries a planner-level recommended action. Instrument faults explicitly say "verify the
transmitter before acting on the machine".

## What worked

These are the gains in mean composite (train+validation) recorded in `EXPERIMENTS.md`:

| change | from â†’ to |
|---|---|
| Stable-type diagnosis and family detectors | 68.1 â†’ 69.4 (fault-type recall 0.27 â†’ 0.58) |
| Segment-reset smoothing, so no stale scores after repairs | â†’ 79.8 |
| Alert management (count quiet only while running; escalation-only) | â†’ 86.7 |
| Absolute physics for pre-existing faults | â†’ 88.4 |
| Derived-channel aware sensor test | â†’ 90.1 |
| Snapshot spectra | â†’ 91.4 |
| Drift timing over running minutes, large-fault signatures | â†’ 92.1 |
| Re-alert memory and stability window N = 20 | â†’ 94.0 |

## What did not work (reverted)

* **Closing machine episodes at every lockout restart** (mean 89.8 vs 91.4). Routine lockouts re-trigger alerts for
  faults that are still present.
* **Keeping an alerted hydraulic episode alive with continuation votes** (89.6). It suppressed genuine new detections.
* **A lower sensor chiÂ² margin.** It helps stress slightly but hurts train. 4 was kept.

## Honest limitations

* **Synthetic data only.** The physics model reuses the simulator's own relations and profile constants. That is
  allowed, but it is close to "knowing the generator". Real pumps will show larger model errors, so the
  absolute-physics and snapshot thresholds would need re-commissioning per site.
* **Profile assumptions.** The test pumps run on a private configuration. If a test profile differs, for example in
  design sea temperature, valve share or suction head, the speed residual and absolute checks could carry a bias. The
  speed bias is learned causally but clipped to a plausible envelope.
* **Weak spots on the stress pumps:**
  * sensor drift vs calibration step on legacy pumps with only current and discharge pressure;
  * restriction on flow-valve pumps without valve feedback;
  * clog vs impeller wear on T0 fixed-speed pumps.
* **Unknown sensor-fault channels.** No single-channel drift calls are made on winding or fluid temperature (to avoid
  false alerts). Drifts there would be missed.
* **Confidence is not calibrated.** The `confidence` column is a fixed per-detector value, raised when snapshot
  evidence agrees. Backlog item 10 (isotonic calibration) was not done.
* **Environment.** The work ran on Python 3.9 (macOS system Python). The package metadata asks for â‰¥ 3.10, so it was
  run with `PYTHONPATH=src` instead of `pip install -e .`. The code itself uses `from __future__ import annotations`
  and runs on 3.9â€“3.13.

## Three most promising next steps

1. **Learn the diagnosis on simulator data.** Generate a few hundred stress pumps and train a small classifier on the
   residual and signature features, with the physics rules as fallback. Stress type recall (0.75) is the main remaining
   gap.
2. **Calibrated confidence and an explicit UNKNOWN.** Use isotonic regression on train + stress to map the signature-fit
   margin and snapshot agreement to probabilities. Emit UNKNOWN when the best two explanations are within the margin.
3. **Better legacy (T0) coverage.** Add current-signature and pressure-pulsation features where 1-s data exist, and a
   drift-vs-step test based on piecewise-linear fits. This should improve the T0 tier, which most retrofit candidates
   are.
