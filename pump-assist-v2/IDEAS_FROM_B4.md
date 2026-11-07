# Ideas from the B4 detector (physics-context-detector/)

**Rule for this folder: no generator values.** Nothing may use the simulator's profile constants (`fleet_public.yaml`), its control laws or its hidden relations. Only the public asset sheet, the telemetry and the data itself. B4's physics signature diagnosis, physics-required speed and absolute-physics tracks depend on those constants, so they are out; data-driven versions of the same ideas are in the table.

B4 scores 92.3 (train) / 95.7 (validation) and 86.8 on stress fleets; the pump-assist system scores 64.1 / 85.5 and 57.9. B4's log
(`../physics-context-detector/EXPERIMENTS.md`) shows where the points came from. This folder tries them one at a time, on the same
splits: settings chosen on train, validation scored once, test pumps never scored.

| # | B4 idea (log entry) | B4 gain | Status here |
|---|---|---|---|
| 1 | Alert management by episodes: one alert per developing problem per fault family, escalation types (cavitation, air) may add; episode closes after steady quiet (#3) | +7 (FA 0.57 -> 0.14) | **done**: train 64.1 -> 66.1, validation 85.5 -> 85.5, FA/day 0.33 -> 0.24 (`assist/evidence/episodes.py`, `scripts/eval_episodes.py`) |
| 2 | First alert waits until the diagnosed type is stable for N min, N tuned on train (#12) | +1.7 | **tried, not kept**: N = 5/10/20/30 gives train 64.6/64.0/63.0/62.7 and validation 80.7 (vs 66.1 / 85.5 at N = 1): waiting delays the alerts and the diagnosis changes inside a stretch (`stable_min` in `to_alerts_with_rules`) |
| 3 | (out: needs generator values) Physics signature diagnosis: steady-state model gives the expected residual vector of clog / restriction / impeller wear at the current operating point; classify by best fit vs "one instrument alone" (#4, #9) | type recall 0.69 -> 0.77, more general | todo (largest expected gain for fault-type recall: B4 0.92 vs 0.46 here) |
| 4 | Segment-reset smoothing: EWMA resets at stops so scores are not stale after repairs; thermal channels gated 1.5-2.5 h after a start (#2) | +10 (with sub-episodes) | **tried (data-driven EWMA of the z-scores, reset at every non-steady minute), not kept**: spans 10/30/60 with limit 6 -> train 67.0, validation 83.7 (vs 66.1 / 85.5), false alerts up (`assist/detect/smoothing.py`, `ewma=` in the pipeline) |
| 5 | (partly out) Context regression on physics-required speed, duty, basin level, lagged load (#1, #2) | type recall x2 | the physics-required speed is out; D02 here used duty, temperatures and speed and was neutral |
| 6 | Absolute physics for pre-existing faults with fleet-level uncertainty (#5) | +1.8 | partly (D04 pump-curve residual; PUMP-03 still missed) |
| 7 | Derived-channel-aware sensor test: a drifting suction transmitter also moves the computed head ratio (#6) | +1.7 | todo |
| 8 | Spectral checks with fixed fleet thresholds: BPFO envelope >= 4.5x, 2x >= 0.9x 1x for misalignment (#7) | +1.3 | evidence only here (V01-V03); could become alerts |
| 9 | Same-type re-alert memory 24 h (#11) | +0.3 | have 12 h per type |
| 10 | Causality test: truncate the data at random cuts and check that earlier alerts do not change (#8) | proof of causality | **done and passing**: three cuts on PUMP-08, alerts before each cut identical to the full run (`tests/test_causality.py`) |

Things B4 tried that did not work and should not be repeated: closing episodes at every lockout restart; keeping alerted hydraulic episodes open
with continuation votes; a lower sensor chi-square margin.

Caveat from B4's own limitations: its physics model reuses the simulator's relations and profile constants, so its numbers are close to
"knowing the generator"; per-site re-commissioning would be needed on real pumps.

## Generator values removed from the pump-assist code in this folder

* The 120 kPa suction pressure assumed for the pressurised fresh-water loop came from `fleet_public.yaml`: removed (no suction transmitter now means 0 kPa gauge). Scores unchanged (train 66.1, validation 85.5).
* Seawater salinity is read from the public asset sheet text ("~6 PSU") instead of a constant.
* The what-if sandbox (R02) builds its model from simulator profile defaults, so it is off by default here (`with_what_if=True` turns it on). Detection and diagnosis do not use it.
* Still used: the nameplate pump curve and design point from `assets.csv` (D04), bearing orders and vane count from `assets.csv`, and the scorer/baseline helper functions from the original package.

## Next data-driven ideas

Learn the fault signatures from the labelled train events instead of computing them from physics (kNN or logistic regression on residual vectors), learn which channels move together from each pump's own healthy data (instead of the simulator's coupling lists) for the sensor-versus-machine test, calibrate the probabilities, and a leave-one-site-out check of every adopted idea.
