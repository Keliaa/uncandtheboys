# Ideas from the B4 detector (physics-context-detector/)

B4 scores 92.3 (train) / 95.7 (validation) and 86.8 on stress fleets; the pump-assist system scores 64.1 / 85.5 and 57.9. B4's log
(`../physics-context-detector/EXPERIMENTS.md`) shows where the points came from. This folder tries them one at a time, on the same
splits: settings chosen on train, validation scored once, test pumps never scored.

| # | B4 idea (log entry) | B4 gain | Status here |
|---|---|---|---|
| 1 | Alert management by episodes: one alert per developing problem per fault family, escalation types (cavitation, air) may add; episode closes after steady quiet (#3) | +7 (FA 0.57 -> 0.14) | **done**: train 64.1 -> 66.1, validation 85.5 -> 85.5, FA/day 0.33 -> 0.24 (`assist/evidence/episodes.py`, `scripts/eval_episodes.py`) |
| 2 | First alert waits until the diagnosed type is stable for N min, N tuned on train (#12) | +1.7 | todo |
| 3 | Physics signature diagnosis: steady-state model gives the expected residual vector of clog / restriction / impeller wear at the current operating point; classify by best fit vs "one instrument alone" (#4, #9) | type recall 0.69 -> 0.77, more general | todo (largest expected gain for fault-type recall: B4 0.92 vs 0.46 here) |
| 4 | Segment-reset smoothing: EWMA resets at stops so scores are not stale after repairs; thermal channels gated 1.5-2.5 h after a start (#2) | +10 (with sub-episodes) | todo |
| 5 | Context regression on physics-required speed, duty, basin level, lagged load (#1, #2); speed residual on VFD-temperature pumps exposes clogging hours earlier | type recall x2 | todo (D02 here used simpler regressors and was neutral) |
| 6 | Absolute physics for pre-existing faults with fleet-level uncertainty (#5) | +1.8 | partly (D04 pump-curve residual; PUMP-03 still missed) |
| 7 | Derived-channel-aware sensor test: a drifting suction transmitter also moves the computed head ratio (#6) | +1.7 | todo |
| 8 | Spectral checks with fixed fleet thresholds: BPFO envelope >= 4.5x, 2x >= 0.9x 1x for misalignment (#7) | +1.3 | evidence only here (V01-V03); could become alerts |
| 9 | Same-type re-alert memory 24 h (#11) | +0.3 | have 12 h per type |
| 10 | Causality test: truncate the data at random cuts and check that earlier alerts do not change (#8) | proof of causality | todo (the replay demo covers one pump) |

Things B4 tried that did not work and should not be repeated: closing episodes at every lockout restart; keeping alerted hydraulic episodes open
with continuation votes; a lower sensor chi-square margin.

Caveat from B4's own limitations: its physics model reuses the simulator's relations and profile constants, so its numbers are close to
"knowing the generator"; per-site re-commissioning would be needed on real pumps.
