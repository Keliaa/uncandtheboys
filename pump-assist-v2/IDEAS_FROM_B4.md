# Ideas from the B4 detector (physics-context-detector/)

**Rule for this folder: no generator values.** Nothing may use the simulator's profile constants (`fleet_public.yaml`), its control laws or its hidden relations. Only the public asset sheet, the telemetry and the data itself. B4's physics signature diagnosis, physics-required speed and absolute-physics tracks depend on those constants, so they are out; data-driven versions of the same ideas are in the table.

B4 scores 92.3 (train) / 95.7 (validation) and 86.8 on stress fleets; the pump-assist system scores 64.1 / 85.5 and 57.9. B4's log
(`../physics-context-detector/EXPERIMENTS.md`) shows where the points came from. This folder tries them one at a time, on the same
splits: settings chosen on train, validation scored once, test pumps never scored.

| # | B4 idea (log entry) | B4 gain | Status here |
|---|---|---|---|
| 1 | Alert management by episodes: one alert per developing problem per fault family, escalation types (cavitation, air) may add; episode closes after steady quiet (#3) | +7 (FA 0.57 -> 0.14) | **done**: train 64.1 -> 66.1, validation 85.5 -> 85.5, FA/day 0.33 -> 0.24 (`assist/evidence/episodes.py`, `scripts/eval_episodes.py`) |
| 2 | First alert waits until the diagnosed type is stable for N min, N tuned on train (#12) | +1.7 | **tried, not kept**: N = 5/10/20/30 gives train 64.6/64.0/63.0/62.7 and validation 80.7 (vs 66.1 / 85.5 at N = 1): waiting delays the alerts and the diagnosis changes inside a stretch (`stable_min` in `to_alerts_with_rules`) |
| 3 | (out: needs generator values; data-driven version tried, see below) Physics signature diagnosis: steady-state model gives the expected residual vector of clog / restriction / impeller wear at the current operating point; classify by best fit vs "one instrument alone" (#4, #9) | type recall 0.69 -> 0.77, more general | todo (largest expected gain for fault-type recall: B4 0.92 vs 0.46 here) |
| 4 | Segment-reset smoothing: EWMA resets at stops so scores are not stale after repairs; thermal channels gated 1.5-2.5 h after a start (#2) | +10 (with sub-episodes) | **tried (data-driven EWMA of the z-scores, reset at every non-steady minute), not kept**: spans 10/30/60 with limit 6 -> train 67.0, validation 83.7 (vs 66.1 / 85.5), false alerts up (`assist/detect/smoothing.py`, `ewma=` in the pipeline) |
| 5 | (partly out) Context regression on physics-required speed, duty, basin level, lagged load (#1, #2) | type recall x2 | the physics-required speed is out; D02 here used duty, temperatures and speed and was neutral |
| 6 | Absolute physics for pre-existing faults with fleet-level uncertainty (#5) | +1.8 | partly (D04 pump-curve residual; PUMP-03 still missed) |
| 7 | Derived-channel-aware sensor test: a drifting suction transmitter also moves the computed head ratio (#6) | +1.7 | **done, on by default**: a head-residual (D04) alert that one pressure transmitter explains, with current, speed, power, flow and the other pressure normal, becomes a SENSOR alert (`assist/evidence/derived.py`). Dev scores unchanged (train 66.1, validation 85.5; leave-one-pump-out mean 70.3 either way), stress +0.3; on PUMP-02 it replaces four wrong IMPELLER_WEAR alerts by the right SENSOR_DRIFT |
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

## Learned fault signatures (data-driven stand-in for B4's physics signatures)

`assist/evidence/signature.py` (cosine prototypes, kNN, logistic regression on signed-log z vectors) against the E02 naive Bayes, all trained
on labelled train events only, leave-one-pump-out for train alerts (`scripts/eval_signatures.py`, `results/signatures.csv`). 'flagged' = the E02
rows (|z| > 8, five fault types); 'relaxed' = steady minutes inside events with |z| > 4, overlaps given to the event that started last,
pre-existing events lowest priority (all eight machine types, 2896 rows).

| training rows | model | train | validation | type recall train / val |
|---|---|---|---|---|
| flagged | naive Bayes (current E02) | 61.0 | 85.5 | 0.38 / 0.67 |
| flagged | cosine / kNN / logreg | 59.5 / 57.7 / 59.5 | 80.7 / 80.7 / 80.7 | 0.38 / 0.35 / 0.38 |
| relaxed | naive Bayes | 62.9 | 80.7 | 0.42 / 0.56 |
| relaxed | cosine | 60.5 | 83.4 | 0.42 / 0.67 |
| relaxed | kNN | 60.0 | 80.7 | 0.38 / 0.56 |
| relaxed | logreg | 61.5 | 83.4 | 0.42 / 0.67 |

None clearly beats the current model, so none is adopted. Covering all eight types adds one correctly typed train event (type recall 0.38 -> 0.42)
but validation does not follow. The limit is data: one to four labelled events per type, from one to three pumps, so a model learns pump-specific
patterns. B4's physics signatures avoid this because they come from knowledge of the generator, which is what this folder does not use.

## Next data-driven ideas

Learn the fault signatures from the labelled train events instead of computing them from physics (kNN or logistic regression on residual vectors), learn which channels move together from each pump's own healthy data (instead of the simulator's coupling lists) for the sensor-versus-machine test, calibrate the probabilities, and a leave-one-site-out check of every adopted idea.

## Leave-one-pump-out, shadow comparison and what it says (see results/LOPO_REPORT.md)

Leave-one-pump-out on the nine labelled pumps (B4 picks N on the other eight; v2 picks its limits and trains the naive Bayes model on the other eight):
B4 93.9, v2 70.3 (identical with and without the derived-channel diagnosis). On three shared stress fleets: B4 87.7, v2 58.9 (58.7 without), B1 52.1.
Fault-type recall is the gap: 33 of 35 events typed correctly by B4, 15 of 35 by v2.

* **Derived-channel diagnosis first** (done): correct attribution at no cost to the scores.
* **Qualify diagnoses with operating context** (not done): v2 already reports UNKNOWN when certainty is Low. The next step is an explicit
  "ambiguous between X and Y" when the top two probabilities are close and the available sensors cannot separate them (for example clogging vs
  impeller wear on a pump with no flow meter). The signature experiment (cosine, kNN, logistic regression all behind naive Bayes on validation)
  says to improve the evidence, not the classifier.
* **Shadow mode** (partly done): the stress-fleet comparison is done. Warning time against the conventional deadline: on the 8 to 11 events that
  have one, B4 alerts on average at the limit (median lead -0.1 h, 5 of 11 before it) while v2 warns earlier (median +6.2 h, 6 of 8 before it) but finds
  fewer events (29 of 35 against 34). Still to do: replay real telemetry with `scripts/c03_demo.py` against `maintenance.csv` and operator review.
