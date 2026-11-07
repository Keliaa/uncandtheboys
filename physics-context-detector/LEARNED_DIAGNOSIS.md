# Opt-in learned B4 diagnosis experiment

The separate runner relabels B4 alerts after detection. It preserves alert count,
publication time, asset, event class and sensor attribution. B4, its parameters,
episode tracking, frozen predictions, official scorer, simulator and operator demo
are unchanged. Accepted changes use B4 actions and fresh measured residual evidence.
Their confidence is **0.6, fixed and uncalibrated**; forest scores are not calibrated
probabilities.

Run from the repository root (`Uncandtheboys`) with repository dependencies installed:

```powershell
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\learned_experiment.py all
.\.venv\Scripts\python.exe -m pytest .\physics-context-detector\tests -q
```

Equivalent resumable stages:

```powershell
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\learned_experiment.py prepare --start 201 --end 240
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\learned_experiment.py train
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\learned_experiment.py prepare --start 241 --end 250
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\learned_experiment.py evaluate
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\learned_experiment.py regression
```

To reuse only the eight already completed fleets without generating more data:

```powershell
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\learned_experiment.py pilot
```

The pilot fixes seeds 201–205 for training (40 pumps), 206 for selection (8 pumps),
and 207–208 for holdout (16 pumps), before fitting or comparing outcomes. It saves
the plan, model, freeze manifest, predictions, diagnostics and reports separately in
`build/learned/pilot_8/`. It retains the ten-distinct-event class requirement and
all acceptance gates. It does not generate fleets or run additional historical
simulations. The holdout caches were already prepared and inspected only for
aggregate cache sanity checks before this reduced split was designated; the pilot
is a small exploratory evaluation, not the original 50-fleet acceptance study.

The sequential generator creates 50 eight-pump, seven-day fleets with snapshots.
Seeds 201–230 fit preprocessing and two random forests (200 trees, depth 8, leaf
minimum 5, balanced class weights, seed 2026). Seeds 231–240 select from twelve
score/margin gates. Seeds 241–250 are prepared and evaluated only after freezing.
No gate that improves official fault-type recall while preserving composite,
event recall and false-plus-excess rate means fallback-only operation. Retraining
over an existing frozen artifact is refused. Do not remove the freeze or rerun
selection in response to holdout results.

All data and outputs live in ignored `physics-context-detector/build/learned/`:

* `stress_<seed>/generation.json`: resumable generation settings and simulator
  source hashes; incompatible existing fleets are refused.
* `stress_<seed>/baseline.csv`, `features.parquet`, `targets.csv`, `features.json`:
  B4 alerts, allowlisted features, target exclusions/event weights and cache hashes.
* `model.pkl`, `freeze.json`: preprocessing, ordered schema, supported classes,
  distinct-event counts, supported control/tier pairs, library/source versions,
  input manifest, all selection gate results and frozen model SHA-256.
* `stress_<seed>/learned_predictions.csv`, `diagnostics.csv`: separate predictions
  and original diagnosis, scores, acceptance and fallback reasons.
* `REPORT.md`, `holdout_report.json`, `per_seed.csv`, `control_tier.csv`,
  `*_confusion.csv`, `matched_alerts.csv`: unchanged official-scoring results,
  paired fleet bootstrap intervals (2,000 resamples), override/fallback coverage,
  and unambiguous matched-alert accuracy independent of official matching.
* `regression_report.json`, `official_trainval_*.csv`: post-freeze historical
  seeds 101–104 and official train/validation results, including frozen B4 CSV
  equality. These results do not feed model or gate selection.

Run the explicitly opt-in model on development inputs:

```powershell
.\.venv\Scripts\python.exe .\physics-context-detector\scripts\run_learned_b4.py --model .\physics-context-detector\build\learned\model.pkl
```

`--data`, `--assets`, and `--out` can select another fleet and a separate build
subdirectory. Official test pump IDs are rejected. The submission CSV has the same
seven fields as the frozen runner. A missing or incompatible model warns visibly
and writes unchanged B4 values. Pickle artifacts must come from a trusted local
source; never load an untrusted model file.

Feature extraction uses only completed measurement windows and snapshots published
by the alert time. Numeric features include residual summaries over 20/60 minutes,
B4 fast/slow scores, rise duration, absolute physics deviations, quantitative
pump-curve and qualitative vibration/transient signature fits, single-sensor fits,
snapshot spectral features and operating context. Static control, tier and sensor
attribution are explicitly one-hot encoded. Every numeric field has a missingness
indicator; medians come only from fitting rows, with all-missing fields filled zero.
Asset IDs, seeds, absolute timestamps and event IDs never enter model features.

Targets use full simulator labels and scorer windows, independently of B4 type.
All overlapping-event, unobservable, benign and unmatched rows are excluded from
fitting but retained for scoring. Rows have inverse event-frequency weights.
Training uses the preserved event class, and sensor fitting/overrides are restricted
to drift versus calibration shift. Communication loss, stuck and noise/spikes remain
rule-based. Acceptance requires a supported control/tier pair and at least ten
distinct fitting events for the proposed class. Changed diagnoses also require
measured residual evidence; sensor changes require identifiable original attribution.

Official event matching prefers matching diagnoses, so relabeling can change event
recall and excess repeats despite identical timestamps. Selection and acceptance
therefore use the unchanged official scorer, alongside a separate unambiguous alert
accuracy report. The holdout target is a five percentage point type-recall gain
with no aggregate regression; it is an experimental target, not a promised result.
