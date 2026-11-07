# Lovable cockpit export

One command builds every data file the Lovable operator cockpit reads (`public/data/` in the cockpit repo),
validates it against the contract below and, on request, pushes it. The cockpit is fully data-driven: files that
follow this contract update the UI with no UI change.

Cockpit repo: `git@github.com:SmokkiG/uncandtheboys.git` (the Lovable project "uncandtheboys"), cloned next to this
repo as `../cockpit`. The exporter writes only to `../cockpit/public/data/` and commits only that folder.

## Run

From the repository root, with the requirements of `original-challenge/requirements.txt` installed:

```sh
git clone git@github.com:SmokkiG/uncandtheboys.git ../cockpit          # once
python lovable-export/export_lovable_data.py --out ../cockpit/public/data           # build + validate
python lovable-export/export_lovable_data.py --out ../cockpit/public/data --push    # build + validate + commit + push
python -m pytest -q lovable-export/tests
```

`--push` pulls with `--ff-only`, refuses to commit anything outside `public/data/`, commits
`data: refresh cockpit exports (<short sha of this repo>)` and does a plain (never forced) push, so Lovable's sync
history is preserved. Validation runs automatically and a failure stops the push. `validate.py <dir>` can also be run
on its own (`--no-causal` skips the slow truncated re-runs).

Inputs (all read only): `data/full/`, the frozen B4 module and parameters in `physics-context-detector/`, the frozen
predictions `b4_trainval_predictions.csv` / `b4_test_predictions.csv`, `baseline-reproduction/results/b1_trainval_predictions.csv`
and `physics-context-detector/results/tier_ablation.csv`. Optional simulated pumps come from
`physics-context-detector/build/stress_<seed>/` (see the B4 README for generating them; `--stress-root` points elsewhere).
B4 is re-run unchanged with frozen parameters; for every benchmark pump the regenerated alerts must equal the frozen
prediction files, otherwise nothing is written. A full run takes about a minute.

**After B4 changes:** regenerate the frozen predictions first (`physics-context-detector/scripts/run_b4.py --final`),
commit them here, then rerun the export with `--push`. The exporter will refuse to run while the detector and the
frozen predictions disagree.

## Rules every file follows

1. Synthetic data only; every file has a `label` saying so.
2. No event labels. `events_*.csv` are read only to validate the two alert scenes (prepare_demo's rule) and to
   compute the aggregated scores in `impact.json`. Test pumps PUMP-10..12 appear with B4 predictions only.
3. Causal: every value for time t uses only data with timestamp <= t. A minute counts once it is complete; plant alarm
   clearing times after a bundle's end are written as `null`. `validate.py` re-runs B4 on inputs truncated at 20 random
   snapshot times (train, validation and simulated pumps) and requires identical snapshots.
4. Deterministic: `generated_utc` is the commit time of this repo's HEAD, not the wall clock; floats are rounded to 3
   decimals. Two runs produce byte-identical files.
5. No em dashes in any text (replaced by `-`).
6. Size budget: scene bundle < 1 MB, `fleet.json` < 2.5 MB, `impact.json` < 300 KB.

## Engines (switchable in the cockpit)

The cockpit can show two alert engines; `manifest.json` lists them in `engines` (`default_engine` is `b4`, and the
top-level `scenes`/`fleet`/`impact` keys stay the B4 ones so an engine-unaware UI keeps working).

| id | alerts | files |
|---|---|---|
| `b4` | B4 detector re-run with frozen parameters, checked against `physics-context-detector/results/b4_*_predictions.csv` | `scene_*.json`, `fleet.json` (benchmark + simulated pumps) |
| `v2` | pump-assist v2, Elias's generator-free engine (`engine: v2` in `pump-assist-v2/configs/pipeline.yaml`), read from `pump-assist-v2/results/episodes_trainval_predictions.csv` + `test_predictions.csv`. The export refuses the file unless it scores exactly what `engines_compare.csv` reports for v2. | `scene_*_v2.json`, `fleet_v2.json` (12 benchmark pumps; v2 publishes no alerts for the simulated pumps) |

Both engines share `impact.json` (`detector.b1/b4/v2`, `per_pump_exposure_by_engine`). Telemetry, expected values,
deviation and monitoring status are the same for both (B4's context model; v2 publishes alerts, not per-minute expected
values). v2 scenes use the same windows rules: the quiet scene is prepare_demo's rule with v2 alerts; an alert scene
is the earliest v2 alert on the scene pump with the scene's event class inside a labelled event of the scene's fault
type (v2 may name the type differently, e.g. SENSOR_CALIBRATION_SHIFT inside the PUMP-09 drift). When v2's evidence
text is not B4's wording, `evidence_channels` are read from the channel names in the text. The causal re-check also
samples 10 `fleet_v2.json` snapshots.

Note: `pump-assist-v2/results/current_trainval_predictions.csv` is the B4-engine output (it scores 92.3 / 95.7), not v2.

## Files

### manifest.json (`operator-demo-replay/1`)
`format, label, generated_utc, source_commit, scenes[{chapter, id, file}], fleet, impact`.

### scene_<n>_<id>.json (`operator-demo-replay/1`)
Exactly what `operator-demo/prepare_demo.py` builds (its scene selection and validation, its `build_bundle`), plus
`plant_alarms: [{tag, alarm_type, raised_utc, cleared_utc}]`: rows of `alarms.csv` for the pump raised inside
`lookback_start_utc .. end_utc`. UI mapping: `minutes.status` drives the status line, `channels.<key>.measured/expected`
the "What we see" charts, `alerts` the inbox cards (evidence is "Why this check?", recommended_action is "Next step",
both verbatim); an active plant alarm turns the pump WARNING.

### fleet.json (`fleet-replay/1`)
`assets[{asset_id, site, system, control, instrumentation_tier, split, snapshots[...]}]`, one snapshot every 30 min for
7 days (337 per pump), `offset_min` = minutes since the pump's monitoring start:

| field | meaning |
|---|---|
| `status` | B4's monitoring state of the last completed minute: active, calibrating, not_running, settling, unavailable |
| `deviation`, `deviation_channel` | max over channels of \|measured - expected\| / sigma, the channel that sets it; `null` while calibrating. sigma = residual std of the pump's 24 h calibration window (`common.calibration_sigmas`, the same method the cockpit uses for the live pump; channels without one fall back to the std of residuals so far). The latest residual up to 30 min old is used. |
| `level` | see the presentation rule below |
| `alert` | most severe open alert (MACHINE before SENSOR, then latest), verbatim B4 text: `fault_type, event_class, evidence, recommended_action, alert_time_utc`; else `null` |

Pumps: all 12 benchmark pumps (`split` train / validation / test) and then simulated pumps (`split` simulated, renamed
`SIM-<seed>-<nn>`, site "Simulated site <seed> (fictional)") in seed order, up to 32. Open-alert snapshots repeat the
alert text, so simulated pumps are dropped from the end until the file fits 2.5 MB (currently 21 of 32 fit). The
impact scores for simulated pumps always use all generated stress fleets.

**Level: a presentation rule, not a detector claim.** WARNING if a plant alarm for the pump is active at the snapshot
time; else CAUTION if a B4 MACHINE alert is open; else ADVISORY if a B4 SENSOR alert is open; else `null`. An alert
group (same pump + same recommended_action) is *open* from an alert's time until 12 h after the group's latest alert,
or until the pump is LOCKED_OUT (a completed LOCKED_OUT minute after that alert), whichever comes first. B4 itself
publishes alerts, not levels or durations; the 12 h hold mirrors the scorer's one-free-reminder-per-12-h window so a
tile does not go quiet between reminders.

### impact.json (`impact/1`)
* `assumptions`: editable cost assumptions (organiser `business_case.py` defaults + team labour rate).
* `maintenance`: from `data/full/maintenance.csv`, hours = end_utc - start_utc. Classification:
  * no_finding: ROUTINE_INSPECTION "no abnormal findings"; STRAINER_CLEANING "Clean."; VIBRATION_ROUTE_MEASUREMENT in
    zone A/B; ALIGNMENT_CHECK "Within tolerance."; COUPLING_INSPECTION "...checked, OK."; INSTRUMENT_CHECK
    "...loop-checked, OK."
  * light: STRAINER_CLEANING "Light debris."; justified: STRAINER_CLEANING "Moderate fouling."
  * LUBRICATION is never no_finding (always needed); corrective = `planned == False`.
  * Known totals (asserted in the tests): 741 records, 458 no_finding, 62 % of hours, 301 strainer cleanings of which
    109 found clean. `no_finding_eur_per_year` = no-finding hours x labour rate / years in the maintenance period.
* `detector`: B1 and B4 from the official scorer on train and validation, and on the simulated stress fleets (mean
  over seeds as in the B4 RESULTS.md, medians pooled over events): composite, event recall, fault-type recall, class
  recall, false+excess alerts per asset-day, median lead time to deadline, median delay after detectable.
* `tier_ablation`: B4 on the T2 pumps degraded to T1 / T0 channels.
* `per_pump_exposure`: per fleet pump, hours with an open alert (same rule as `level`) and alert counts.

Energy is left out for now (power is not measured on T0 pumps).

## Layout

| path | what |
|---|---|
| `export_lovable_data.py` | builds all files, runs validation, optional push |
| `common.py` | shared rules: B4 run, truncation, sigma/deviation, open-alert rule, snapshot, JSON hygiene |
| `validate.py` | contract checks + causal re-check + summary table |
| `tests/` | rule tests, maintenance totals, validator tests, contract test on `../cockpit/public/data` (or `LOVABLE_EXPORT_DIR`) |
