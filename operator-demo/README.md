# B4 operator demo — quiet monitoring, useful actions

A 3–5 minute, presenter-controlled replay for a shift operator, running offline on a laptop. It shows three
behaviours of the B4 detector ([../physics-context-detector](../physics-context-detector)):

1. **Normal change** — a labelled load change on PUMP-08 creates no task.
2. **Worth checking** — the PUMP-08 suction-strainer alert (2025-09-10 08:23 UTC) becomes an understandable check.
3. **Verify the reading** — the PUMP-09 motor-current drift alert (2026-02-11 00:09 UTC) leads to instrument
   verification before any machine work.

The main screen answers *“What needs my attention, why, and what should I do next?”*. Root-cause detail sits behind
**Show evidence** on each card. Plan: [../docs/heikki/PLAN.md](../docs/heikki/PLAN.md). Presenter script:
[PRESENTER_SCRIPT.md](PRESENTER_SCRIPT.md).

## Run it

Run the commands in this section from `operator-demo/`.

### macOS / Linux

```sh
./run_demo.sh
```

The first run creates `.venv/` and installs [requirements.txt](requirements.txt) (needs network once). After that the
presentation is fully offline: it opens http://localhost:8501, serves Streamlit/Plotly assets locally and sends no
telemetry (see [.streamlit/config.toml](.streamlit/config.toml)). It never runs B4 — it reads the prepared bundles.

Manual equivalent:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

### Windows PowerShell

From the repository root (`Uncandtheboys`), enter the demo directory and install its dependencies once:

```powershell
cd .\operator-demo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\app.py --server.headless true --browser.gatherUsageStats false
```

Open http://localhost:8501 in your browser. Stop the server with Ctrl+C. On later runs, execute only the last command from `operator-demo/`.

Use Python 3.10 or newer. `run_demo.sh` requires Bash and Unix virtualenv paths (`.venv/bin/`); native Windows virtualenvs use `.venv\Scripts\`. Use the PowerShell commands above on Windows. They do not require virtualenv activation or execution-policy changes. If `python` is unavailable but the Python launcher is installed, use `py -3 -m venv .venv` for the creation step.

## Replay bundles

`bundles/` is committed and is all the presentation needs. To regenerate it from the shared development data:

```sh
./run_demo.sh --prepare          # or: .venv/bin/python prepare_demo.py
```

Windows PowerShell, from `operator-demo/` after installing dependencies:

```powershell
.\.venv\Scripts\python.exe .\prepare_demo.py
```

[prepare_demo.py](prepare_demo.py) imports `b4.py` unchanged, runs it with the frozen parameters over each pump's whole
window (so calibration starts at monitoring start, not at the scene boundary), and refuses to write anything unless the
regenerated alerts are identical to `physics-context-detector/results/b4_trainval_predictions.csv`. The quiet window
is the earliest labelled `LOAD_CHANGE` after calibration, before the pump's first alert, with monitoring active and no
alert in the replay window; if none qualifies, preparation fails instead of inventing one. Labels are used only for
this selection/validation and go to `bundles/prep_report.json`, which the app never reads.

## What it does and does not claim

- Only information available at the replay clock is shown: a one-minute average appears when its minute is complete;
  an alert appears at its recorded publication time. Bundles contain nothing after each chapter's end.
- No health score, confidence percentages, urgency, remaining life or deadlines. The **Next step** is B4's recommended
  action verbatim; **Why this check?** is built only from facts in the alert's evidence text.
- Repeated alerts for the same pump and check are grouped into one card (first/latest time, history); every underlying
  alert is kept. Concerns raised before a chapter starts appear under **Existing items**.
- Workflow: New → Acknowledged → Check recorded (Concern observed / Nothing unusual found / Unable to verify / Passed to
  maintenance, plus notes). Recording a check never marks the pump healthy or changes monitoring: *“Check recorded;
  monitoring continues.”* New detector evidence re-opens the card.
- Missing data, calibration, stopped or settling pumps show an explicit monitoring state — an empty list is never
  presented as “fine”.
- Actions are logged locally to `state/interactions.jsonl` (git-ignored). **Reset demo** clears it; **Restart**
  restarts the current chapter. No work orders, device commands or notifications are created.

## Tests

```sh
.venv/bin/python -m pytest -q tests                      # replay timing, grouping, workflow, monitoring, app walkthrough
.venv/bin/python -m pytest -q ../physics-context-detector/tests   # B4's causality checks
```

Windows PowerShell, from `operator-demo/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q .\tests
.\.venv\Scripts\python.exe -m pytest -q ..\physics-context-detector\tests
```

## Layout

| Path | What |
|---|---|
| `app.py` | Streamlit operator interface |
| `prepare_demo.py` | offline bundle preparation (runs B4) |
| `demo/replay.py` | bundle loading, replay clock, monitoring availability |
| `demo/adapter.py` | B4 alert → operator card wording, grouping, task status |
| `demo/interactions.py` | local interaction log |
| `bundles/` | prepared replay bundles + manifest |
| `state/` | local interaction log (created at runtime, git-ignored) |
