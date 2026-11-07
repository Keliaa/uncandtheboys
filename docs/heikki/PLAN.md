# B4 demo: quiet monitoring, useful actions

## Summary

Build a **3–5 minute guided, interactive demo for a shift operator**, running offline on a laptop. Demonstrate three behaviors: normal operating changes do not create unnecessary tasks; persistent concerns produce understandable checks; unreliable measurements lead to instrument verification.

The main screen answers: **“What needs my attention, why, and what should I do next?”** Root-cause analysis stays behind an optional evidence view.

Success means a viewer can understand each recommendation without interpreting spectra, fault codes, z-scores, or confidence percentages.

## Demo experience

Use three presenter-controlled replay chapters, clearly labelled **“Recorded replay · synthetic pump data.”** Provide Play/Pause, Next scene, Restart, and a visible replay timestamp. Pause automatically when a new action appears.

| Scene | What happens | Operator experience |
|---|---|---|
| **1. Normal change** | Replay a labelled benign load change on PUMP-08 after its initial 24-hour calibration period and before its first alert. | Readings change, but no task appears. Show “No new checks recommended,” operating context, and monitoring availability. Never claim the pump is fault-free. |
| **2. Worth checking** | Replay the PUMP-08 strainer-clogging alert at **2025-09-10 08:23 UTC**, with preceding context. | Show “Check the suction strainer,” a short factual explanation, and the inspection recommendation. The operator acknowledges it and records a check or maintenance handoff. |
| **3. Verify the reading** | Replay PUMP-09’s motor-current sensor-drift alert at **2026-02-11 00:09 UTC**. | Show “Verify the motor-current reading.” Explain that the current reading changed without the accompanying changes expected from a machine issue. Recommend instrument verification before machine maintenance. |

Each chapter starts from its historical state; explicitly label jumps between pumps and dates. Include earlier unresolved concerns in a secondary “Existing items” area rather than hiding them.

Select the normal-change window deterministically: the earliest qualifying labelled load-change window with available monitoring and no overlapping B4 alert. Fail preparation visibly if none qualifies; do not fabricate a quiet example.

## Interface and action behavior

- **Main view:** selected pump, monitoring availability, replay controls, and action cards. Use calm styling without flashing alarms, audible alerts, or a fabricated overall health score.
- **Each card:** plain-language check, pump/location, first-raised time, concise “Why this check?” explanation, and next step. Preserve the recommendation’s conditional wording; do not invent urgency, remaining life, or maintenance deadlines.
- **Evidence drawer:** measured-versus-expected trends where available, relevant operating context, source evidence text, and suspected fault type. Omit uncalibrated confidence percentages.
- **Workflow:** New → Acknowledged → Check recorded. Recording a check offers “Concern observed,” “Nothing unusual found,” “Unable to verify,” or “Passed to maintenance,” plus optional notes.
- **Separate task and monitoring state:** acknowledging or recording a check never declares the machine healthy or suppresses future detector evidence. Say “Check recorded; monitoring continues.”
- Group repeated alerts for the same pump and recommended check into one card with first/latest occurrence and update history. A different recommended check remains a separate item. Retain every underlying B4 alert.
- Missing monitoring data produces an explicit unavailable state. A blank action queue must never substitute for monitoring availability.
- Store demo interactions locally and provide a clearly labelled reset. No external work orders, device commands, or notifications.

## Implementation

Create an isolated `operator-demo/` application using **Streamlit and Plotly**, with locally bundled assets and no network calls during presentation.

- Add an offline preparation command that reads shared development data and imports B4 unchanged. Generate compact replay bundles containing completed one-minute observations, expected values, monitoring status, and timestamped alerts.
- Use frozen B4 parameters. Process historical context from each pump’s monitoring start, then extract presentation windows; do not restart calibration at the visible scene boundary.
- Display only information available by the replay clock. Reveal completed minute aggregates at minute end and alerts at their recorded publication times.
- Use labels only to select and validate demonstration scenes. Exclude fault labels, future deadlines, and future observations from operator-facing logic.
- Keep B4’s prediction format unchanged. Add a presentation adapter for operator wording, grouping, and local interaction state, with source-alert references for traceability.
- Save generated bundles and local interaction logs inside the demo directory. Preserve shared data, original challenge files, and frozen benchmark outputs.
- Supply setup instructions, a one-command launcher, prepared replay bundles, and a short presenter script. The presentation itself must not require rerunning B4.

## Verification and boundaries

- Confirm quiet-scene behavior and exact alert timing against B4 outputs.
- Verify seeking, restarting, and scene transitions never reveal future evidence.
- Test repeated-alert grouping, separate checks, acknowledgement, recorded findings, and reset.
- Test unavailable data and incomplete calibration explicitly; neither may display a healthy state.
- Confirm checking “Nothing unusual found” does not erase evidence or disable monitoring.
- Run B4’s existing causality checks and perform a complete browser walkthrough with networking disabled.
- Validate that every card’s explanation is supported by its source alert and that all controls work at typical laptop resolution.

**Defaults:** English interface; UTC replay timestamps; local single-user session; prepared recorded inference rather than live streaming. Sandbox comparisons, spectrum views, probability calibration, hosted deployment, and operational urgency classification are deferred.

Trust is demonstrated through restraint, understandable evidence, and honest monitoring limits—not a promise that every recommendation is correct.
