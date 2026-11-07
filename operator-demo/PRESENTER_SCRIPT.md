# Presenter script (≈4 minutes)

Before the audience arrives: run `./run_demo.sh` on macOS/Linux, or the command below in Windows PowerShell from `operator-demo/` after the [first-time setup](README.md#windows-powershell):

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py --server.headless true --browser.gatherUsageStats false
```

Open http://localhost:8501, then press **Reset demo** in the sidebar. Laptop at a normal resolution,
browser zoom 100 %. Wi-Fi can be off.

## Opening (20 s)

> “This is what a shift operator sees. It answers one question: what needs my attention, why, and what do I do next?
> Everything here is a recorded replay of synthetic pump data — the label at the top says so. The clock in the middle
> is the replay time; the screen only ever shows what was known at that moment.”

Point at: the replay label, the **Monitoring availability** panel on the right, the operating context.

## Chapter 1 — Normal change (60 s)

Press **▶ Play**.

> “The ship is arriving in port: cooling demand falls from about 96 % to under 40 %. The pump is already at its
> minimum speed, so its readings move only a little — bearing temperature and vibration ease off.”

Point at the **Recent readings** charts: demand on top, then measured (solid) following expected (dashed).

> “No task appears. It says *No new checks recommended* — not ‘pump healthy’. And the monitoring panel says monitoring
> is active, so this quiet screen actually means something. If data stopped, it would say so instead.”

Pause or let it run to the end, then **⏭ Next scene**.

## Chapter 2 — Worth checking (75 s)

> “Next day, same pump.” Press **▶ Play**. The replay pauses by itself at 08:23 when a card appears.

Read the card aloud: **Check the suction strainer** — *Why this check?* — *Next step*.

> “No percentages, no z-scores, no deadline. One measurable fact, what it can mean, and the recommended action in the
> detector’s own words.”

Press **Acknowledge**, then **Record check → Nothing unusual found → Save check**.

> “It says *Check recorded; monitoring continues.* Recording a check doesn’t declare the pump healthy, and if the
> detector keeps seeing it, the card re-opens.”

Optional: **Show evidence** — suction pressure against the expected value, the operating context, the original
detector text and where it came from.

## Chapter 3 — Verify the reading (60 s)

Press **⏭ Next scene**. Point at the jump notice: different pump, five months later. Point at **Existing items**:

> “An earlier concern on this pump is still shown, not hidden.”

Press **▶ Play**; it pauses at 00:09 on **Verify the motor-current reading**.

> “The motor current crept up, but flow, pressure and power — which would move with it if the motor really worked
> harder — didn’t. So the recommendation is to verify the instrument before touching the machine.”

## Close (15 s)

> “Three behaviours: stay quiet on normal changes, raise an understandable check when a concern persists, and question
> the measurement when it doesn’t add up. Trust comes from restraint and honest limits — not from promising every
> recommendation is right.”

## If something goes wrong

- Pressed the wrong button: **↺ Restart** restarts the current chapter; **Reset demo** starts over.
- Need to skip ahead: drag **Replay position** — it stops at the next new check, just like Play.
- “No replay bundles found”: run `./run_demo.sh --prepare` on macOS/Linux, or `.\.venv\Scripts\python.exe .\prepare_demo.py` in Windows PowerShell from `operator-demo/` (needs the shared `data/full` directory). Then restart the demo.
