#!/usr/bin/env python3
"""C03: end-to-end demo run, replay -> alert -> evidence, certainty, action, what-if -> decision, with no manual step.

  python pump-assist/scripts/c03_demo.py                      # PUMP-08, fast
  python pump-assist/scripts/c03_demo.py --speed 7200         # paced: 7200 simulated seconds per wall second
  streamlit run pump-assist/app/main.py                       # then choose "Replay demo" in the sidebar

The replay (I02) streams the stored per-second telemetry of one pump as messages; the 1-minute buffer builds the same rows as I01;
every `--check-min` simulated minutes the system (the frozen pipeline) runs on the rows received so far, so an alert appears at the
first check after the data that causes it. Nothing is read from the future. The operator step is a SCRIPTED policy (approve machine
alerts with certainty Medium or higher, 'not sure' otherwise); it writes results/demo/decisions_demo.csv, never the real decision log.
Screen recording for the demo video is the one thing this script cannot do."""
import argparse, csv, json, sys, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from assist.ingest.replay import minute_rows, replay_messages
from assist.pipeline import run_pipeline
from assist.splits import DATA_FULL, load_split_map

DEMO = ROOT / "results" / "demo"


def scripted_decision(alert) -> str:
    if alert["event_class"] == "MACHINE" and alert["certainty"] in ("Medium", "High"):
        return "approved"
    return "not sure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="PUMP-08")
    ap.add_argument("--start", default="2025-09-08T00:00:00Z")
    ap.add_argument("--end", default="2025-09-12T00:00:00Z")
    ap.add_argument("--check-min", type=int, default=180, help="run the system every N simulated minutes")
    ap.add_argument("--speed", type=float, default=None, help="simulated seconds per wall-clock second (default: as fast as possible)")
    a = ap.parse_args()
    DEMO.mkdir(parents=True, exist_ok=True)
    split = load_split_map(DATA_FULL / "assets.csv")[a.asset]
    log_lines = []

    def say(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    say(f"Replay of {a.asset} ({split}) from {a.start} to {a.end}; system check every {a.check_min} simulated minutes")
    rows, seen, live, decisions = [], set(), [], []
    t_wall = time.time()
    for row in minute_rows(replay_messages(a.asset, DATA_FULL, a.start, a.end), a.asset, a.speed):
        rows.append(row)
        if len(rows) % a.check_min:
            continue
        now = row["window_start_utc"] + pd.Timedelta(minutes=1)
        _, alerts = run_pipeline([a.asset], (split,), features=pd.DataFrame(rows))
        for al in alerts:
            key = (al.alert_time_utc, al.event_class, al.fault_type)
            if key in seen or pd.Timestamp(al.alert_time_utc) > now:
                continue
            seen.add(key)
            d = asdict(al)
            d["z_at_alert"] = {}
            d["detected_at_replay_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            d["operator_decision"] = scripted_decision(d)
            live.append(d)
            decisions.append({"decided_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "asset_id": al.asset_id,
                              "alert_time_utc": al.alert_time_utc, "fault_type": al.fault_type, "decision": d["operator_decision"],
                              "source": "scripted demo policy"})
            w = al.what_if or {}
            wi = (f"flow {w['flow_change_pct']:+.1f} %, risk {w['risk_before']:.2f} -> {w['risk_after']:.2f}, cost ~{w['cost_eur']:.0f} EUR"
                  if "flow_change_pct" in w else "no what-if (sensor alert or no sandbox action)")
            say(f"[replay {now:%m-%d %H:%M}] ALERT {al.alert_time_utc} {al.event_class} {al.fault_type} | certainty {al.certainty} ({al.probability:.0%})")
            for e in al.evidence:
                say(f"    evidence: {e.text}")
            say(f"    action: {al.recommended_action}")
            say(f"    what-if: {wi}")
            say(f"    decision (scripted): {d['operator_decision']}")
    say(f"Replay done: {len(rows)} minutes, {len(live)} alerts, {time.time() - t_wall:.0f} s wall time")
    (DEMO / "alerts_live.json").write_text(json.dumps({"asset_id": a.asset, "start": a.start, "end": a.end, "check_min": a.check_min, "alerts": live},
                                                      indent=1, default=str))
    with open(DEMO / "decisions_demo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["decided_utc", "asset_id", "alert_time_utc", "fault_type", "decision", "source"])
        w.writeheader(); w.writerows(decisions)
    (DEMO / "transcript.txt").write_text("\n".join(log_lines) + "\n")
    pd.DataFrame(rows).to_parquet(DEMO / "replayed_minutes.parquet")


if __name__ == "__main__":
    main()
