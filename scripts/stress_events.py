#!/usr/bin/env python3
"""Development helper: run B4 on stress fleets and print the events that were missed or mistyped."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pandas as pd
from retrofit_sim import b4
import eval as ev
P = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
rows = []
for sd in sys.argv[1].split(","):
    d = os.path.join(ROOT, "build", f"stress_{sd}")
    assets = pd.read_csv(os.path.join(d, "public", "assets.csv"))
    inp = b4.load_inputs(os.path.join(d, "public"), list(assets.asset_id))
    pr = b4.run_b4(inp, P)
    pr.to_csv(os.path.join(d, "b4_pred.csv"), index=False)
    r = ev.score_stress_dir(pr, d)
    evs = pd.read_csv(os.path.join(d, "private", "events_stress_all.csv")).set_index("event_id")
    A = assets.set_index("asset_id")
    for e in r["per_event"]:
        x = evs.loc[e["event_id"]]
        rows.append({"seed": sd, "event": e["event_id"], "ctrl": A.loc[e["asset_id"]].control[:9], "tier": A.loc[e["asset_id"]].instrumentation_tier[:2],
                     "true": e["fault_type"], "tag": str(x.sensor_tag)[:12], "pred": e["predicted_fault_type"], "det": e["detected"],
                     "ok": e["type_correct"], "delay_h": None if e["delay_after_detectable_h"] is None else round(e["delay_after_detectable_h"], 1),
                     "tl": round(e["timeliness"], 2), "ch": str(x.evidence_channels)[:40]})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(df[~df.ok].to_string(index=False))
print("\ntype recall by true type:"); print(df.groupby("true").agg(n=("ok", "size"), det=("det", "mean"), ok=("ok", "mean")).round(2))
