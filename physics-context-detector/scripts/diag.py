#!/usr/bin/env python3
"""Development helper: list alerts with the event windows that contain them (FALSE = no window) on train/val or stress."""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "original-challenge", "src"))
sys.path.insert(0, ROOT)
import pandas as pd
from retrofit_sim.scoring import PRE_TOL, POST_TOL, SOFT_END


def windows(ev, assets):
    ev = ev.copy()
    ws = dict(zip(assets.asset_id, pd.to_datetime(assets.window_start_utc, utc=True)))
    we = dict(zip(assets.asset_id, pd.to_datetime(assets.window_end_utc, utc=True)))
    for c in ("onset_utc", "detectable_utc", "end_utc"):
        ev[c] = pd.to_datetime(ev[c], utc=True, format="ISO8601")
    st, en = [], []
    for _, e in ev.iterrows():
        s = max(e.onset_utc, ws[e.asset_id])
        if pd.notna(e.detectable_utc):
            s = max(s, e.detectable_utc - PRE_TOL)
        if pd.isna(e.end_utc):
            x = we[e.asset_id]
        else:
            soft = e.fault_type in SOFT_END or e.event_class == "SENSOR"
            x = min(e.end_utc + (POST_TOL if soft else pd.Timedelta(0)), we[e.asset_id])
        st.append(s); en.append(x)
    ev["w_start"], ev["w_end"] = st, en
    return ev


def annotate(pred, ev, assets):
    ev = windows(ev, assets)
    p = pred.copy()
    p["t"] = pd.to_datetime(p.alert_time_utc, utc=True)
    out = []
    for _, r in p.iterrows():
        c = ev[(ev.asset_id == r.asset_id) & (ev.w_start <= r.t) & (r.t <= ev.w_end)]
        out.append(";".join(f"{e.event_id}:{e.fault_type[:14]}{'' if e.observable else '(unobs)'}" for _, e in c.iterrows()) or "FALSE")
    p["in_windows"] = out
    return p


if __name__ == "__main__":
    pred = pd.read_csv(sys.argv[1])
    data = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(ROOT), "data", "full")
    if "stress" in data:
        assets = pd.read_csv(os.path.join(data, "public", "assets.csv"))
        ev = pd.read_csv(os.path.join(data, "private", "events_stress_all.csv"))
    else:
        assets = pd.read_csv(os.path.join(data, "assets.csv"))
        ev = pd.concat([pd.read_csv(os.path.join(data, "events_train.csv")), pd.read_csv(os.path.join(data, "events_validation.csv"))])
    p = annotate(pred[pred.asset_id.isin(set(ev.asset_id) | set(assets[assets.split != "test"].asset_id))], ev, assets)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110); pd.set_option("display.max_rows", 500)
    only_false = "--false" in sys.argv
    q = p[p.in_windows == "FALSE"] if only_false else p
    print(q[["asset_id", "alert_time_utc", "fault_type", "detector", "in_windows", "evidence"]].to_string(index=False))
