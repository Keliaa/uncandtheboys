#!/usr/bin/env python3
"""T02: score events that start less than 24 h after a maintenance lockout, with and without the new-baseline step alerts.

Reports: composite on train and validation, the subset of events within 24 h after a lockout (detected, type correct),
false alerts per asset-day. Test pumps are never read."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.detect.maintenance import lockout_restarts
from assist.detect.zscore import z_frame
from assist.ingest.features import load_assets, load_features
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev


def post_lockout_events(assets):
    rows = []
    for split, ev_file in (("train", "events_train.csv"), ("validation", "events_validation.csv")):
        ids = pumps_in(split)
        f = load_features(ids, splits=(split,))
        L = lockout_restarts(z_frame(f, assets))
        e = pd.read_csv(ROOT.parent / "data" / "full" / ev_file)
        e["onset"] = pd.to_datetime(e.onset_utc, utc=True)
        for _, x in e.iterrows():
            l = L[(L.asset_id == x.asset_id) & (L.restart <= x.onset) & (x.onset - L.restart < pd.Timedelta(hours=24))]
            if len(l):
                rows.append(x.event_id)
    return set(rows)


def main():
    assets = load_assets()
    sub = post_lockout_events(assets)
    print(f"events starting <24 h after a lockout: {len(sub)}")
    out = []
    # step limit chosen on TRAIN pumps only
    grid = []
    for z in (4.0, 6.0, 8.0, 12.0):
        pt = predictions_frame(run_pipeline(pumps_in("train"), ("train",), t02_alerts=True, t02_step_z=z)[1])
        grid.append({"step_z": z, "train_composite": ev.score_splits(pt, assets_subset=pumps_in("train"))["train"]["composite_score"], "alerts": len(pt)})
        print(grid[-1], flush=True)
    g = pd.DataFrame(grid).sort_values(["train_composite", "step_z"], ascending=[False, False])
    g.to_csv(ROOT / "results" / "t02_grid.csv", index=False)
    STEP = float(g.iloc[0].step_z)
    print("chosen on train: step_z =", STEP)
    for label, on in (("frozen", False), ("+ T02 after-maintenance alerts", True)):
        p = pd.concat([predictions_frame(run_pipeline(pumps_in(s), (s,), t02_alerts=on, t02_step_z=STEP)[1]) for s in ("train", "validation")], ignore_index=True)
        r = ev.score_splits(p)
        pe = pd.DataFrame([e for sp in ("train", "validation") for e in r[sp]["per_event"]])
        s = pe[pe.event_id.isin(sub)]
        out.append({"variant": label, "train": round(r["train"]["composite_score"], 1), "validation": round(r["validation"]["composite_score"], 1),
                    "FA_day_train": round(r["train"]["false_alerts_per_asset_day"], 2), "FA_day_val": round(r["validation"]["false_alerts_per_asset_day"], 2),
                    "post_lockout_events": len(s), "detected": int(s.detected.sum()), "type_correct": int(s.type_correct.sum()),
                    "mean_timeliness": round(float(s.timeliness.mean()), 2)})
        print(out[-1], flush=True)
        if on:
            p.to_csv(ROOT / "results" / "t02_trainval_predictions.csv", index=False)
            print(s[["event_id", "fault_type", "detected", "type_correct", "timeliness"]].to_string(index=False))
    pd.DataFrame(out).to_csv(ROOT / "results" / "t02_summary.csv", index=False)


if __name__ == "__main__":
    main()
