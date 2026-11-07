#!/usr/bin/env python3
"""Tier ablation: rerun B4 on the fully instrumented (T2) train/validation pumps with channels removed to imitate a
T1 retrofit and a T0 legacy installation, and report the composite per tier (official scorer, these 4 pumps only).

T1: no flow meter, power meter (kept on VFD pumps: drives report power), winding PT100 or valve feedback; vibration and
    bearing temperature become a battery wireless sensor (one value every 5 min); snapshots kept.
T0: motor current, discharge pressure, fluid and ambient temperature (+ drive speed / power on VFD pumps); no snapshots.
Caveat: event labels stay those of the T2 asset (observable = True), so T1/T0 are scored on events that their
instruments may not be able to reveal; the lower-tier numbers are therefore pessimistic.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "original-challenge", "src"))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import pandas as pd
import b4
from retrofit_sim.scoring import score
import eval as ev

T2_PUMPS = ["PUMP-02", "PUMP-04", "PUMP-07", "PUMP-09"]
KEEP = {"T1": {"i", "pd", "ps", "tf", "v", "a", "c", "bt"}, "T0": {"i", "pd", "tf"}}


def degrade(inp, tier):
    out = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in inp.items()}
    A = out["assets"]
    feat = out["feat"]
    for idx, a in A.iterrows():
        keep = set(KEEP[tier]) | ({"n", "pw"} if a.control.startswith("VFD") else set())
        keep_cols = [c for k, c in b4.CH.items() if k in keep] + ["ambient_temp_c"]
        A.loc[idx, "channels_available"] = "|".join(c for c in str(a.channels_available).split("|") if c in keep_cols)
        A.loc[idx, "instrumentation_tier"] = f"{tier}_SIM"
        A.loc[idx, "wireless_update_s"] = 300.0 if tier == "T1" else np.nan
        m = feat.asset_id == a.asset_id
        for k, c in b4.CH.items():
            if k not in keep:
                for s in ("mean", "std", "min", "max", "count"):
                    if f"{c}__{s}" in feat:
                        feat.loc[m, f"{c}__{s}"] = np.nan
            elif tier == "T1" and k in b4.WIRELESS:
                minute = pd.to_datetime(feat.loc[m, "window_start_utc"], utc=True).dt.minute
                drop = (minute % 5 != 0).values
                ii = feat.index[m][drop]
                feat.loc[ii, [f"{c}__{s}" for s in ("mean", "std", "min", "max")]] = np.nan
                feat.loc[ii, f"{c}__count"] = 0
                feat.loc[feat.index[m][~drop], f"{c}__std"] = np.nan
                feat.loc[feat.index[m][~drop], f"{c}__count"] = 1
    out["feat"] = feat
    if tier == "T0":
        out["snaps"] = None
    return out


def main():
    inp = b4.load_inputs(os.path.join(os.path.dirname(ROOT), "data", "full"), T2_PUMPS)
    assets, evs, bn = ev.load_labels()
    E = pd.concat([evs["train"], evs["validation"]])
    rows = []
    for tier in ("T2", "T1", "T0"):
        x = inp if tier == "T2" else degrade(inp, tier)
        pred = b4.run_b4(x)
        sub = assets[assets.asset_id.isin(T2_PUMPS)].copy()
        sub["split"] = "abl"
        r = score(pred, E, sub, bn, "abl")
        rows.append({"tier": tier, "composite": r["composite_score"], "F1": r["event_f1"], "recall": r["event_recall"],
                     "precision": r["alert_precision"], "timeliness": r["timeliness"], "type_rec": r["fault_type_recall"],
                     "class_rec": r["class_recall"], "FA+ex/day": r["false_alerts_per_asset_day"], "events": r["events_scored"]})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    df.to_csv(os.path.join(ROOT, "results", "tier_ablation.csv"), index=False)


if __name__ == "__main__":
    main()
