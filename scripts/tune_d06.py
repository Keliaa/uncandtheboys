#!/usr/bin/env python3
"""D06: choose tail_min and the sea-state limit on TRAIN pumps only, then score validation once. Test pumps are never read.

Baseline for comparison: the Q01 system (D01+E01+R01+D04+Q01) with no benign-period hold."""
import itertools, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.config import _asset_from_dict, assets_from_csv, load_machine
from assist.detect.benign import benign_minutes
from assist.detect.physics import detect_physics, physics_alerts
from assist.detect.zscore import DEFAULT_K, DEFAULT_PERSIST, detect, to_alerts_with_rules, z_frame
from assist.ingest.features import load_assets, load_features
from assist.ingest.quality import sensor_alerts, trust_frame
from assist.pipeline import predictions_frame
from assist.recommend.actions import add_actions
from assist.splits import pumps_in
import eval as ev

TAIL, HS = [0, 15, 30, 60], [2.2, 3.0]
GAP, FLAT = 30, 30   # Q01 settings


def main():
    assets, machine = load_assets(), load_machine()
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(ROOT / "data" / "full" / "assets.csv")}
    tr, va = pumps_in("train"), pumps_in("validation")
    data = {}
    for name, ids in (("train", tr), ("validation", va)):
        f = load_features(ids, splits=(name,))
        zf = z_frame(f, assets)
        data[name] = (f, zf, trust_frame(zf, FLAT), ids)

    def preds(name, tail=None, hs=2.2):
        f, zf, tf, ids = data[name]
        ben = None if tail is None else benign_minutes(ids, tail_min=tail, hs_limit=hs)
        det = detect(f, assets, DEFAULT_K, DEFAULT_PERSIST, trust_gate=True, zf=zf, benign=ben)
        a = add_actions(to_alerts_with_rules(det, cfg, machine, DEFAULT_K), machine)
        a += physics_alerts(detect_physics(f, assets, machine, 2.0, 30, benign=ben), machine)
        a += sensor_alerts(zf, tf, machine, GAP, FLAT)
        return predictions_frame(a)

    grid = []
    for tail, hs in itertools.product(TAIL, HS):
        p = preds("train", tail, hs)
        r = ev.score_splits(p, assets_subset=tr)["train"]
        grid.append({"tail_min": tail, "hs_limit": hs, "train_composite": r["composite_score"], "alerts": len(p),
                     "recall": r["event_recall"], "FA_per_day": r["false_alerts_per_asset_day"]})
        print(grid[-1], flush=True)
    g = pd.DataFrame(grid).sort_values(["train_composite", "tail_min"], ascending=[False, True])
    g.to_csv(ROOT / "results" / "d06_grid.csv", index=False)
    tail, hs = int(g.iloc[0].tail_min), float(g.iloc[0].hs_limit)
    print(f"chosen on train: tail_min={tail} hs_limit={hs}")
    for label, kw in (("no hold (Q01 system)", dict(tail=None)), ("with D06", dict(tail=tail, hs=hs))):
        p = pd.concat([preds("train", **kw), preds("validation", **kw)], ignore_index=True)
        s = ev.score_splits(p)
        print(f"{label}: train {s['train']['composite_score']:.1f} FA/day {s['train']['false_alerts_per_asset_day']:.2f} | "
              f"validation {s['validation']['composite_score']:.1f} FA/day {s['validation']['false_alerts_per_asset_day']:.2f} "
              f"recall {s['validation']['event_recall']:.2f}")
        if kw["tail"] is not None:
            p.to_csv(ROOT / "results" / "d06_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
