#!/usr/bin/env python3
"""Q01: choose gap_persist and flat_min on TRAIN pumps only, then score validation once. Test pumps are never read.

Compares D01+E01+R01+D04 with the same plus Q01 (trust gate + sensor alerts)."""
import itertools, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.config import _asset_from_dict, assets_from_csv, load_machine
from assist.detect.physics import detect_physics, physics_alerts
from assist.detect.zscore import DEFAULT_K, DEFAULT_PERSIST, detect, to_alerts_with_rules, z_frame
from assist.ingest.features import load_assets, load_features
from assist.ingest.quality import sensor_alerts, trust_frame
from assist.pipeline import predictions_frame
from assist.recommend.actions import add_actions
from assist.splits import pumps_in
import eval as ev

GAP, FLAT = [5, 10, 30], [10, 15, 30]


def machine_part(feats, assets, cfg, machine, zf, gate):
    det = detect(feats, assets, DEFAULT_K, DEFAULT_PERSIST, trust_gate=gate, zf=zf)
    a = add_actions(to_alerts_with_rules(det, cfg, machine, DEFAULT_K), machine)
    a += physics_alerts(detect_physics(feats, assets, machine, 2.0, 30), machine)
    return a


def main():
    assets, machine = load_assets(), load_machine()
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(ROOT.parent / "data" / "full" / "assets.csv")}
    tr, va = pumps_in("train"), pumps_in("validation")
    data = {}
    for name, ids, sp in (("train", tr, "train"), ("validation", va, "validation")):
        f = load_features(ids, splits=(sp,))
        zf = z_frame(f, assets)
        data[name] = (f, zf, trust_frame(zf))
    def preds(name, gate, gp=None, fm=None):
        f, zf, tf = data[name]
        a = machine_part(f, assets, cfg, machine, zf, gate)
        if gp is not None:
            a += sensor_alerts(zf, trust_frame(zf, fm) if fm != 15 else tf, machine, gp, fm)
        return predictions_frame(a)
    grid = []
    for gp, fm in itertools.product(GAP, FLAT):
        p = preds("train", True, gp, fm)
        grid.append({"gap_persist": gp, "flat_min": fm, "train_composite": ev.score_splits(p, assets_subset=tr)["train"]["composite_score"], "alerts": len(p)})
        print(grid[-1], flush=True)
    g = pd.DataFrame(grid).sort_values(["train_composite", "gap_persist", "flat_min"], ascending=False)  # ties: most conservative
    g.to_csv(ROOT / "results" / "q01_grid.csv", index=False)
    gp, fm = int(g.iloc[0].gap_persist), int(g.iloc[0].flat_min)
    print(f"chosen on train: gap_persist={gp} flat_min={fm}")
    full = {}
    for label, gate, use in (("before Q01", False, False), ("trust gate only", True, False), ("Q01 gate+sensor alerts", True, True)):
        p = pd.concat([preds("train", gate, gp if use else None, fm), preds("validation", gate, gp if use else None, fm)], ignore_index=True)
        s = ev.score_splits(p)
        full[label] = p
        print(f"{label}: train {s['train']['composite_score']:.1f} (class rec {s['train']['class_recall']:.2f}, FA/day {s['train']['false_alerts_per_asset_day']:.2f}) "
              f"validation {s['validation']['composite_score']:.1f} (class rec {s['validation']['class_recall']:.2f}, FA/day {s['validation']['false_alerts_per_asset_day']:.2f})")
    full["Q01 gate+sensor alerts"].to_csv(ROOT / "results" / "q01_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
