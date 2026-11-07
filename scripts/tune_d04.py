#!/usr/bin/env python3
"""D04: choose limit and persistence on TRAIN pumps only, then score validation once. Test pumps are never read.

Scores D04 alone and D04 added to D01+E01+R01 (union of alerts)."""
import itertools, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.config import load_machine
from assist.detect.physics import detect_physics, physics_alerts
from assist.ingest.features import load_assets, load_features
from assist.pipeline import predictions_frame
from assist.splits import pumps_in
import eval as ev

LIMITS, PERSIST = [2.0, 3.0, 5.0, 8.0, 12.0], [30, 60, 180]


def preds(feats, assets, machine, lim, per):
    return predictions_frame(physics_alerts(detect_physics(feats, assets, machine, lim, per), machine))


def main():
    assets, machine = load_assets(), load_machine()
    tr, va = pumps_in("train"), pumps_in("validation")
    ftr, fva = load_features(tr, splits=("train",)), load_features(va, splits=("validation",))
    grid = []
    for lim, per in itertools.product(LIMITS, PERSIST):
        p = preds(ftr, assets, machine, lim, per)
        r = ev.score_splits(p, assets_subset=tr)["train"] if len(p) else {"composite_score": 0.0}
        grid.append({"limit_pct": lim, "persist": per, "train_composite": r["composite_score"], "alerts": len(p)})
        print(grid[-1], flush=True)
    g = pd.DataFrame(grid).sort_values("train_composite", ascending=False)
    g.to_csv(ROOT / "results" / "d04_grid.csv", index=False)
    lim, per = float(g.iloc[0].limit_pct), int(g.iloc[0].persist)
    d04 = pd.concat([preds(ftr, assets, machine, lim, per), preds(fva, assets, machine, lim, per)], ignore_index=True)
    d04.to_csv(ROOT / "results" / "d04_trainval_predictions.csv", index=False)
    e01 = pd.read_csv(ROOT / "results" / "e01_trainval_predictions.csv")
    comb = pd.concat([e01, d04], ignore_index=True).sort_values(["asset_id", "alert_time_utc"])
    comb.to_csv(ROOT / "results" / "d04_plus_e01_trainval_predictions.csv", index=False)
    print(f"chosen on train: limit={lim}% persist={per}")
    for name, p in (("D04", d04), ("D01+E01+R01+D04", comb)):
        s = ev.score_splits(p)
        print(name, "train %.1f validation %.1f" % (s["train"]["composite_score"], s["validation"]["composite_score"]),
              "| PUMP-03/05 alerts:", p[p.asset_id.isin(["PUMP-03", "PUMP-05"]) & (p.fault_type == "IMPELLER_WEAR")].asset_id.tolist())


if __name__ == "__main__":
    main()
