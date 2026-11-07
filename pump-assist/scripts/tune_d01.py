#!/usr/bin/env python3
"""Choose D01 settings (k, persist) on TRAIN pumps only, then score validation once.

  python scripts/tune_d01.py
Writes results/d01_trainval_predictions.csv and results/d01_grid.csv. Test pumps are never read.
"""
import itertools, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.alert import to_submission_rows
from assist.ingest.features import load_assets, load_features
from assist.detect.zscore import z_frame, to_alerts, detect
from assist.splits import pumps_in
import eval as ev  # baseline-reproduction/scripts/eval.py (official scorer wrapper)

K_GRID, PERSIST_GRID = [4.0, 5.0, 6.0, 8.0], [10, 15, 30]


def run(feats, assets, k, persist):
    rows = to_submission_rows(to_alerts(detect(feats, assets, k, persist)))
    return pd.DataFrame(rows, columns=["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"])


def main():
    assets = load_assets()
    train_ids, val_ids = pumps_in("train"), pumps_in("validation")
    ftr = load_features(train_ids, splits=("train",))
    grid = []
    for k, p in itertools.product(K_GRID, PERSIST_GRID):
        pred = run(ftr, assets, k, p)
        r = ev.score_splits(pred, assets_subset=train_ids)["train"]
        grid.append({"k": k, "persist": p, "train_composite": r["composite_score"], "alerts": len(pred)})
        print(grid[-1], flush=True)
    g = pd.DataFrame(grid).sort_values("train_composite", ascending=False)
    g.to_csv(ROOT / "results" / "d01_grid.csv", index=False)
    best = g.iloc[0]
    k, p = float(best.k), int(best.persist)
    fva = load_features(val_ids, splits=("validation",))
    pred = pd.concat([run(ftr, assets, k, p), run(fva, assets, k, p)], ignore_index=True)
    pred.to_csv(ROOT / "results" / "d01_trainval_predictions.csv", index=False)
    s = ev.score_splits(pred)
    print(f"chosen on train: k={k} persist={p}")
    print(f"train composite {s['train']['composite_score']:.1f}  validation composite {s['validation']['composite_score']:.1f}")


if __name__ == "__main__":
    main()
