#!/usr/bin/env python3
"""D01 flags + E01 rules. Settings come from D01 (chosen on train). Scores train and validation; test is never read."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.alert import to_submission_rows, SUBMISSION_COLUMNS
from assist.config import assets_from_csv, _asset_from_dict, load_machine
from assist.ingest.features import load_assets, load_features
from assist.detect.zscore import detect, to_alerts_with_rules, DEFAULT_K, DEFAULT_PERSIST
from assist.splits import pumps_in
import eval as ev


def predict(ids, split, assets, cfg, machine):
    det = detect(load_features(ids, splits=(split,)), assets, DEFAULT_K, DEFAULT_PERSIST)
    return pd.DataFrame(to_submission_rows(to_alerts_with_rules(det, cfg, machine, DEFAULT_K)), columns=SUBMISSION_COLUMNS)


def main():
    assets, machine = load_assets(), load_machine()
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(ROOT / "data" / "full" / "assets.csv")}
    pred = pd.concat([predict(pumps_in("train"), "train", assets, cfg, machine),
                      predict(pumps_in("validation"), "validation", assets, cfg, machine)], ignore_index=True)
    pred.to_csv(ROOT / "results" / "e01_trainval_predictions.csv", index=False)
    s = ev.score_splits(pred)
    for sp in ("train", "validation"):
        r = s[sp]
        print(sp, {k: round(r[k], 3) for k in ("composite_score", "event_recall", "alert_precision", "fault_type_recall", "class_recall")})
    print("alerts", len(pred), pred.fault_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
