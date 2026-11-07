#!/usr/bin/env python3
"""V02: spectral patterns per snapshot, and fault-type recall (IMBALANCE, MISALIGNMENT) with and without spectral typing.

Limits are round numbers (see assist/detect/orders.py), not tuned on scores. Test pumps are never read."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.detect.orders import snapshot_orders
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev

TYPES = ["IMBALANCE", "MISALIGNMENT", "BEARING_WEAR"]


def per_type(r) -> dict:
    rows = []
    for sp in ("train", "validation"):
        for e in r[sp]["per_event"]:
            rows.append({"split": sp, **e})
    d = pd.DataFrame(rows)
    out = {}
    for t in TYPES:
        x = d[d.fault_type == t]
        out[t] = f"{int(x.type_correct.sum())}/{len(x)}"
    return out


def main():
    o = snapshot_orders(pumps_in("train") + pumps_in("validation"), splits=("train", "validation"))
    t = o.groupby("asset_id").pattern.value_counts().unstack(fill_value=0)
    t.to_csv(ROOT / "results" / "v02_patterns.csv")
    print("snapshots by spectral pattern:\n", t.to_string())
    res = {}
    for label, on in (("frozen (evidence only)", False), ("+ V02 spectral typing", True)):
        p = pd.concat([predictions_frame(run_pipeline(pumps_in(s), (s,), v02_typing=on)[1]) for s in ("train", "validation")], ignore_index=True)
        r = ev.score_splits(p)
        res[label] = p
        print(f"{label}: train {r['train']['composite_score']:.1f} type recall {r['train']['fault_type_recall']:.2f} | "
              f"validation {r['validation']['composite_score']:.1f} type recall {r['validation']['fault_type_recall']:.2f} | "
              f"events typed correctly (all train+val): {per_type(r)}", flush=True)
    res["+ V02 spectral typing"].to_csv(ROOT / "results" / "v02_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
