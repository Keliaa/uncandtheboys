#!/usr/bin/env python3
"""D02: choose the EWMA limit on TRAIN pumps only, then score validation once. Test pumps are never read.

D02 flags are added to D01 flags (same downstream: trust gate, benign hold, E01/E02, D04, Q01)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev

LIMITS = [None, 3.0, 4.0, 5.0, 7.0]


def run(limit, split):
    return predictions_frame(run_pipeline(pumps_in(split), (split,), d02_limit=limit)[1])


def main():
    tr = pumps_in("train")
    rows = []
    for L in LIMITS:
        p = run(L, "train")
        r = ev.score_splits(p, assets_subset=tr)["train"]
        rows.append({"d02_limit": L if L is not None else "off", "train_composite": r["composite_score"], "recall": r["event_recall"],
                     "FA_per_day": r["false_alerts_per_asset_day"], "alerts": len(p)})
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(ROOT / "results" / "d02_grid.csv", index=False)
    on = g[g.d02_limit != "off"].sort_values("train_composite", ascending=False).iloc[0]
    off = g[g.d02_limit == "off"].iloc[0]
    L = float(on.d02_limit)
    print(f"best D02 limit on train: {L} ({on.train_composite:.1f} vs off {off.train_composite:.1f})")
    for label, lim in (("without D02", None), (f"with D02 (limit {L})", L)):
        pv = run(lim, "validation")
        p = pd.concat([run(lim, "train"), pv], ignore_index=True)
        s = ev.score_splits(p)
        print(f"{label}: train {s['train']['composite_score']:.1f} | validation {s['validation']['composite_score']:.1f} "
              f"recall {s['validation']['event_recall']:.2f} FA/day {s['validation']['false_alerts_per_asset_day']:.2f}")
        if lim is not None:
            p.to_csv(ROOT / "results" / "d02_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
