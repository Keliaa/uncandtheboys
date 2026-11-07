#!/usr/bin/env python3
"""V03: score the frozen system with and without bearing-envelope alerts (limit fixed from healthy pumps, not tuned on scores)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev


def main():
    out = {}
    for label, on in (("frozen (evidence only)", False), ("+ V03 bearing alerts", True)):
        p = pd.concat([predictions_frame(run_pipeline(pumps_in(s), (s,), v03_alerts=on)[1]) for s in ("train", "validation")], ignore_index=True)
        r = ev.score_splits(p)
        out[label] = p
        print(f"{label}: train {r['train']['composite_score']:.1f} type recall {r['train']['fault_type_recall']:.2f} recall {r['train']['event_recall']:.2f} FA/day {r['train']['false_alerts_per_asset_day']:.2f} | "
              f"validation {r['validation']['composite_score']:.1f} type recall {r['validation']['fault_type_recall']:.2f} recall {r['validation']['event_recall']:.2f} FA/day {r['validation']['false_alerts_per_asset_day']:.2f}", flush=True)
    out["+ V03 bearing alerts"].to_csv(ROOT / "results" / "v03_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
