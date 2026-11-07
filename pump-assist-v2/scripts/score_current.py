#!/usr/bin/env python3
"""Score the current pipeline defaults on train and validation (and write the predictions). Optional keyword settings: key=value ..."""
import ast, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev


def main():
    kw = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}
    p = pd.concat([predictions_frame(run_pipeline(pumps_in(s), (s,), with_what_if=False, **kw)[1]) for s in ("train", "validation")], ignore_index=True)
    r = ev.score_splits(p)
    print(kw or "defaults", "| train %.1f (recall %.2f, type %.2f, FA/day %.2f) | validation %.1f (recall %.2f, type %.2f, FA/day %.2f)" % (
        r["train"]["composite_score"], r["train"]["event_recall"], r["train"]["fault_type_recall"], r["train"]["false_alerts_per_asset_day"],
        r["validation"]["composite_score"], r["validation"]["event_recall"], r["validation"]["fault_type_recall"], r["validation"]["false_alerts_per_asset_day"]))
    p.to_csv(ROOT / "results" / "current_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
