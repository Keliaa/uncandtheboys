#!/usr/bin/env python3
"""T01: static vs always-learning vs guarded baseline, scored on train and validation (same settings for every pump).

Also lists the detection of slow faults (BEARING_WEAR, SUCTION_STRAINER_CLOGGING) per variant. Test pumps are never read."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev

SLOW = ["BEARING_WEAR", "SUCTION_STRAINER_CLOGGING"]


def main():
    rows = []
    for mode in ("static", "always", "guarded"):
        p = pd.concat([predictions_frame(run_pipeline(pumps_in(s), (s,), baseline=mode)[1]) for s in ("train", "validation")], ignore_index=True)
        p.to_csv(ROOT / "results" / f"t01_{mode}_trainval_predictions.csv", index=False)
        r = ev.score_splits(p)
        pe = pd.DataFrame([e for sp in ("train", "validation") for e in r[sp]["per_event"]])
        slow = pe[pe.fault_type.isin(SLOW)]
        rows.append({"variant": mode, "train": round(r["train"]["composite_score"], 1), "validation": round(r["validation"]["composite_score"], 1),
                     "recall_train": round(r["train"]["event_recall"], 2), "recall_val": round(r["validation"]["event_recall"], 2),
                     "FA_day_train": round(r["train"]["false_alerts_per_asset_day"], 2), "FA_day_val": round(r["validation"]["false_alerts_per_asset_day"], 2),
                     "slow_faults_detected": f"{int(slow.detected.sum())}/{len(slow)}", "alerts": len(p)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "t01_variants.csv", index=False)


if __name__ == "__main__":
    main()
