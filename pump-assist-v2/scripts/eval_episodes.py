#!/usr/bin/env python3
"""B4 idea 1: alert management by episodes. Quiet time that closes an episode is chosen on TRAIN pumps only (B4 used 12 h); validation is
scored once. Test pumps are not used."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev

QUIET = [None, 240, 480, 720, 1440]


def run(split, q):
    return predictions_frame(run_pipeline(pumps_in(split), (split,), episodes=q, with_what_if=False)[1])


def main():
    tr = pumps_in("train")
    rows = []
    for q in QUIET:
        p = run("train", q)
        r = ev.score_splits(p, assets_subset=tr)["train"]
        rows.append({"quiet_min": q or "off", "train_composite": round(r["composite_score"], 1), "recall": round(r["event_recall"], 2), "type_recall": round(r["fault_type_recall"], 2),
                     "precision": round(r["alert_precision"], 2), "FA_per_asset_day": round(r["false_alerts_per_asset_day"], 2), "alerts": len(p)})
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(ROOT / "results" / "episodes_grid.csv", index=False)
    best = g[g.quiet_min != "off"].sort_values(["train_composite", "quiet_min"], ascending=[False, False]).iloc[0]
    q = int(best.quiet_min)
    print("chosen on train:", q)
    for label, qq in (("current policy", None), (f"episodes (quiet {q} min)", q)):
        p = pd.concat([run("train", qq), run("validation", qq)], ignore_index=True)
        r = ev.score_splits(p)
        print(f"{label}: train {r['train']['composite_score']:.1f} FA/day {r['train']['false_alerts_per_asset_day']:.2f} | validation {r['validation']['composite_score']:.1f} "
              f"recall {r['validation']['event_recall']:.2f} FA/day {r['validation']['false_alerts_per_asset_day']:.2f}", flush=True)
        if qq:
            p.to_csv(ROOT / "results" / "episodes_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
