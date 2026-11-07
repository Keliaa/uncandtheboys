#!/usr/bin/env python3
"""C01: score each tested option on train and validation and compare with baselines B0-B3.

Writes results/results.csv (one row per option and split) and results/results.md. Test pumps are never read.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
import eval as ev

OPTIONS = {"D01_zscore": "results/d01_trainval_predictions.csv",
           "D01+E01+R01": "results/e01_trainval_predictions.csv",
           "+D04": "results/d04_plus_e01_trainval_predictions.csv",
           "+D04+Q01": "results/q01_trainval_predictions.csv",
           "+D04+Q01+D06": "results/d06_trainval_predictions.csv",
           "+D04+Q01+D06+E02": "results/e02_trainval_predictions.csv"}
KEYS = ["composite_score", "event_f1", "event_recall", "alert_precision", "false_alerts_per_asset_day",
        "timeliness", "fault_type_recall", "class_recall"]


def main():
    rows = []
    for name, path in OPTIONS.items():
        pred = pd.read_csv(ROOT / path)
        s = ev.score_splits(pred)
        for sp in ("train", "validation"):
            rows.append({"option": name, "split": sp, **{k: s[sp][k] for k in KEYS}})
    base = pd.read_csv(ROOT / "original-challenge/baselines/results/baseline_metrics.csv")
    for _, r in base.iterrows():
        rows.append({"option": r["baseline"], "split": r["split"], **{k: r[k] for k in KEYS}})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "results.csv", index=False)
    show = df.copy()
    show[KEYS] = show[KEYS].round(3)
    md = ["| option | split | composite | F1 | recall | precision | FA+ex/day | timeliness | type recall | class recall |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in show.iterrows():
        md.append("| " + " | ".join([r.option, r.split] + [str(r[k]) for k in KEYS]) + " |")
    (ROOT / "results" / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
