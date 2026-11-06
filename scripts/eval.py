#!/usr/bin/env python3
"""Score prediction files on TRAIN and VALIDATION with the official scorer (src/retrofit_sim/scoring.py).

  python scripts/eval.py results/b4_trainval_predictions.csv            # one file holding train + validation alerts
  python scripts/eval.py --b1                                           # regenerate and score baseline B1 (k=6, persist=15)
  python scripts/eval.py preds.csv --compare-b1 --events-table          # side by side with B1, per-event table
  python scripts/eval.py preds.csv --lopo                               # per-pump composite (leave-one-pump-out view)

Prints composite, F1, recall, precision, timeliness, fault-type recall, class recall, false+excess per asset-day and a
per-event table (event_id, true type, detected, predicted type, lead time). Test pumps are never scored or read here.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402

from retrofit_sim.scoring import score  # noqa: E402

DATA = os.path.join(ROOT, "data", "full")
TEST_IDS = {"PUMP-10", "PUMP-11", "PUMP-12"}
KEYS = [("composite", "composite_score", 1), ("F1", "event_f1", 3), ("recall", "event_recall", 3),
        ("precision", "alert_precision", 3), ("timeliness", "timeliness", 3), ("type_rec", "fault_type_recall", 3),
        ("class_rec", "class_recall", 3), ("FA+ex/day", "false_alerts_per_asset_day", 3)]


def load_labels(data=DATA):
    assets = pd.read_csv(os.path.join(data, "assets.csv"))
    ev = {"train": pd.read_csv(os.path.join(data, "events_train.csv")),
          "validation": pd.read_csv(os.path.join(data, "events_validation.csv"))}
    bn = pd.read_csv(os.path.join(data, "benign_windows_train_validation.csv"))
    return assets, ev, bn


def score_splits(pred: pd.DataFrame, data=DATA, assets_subset=None) -> dict:
    """Return {'train': report, 'validation': report, 'mean': composite mean}."""
    assets, ev, bn = load_labels(data)
    pred = pred[~pred["asset_id"].isin(TEST_IDS)]
    out = {}
    for split in ("train", "validation"):
        a = assets if assets_subset is None else assets[assets.asset_id.isin(assets_subset)]
        if not (a.split == split).any():
            continue
        out[split] = score(pred, ev[split], a, bn, split)
    out["mean"] = sum(out[s]["composite_score"] for s in ("train", "validation") if s in out) / max(
        1, sum(s in out for s in ("train", "validation")))
    return out


def score_stress_dir(pred: pd.DataFrame, sdir: str) -> dict:
    """Score predictions on a simulator-generated stress fleet (full private labels written by make_dataset.py)."""
    assets = pd.read_csv(os.path.join(sdir, "public", "assets.csv"))
    ev = pd.read_csv(os.path.join(sdir, "private", "events_stress_all.csv"))
    bn = pd.read_csv(os.path.join(sdir, "public", "benign_windows_stress.csv"))
    return score(pred, ev, assets, bn, "stress")


def per_pump(pred: pd.DataFrame, data=DATA) -> pd.DataFrame:
    assets, ev, bn = load_labels(data)
    rows = []
    for _, a in assets[assets.split.isin(["train", "validation"])].iterrows():
        r = score(pred, ev[a.split], assets[assets.asset_id == a.asset_id], bn, a.split)
        rows.append({"asset_id": a.asset_id, "split": a.split, "control": a.control, "tier": a.instrumentation_tier,
                     **{k: r[m] for k, m, _ in KEYS}, "events": r["events_scored"], "detected": r["true_positive_events"]})
    return pd.DataFrame(rows)


def summary_table(results: dict) -> str:
    lines = ["| method | split | " + " | ".join(k for k, _, _ in KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
    for name, res in results.items():
        for split in ("train", "validation"):
            if split in res:
                r = res[split]
                lines.append(f"| {name} | {split} | " + " | ".join(f"{r[m]:.{d}f}" for _, m, d in KEYS) + " |")
        lines.append(f"| {name} | mean(train,val) composite | {res['mean']:.1f} |" + " |" * (len(KEYS) - 1))
    return "\n".join(lines)


def events_table(res: dict) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation"):
        for e in res.get(split, {}).get("per_event", []):
            rows.append({"split": split, "event_id": e["event_id"], "true_type": e["fault_type"],
                         "detected": "yes" if e["detected"] else "no", "pred_type": e["predicted_fault_type"] or "-",
                         "type_ok": "Y" if e["type_correct"] else "", "class_ok": "Y" if e["class_correct"] else "",
                         "first_alert": (e["first_alert_utc"] or "-")[5:16],
                         "delay_after_detectable_h": None if e["delay_after_detectable_h"] is None else round(e["delay_after_detectable_h"], 2),
                         "lead_to_deadline_h": None if e["lead_time_to_deadline_h"] is None else round(e["lead_time_to_deadline_h"], 1),
                         "timeliness": round(e["timeliness"], 2)})
    return pd.DataFrame(rows)


def b1_predictions(data=DATA, k=6.0, persist=15) -> pd.DataFrame:
    """Baseline B1 with the parameters run_baselines.py selects on train (k=6, persist=15), train + validation only."""
    from retrofit_sim import baselines as B
    assets = pd.read_csv(os.path.join(data, "assets.csv"))
    feat = pd.read_parquet(os.path.join(data, "features_1min.parquet"))
    feat = feat[~feat["asset_id"].isin(TEST_IDS)]
    zf = B.zscores(B.regime_frame(feat, assets))
    return B.b1_threshold(zf, k, persist)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("predictions", nargs="*", help="prediction CSV(s) covering train and/or validation assets")
    p.add_argument("--b1", action="store_true", help="score baseline B1 (regenerated)")
    p.add_argument("--compare-b1", action="store_true", help="also show B1")
    p.add_argument("--events-table", action="store_true")
    p.add_argument("--lopo", action="store_true", help="per-pump scores")
    p.add_argument("--data", default=DATA)
    a = p.parse_args(argv)
    preds = {}
    if a.b1 or a.compare_b1:
        cache = os.path.join(ROOT, "results", "b1_trainval_predictions.csv")
        if not os.path.exists(cache):
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            b1_predictions(a.data).to_csv(cache, index=False)
        preds["B1"] = pd.read_csv(cache)
    for f in a.predictions:
        preds[os.path.basename(f).replace(".csv", "")] = pd.read_csv(f)
    results = {n: score_splits(p_, a.data) for n, p_ in preds.items()}
    print(summary_table(results))
    for n, p_ in preds.items():
        if a.events_table:
            print(f"\nper-event table: {n}")
            print(events_table(results[n]).to_string(index=False))
        if a.lopo:
            print(f"\nper-pump: {n}")
            print(per_pump(p_, a.data).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
