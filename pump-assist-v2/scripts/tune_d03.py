#!/usr/bin/env python3
"""D03: PCA / Hotelling T2. Fit per tier on healthy TRAIN minutes, choose the limit quantile on train, score validation once.

Compares the frozen system (D01) with D03 alone and D01 + D03. Test pumps are never read."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.detect.pca import fit_tier_model, healthy_mask, pca_flags
from assist.detect.zscore import z_frame
from assist.ingest.features import load_assets, load_features
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev

QUANTILES = [0.99, 0.995, 0.999]


def main():
    assets = load_assets()
    tr, va = pumps_in("train"), pumps_in("validation")
    events = pd.read_csv(ROOT.parent / "data" / "full" / "events_train.csv")
    zf = z_frame(load_features(tr, splits=("train",)), assets)
    healthy = healthy_mask(zf, events)           # outside labelled events and every |z| < 4
    healthy_any = healthy_mask(zf, None)         # fallback when pre-existing faults cover a tier's whole window
    tiers = assets[assets.asset_id.isin(tr)].groupby("instrumentation_tier").asset_id.apply(list).to_dict()
    use = {}
    for t, ids in tiers.items():
        n = int((healthy & zf.asset_id.isin(ids)).sum())
        use[t] = healthy if n >= 1000 else healthy_any
        print(f"{t}: {n} healthy minutes outside events" + ("" if n >= 1000 else f" -> fallback to |z|<4 only ({int((healthy_any & zf.asset_id.isin(ids)).sum())})"))

    def models(q):
        return {t: fit_tier_model(t, zf, use[t], ids, q) for t, ids in tiers.items()}

    def run(split, m, d01=True):
        return predictions_frame(run_pipeline(pumps_in(split), (split,), d03_models=m, d01_flags=d01)[1])

    base = run("train", None)
    rows = [{"quantile": "off", "train_composite": ev.score_splits(base, assets_subset=tr)["train"]["composite_score"], "alerts": len(base)}]
    print(rows[-1], flush=True)
    for q in QUANTILES:
        p = run("train", models(q))
        rows.append({"quantile": q, "train_composite": ev.score_splits(p, assets_subset=tr)["train"]["composite_score"], "alerts": len(p)})
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(ROOT / "results" / "d03_grid.csv", index=False)
    q = float(g[g["quantile"] != "off"].sort_values("train_composite", ascending=False).iloc[0]["quantile"])
    print("chosen on train:", q)
    m = models(q)
    for name, mm, d01 in (("D01 only (frozen)", None, True), ("D03 only", m, False), ("D01 + D03", m, True)):
        p = pd.concat([run("train", mm, d01), run("validation", mm, d01)], ignore_index=True)
        s = ev.score_splits(p)
        print(f"{name}: train {s['train']['composite_score']:.1f} recall {s['train']['event_recall']:.2f} FA/day {s['train']['false_alerts_per_asset_day']:.2f} | "
              f"validation {s['validation']['composite_score']:.1f} recall {s['validation']['event_recall']:.2f} FA/day {s['validation']['false_alerts_per_asset_day']:.2f}", flush=True)
        if name == "D03 only":
            p.to_csv(ROOT / "results" / "d03_only_trainval_predictions.csv", index=False)
        if name == "D01 + D03":
            p.to_csv(ROOT / "results" / "d03_union_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
