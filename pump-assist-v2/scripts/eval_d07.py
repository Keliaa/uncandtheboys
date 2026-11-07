#!/usr/bin/env python3
"""D07: autoencoder vs PCA (D03), both trained on the same healthy train z-scores and used as the only anomaly detector
(D01 flags off, everything downstream unchanged), at the same limit quantile of healthy minutes. The quantile for each is chosen
on TRAIN pumps only; validation is scored once. Test pumps are not used."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.detect.autoencoder import fit_tier_autoencoder
from assist.detect.pca import fit_tier_model, healthy_mask
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
    healthy, healthy_any = healthy_mask(zf, events), healthy_mask(zf, None)
    tiers = assets[assets.asset_id.isin(tr)].groupby("instrumentation_tier").asset_id.apply(list).to_dict()
    use = {t: (healthy if int((healthy & zf.asset_id.isin(ids)).sum()) >= 1000 else healthy_any) for t, ids in tiers.items()}   # same rule as D03

    def pca(q):
        return {t: fit_tier_model(t, zf, use[t], ids, q) for t, ids in tiers.items()}

    def ae(q):
        return {t: fit_tier_autoencoder(t, zf, use[t], ids, q) for t, ids in tiers.items()}

    def run(split, **kw):
        return predictions_frame(run_pipeline(pumps_in(split), (split,), d01_flags=False, with_what_if=False, **kw)[1])

    rows = []
    for q in QUANTILES:
        for name, kw in (("D03 PCA", {"d03_models": pca(q)}), ("D07 autoencoder", {"d07_models": ae(q)})):
            p = run("train", **kw)
            r = ev.score_splits(p, assets_subset=tr)["train"]
            rows.append({"detector": name, "quantile": q, "split": "train", "composite": round(r["composite_score"], 1), "recall": round(r["event_recall"], 2),
                         "FA_per_asset_day": round(r["false_alerts_per_asset_day"], 2), "alerts": len(p)})
            print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    best = {n: float(g[g.detector == n].sort_values(["composite", "quantile"], ascending=[False, False]).iloc[0].quantile_) if False else
            float(g[g.detector == n].sort_values(["composite", "quantile"], ascending=[False, False]).iloc[0]["quantile"]) for n in ("D03 PCA", "D07 autoencoder")}
    print("chosen on train:", best)
    for name, mk, kwname in (("D03 PCA", pca, "d03_models"), ("D07 autoencoder", ae, "d07_models")):
        m = mk(best[name])
        p_tr, p_va = run("train", **{kwname: m}), run("validation", **{kwname: m})
        r = ev.score_splits(pd.concat([p_tr, p_va], ignore_index=True))
        rows.append({"detector": name, "quantile": best[name], "split": "train (chosen)", "composite": round(r["train"]["composite_score"], 1), "recall": round(r["train"]["event_recall"], 2),
                     "FA_per_asset_day": round(r["train"]["false_alerts_per_asset_day"], 2), "alerts": len(p_tr)})
        rows.append({"detector": name, "quantile": best[name], "split": "validation", "composite": round(r["validation"]["composite_score"], 1), "recall": round(r["validation"]["event_recall"], 2),
                     "FA_per_asset_day": round(r["validation"]["false_alerts_per_asset_day"], 2), "alerts": len(p_va)})
        print(rows[-2], "\n", rows[-1], flush=True)
        pd.concat([p_tr, p_va], ignore_index=True).to_csv(ROOT / "results" / f"d07_{'ae' if 'auto' in name else 'pca'}_only_trainval_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "d07_vs_d03.csv", index=False)


if __name__ == "__main__":
    main()
