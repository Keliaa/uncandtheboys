#!/usr/bin/env python3
"""D05: operating modes by clustering (K-means, GMM) instead of speed bands, whole system otherwise unchanged.

Reports composite on train and validation, and the normal variation of the detector score (robust spread of max|z| on steady minutes
outside labelled events, after the baseline day; lower means fewer false alerts) per pump, split by speed-band vs cluster modes.
Fixed-speed pumps (PUMP-03, 05, 09) have a single mode with speed bands; with clusters they get several. No label is used to fit.
Test pumps are not used."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import numpy as np
import pandas as pd
from assist import _ref  # noqa: F401
from assist.detect.pca import healthy_mask  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev

FIXED = ["PUMP-03", "PUMP-05", "PUMP-09"]


def normal_variation(det: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    d = det.copy()
    ev_ = events[~events.pre_existing.astype(bool)].copy()     # pre-existing faults cover a whole window and are already inside the baseline
    for c in ("onset_utc", "end_utc"):
        ev_[c] = pd.to_datetime(ev_[c], utc=True)
    ok = pd.Series(True, index=d.index)
    for _, e in ev_.iterrows():
        end = e.end_utc + pd.Timedelta(minutes=30) if pd.notna(e.end_utc) else pd.Timestamp("2100-01-01", tz="UTC")
        ok &= ~((d.asset_id == e.asset_id) & (d.window_start_utc >= e.onset_utc - pd.Timedelta(hours=1)) & (d.window_start_utc <= end))
    d = d[ok & ~d.in_baseline & d.score.notna()]
    return d.groupby("asset_id").score.apply(lambda s: float(np.median(np.abs(s - s.median())) * 1.4826 + s.median()))   # median + robust spread


def main():
    events = pd.concat([pd.read_csv(ROOT.parent / "data" / "full" / "events_train.csv"), pd.read_csv(ROOT.parent / "data" / "full" / "events_validation.csv")])
    out, var = [], {}
    for regime in ("speed", "kmeans", "gmm"):
        preds, dets = [], []
        for s in ("train", "validation"):
            det, al = run_pipeline(pumps_in(s), (s,), regime=regime, with_what_if=False)
            preds.append(predictions_frame(al)); dets.append(det)
        r = ev.score_splits(pd.concat(preds, ignore_index=True))
        v = normal_variation(pd.concat(dets, ignore_index=True), events)
        var[regime] = v
        out.append({"regime": regime, "train": round(r["train"]["composite_score"], 1), "validation": round(r["validation"]["composite_score"], 1),
                    "train_recall": round(r["train"]["event_recall"], 2), "val_recall": round(r["validation"]["event_recall"], 2),
                    "FA_day_train": round(r["train"]["false_alerts_per_asset_day"], 2), "FA_day_val": round(r["validation"]["false_alerts_per_asset_day"], 2),
                    "normal_variation_all": round(float(v.mean()), 2), "normal_variation_fixed_speed": round(float(v.reindex(FIXED).mean()), 2)})
        print(out[-1], flush=True)
        if regime != "speed":
            pd.concat(preds, ignore_index=True).to_csv(ROOT / "results" / f"d05_{regime}_trainval_predictions.csv", index=False)
    pd.DataFrame(out).to_csv(ROOT / "results" / "d05_modes.csv", index=False)
    pd.DataFrame(var).round(2).to_csv(ROOT / "results" / "d05_normal_variation_per_pump.csv")
    print(pd.DataFrame(var).round(2).to_string())


if __name__ == "__main__":
    main()
