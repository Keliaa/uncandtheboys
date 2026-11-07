#!/usr/bin/env python3
"""Learned fault signatures (cosine, kNN, logistic regression) vs the E02 naive Bayes, trained on labelled TRAIN events only.

Two training sets: 'flagged' = the E02 rows (minutes where the detector flags |z| > 8 inside labelled machine events) and 'relaxed' =
steady minutes inside the events with max |z| > 4, which also gives rows to events that never cross 8 (cavitation, restriction, air).
Alerts on train pumps use leave-one-pump-out models; validation uses models trained on all train pumps and is scored once.
No generator values are used. Test pumps are not used."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import numpy as np
import pandas as pd
from assist import _ref  # noqa: F401
from assist.detect.zscore import z_frame
from assist.evidence.bayes import GaussianNB
from assist.evidence.signature import LearnedSignatures
from assist.ingest.features import load_assets, load_features
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
from retrofit_sim.baselines import CH
from tune_e02 import training_rows
import eval as ev

MAX_PER_EVENT = 200


def relaxed_rows(zf: pd.DataFrame, events: pd.DataFrame, thr: float = 4.0):
    """Steady minutes inside labelled machine events with max |z| > thr. Where events overlap, a minute gets the label of the event that
    started last (an escalation such as cavitation on top of a clog is the newer change); pre-existing events have the lowest priority."""
    keys = [k for k in CH if f"z_{k}" in zf]
    ev_m = events[(events.event_class == "MACHINE") & events.observable].copy()
    for c in ("onset_utc", "end_utc"):
        ev_m[c] = pd.to_datetime(ev_m[c], utc=True)
    ev_m["pre"] = ev_m.pre_existing.astype(bool)
    rows, labels, pumps = [], [], []
    for aid, g in zf.groupby("asset_id"):
        es = ev_m[ev_m.asset_id == aid].sort_values(["pre", "onset_utc"], ascending=[False, True])
        Z = g[[f"z_{k}" for k in keys]].to_numpy(float)
        strong = g["steady"].to_numpy(bool) & (np.nan_to_num(np.abs(Z), nan=0.0).max(axis=1) > thr)
        owner = np.full(len(g), -1)
        for n, (_, e) in enumerate(es.iterrows()):
            end = e.end_utc if pd.notna(e.end_utc) else pd.Timestamp("2100-01-01", tz="UTC")
            owner[((g.window_start_utc >= e.onset_utc) & (g.window_start_utc <= end)).to_numpy()] = n
        for n, (_, e) in enumerate(es.iterrows()):
            idx = np.flatnonzero(strong & (owner == n))
            if len(idx) > MAX_PER_EVENT:
                idx = idx[np.linspace(0, len(idx) - 1, MAX_PER_EVENT).astype(int)]
            for i in idx:
                rows.append({CH[k]: float(Z[i, j]) for j, k in enumerate(keys) if np.isfinite(Z[i, j])})
                labels.append(e.fault_type); pumps.append(aid)
    return rows, labels, pumps


def main():
    assets = load_assets()
    tr, va = pumps_in("train"), pumps_in("validation")
    events = pd.read_csv(ROOT.parent / "data" / "full" / "events_train.csv")
    det_tr, _ = run_pipeline(tr, ("train",), use_e02=False, with_what_if=False)
    zf = z_frame(load_features(tr, splits=("train",)), assets)
    sets = {"flagged": training_rows(det_tr, events), "relaxed": relaxed_rows(zf, events)}
    for n, (r, l, p) in sets.items():
        print(n, len(r), pd.Series(l).value_counts().to_dict(), flush=True)
    channels = list(CH.values())
    makers = {"naive Bayes (E02)": lambda: GaussianNB(channels), "cosine": lambda: LearnedSignatures(channels, "cosine"),
              "kNN": lambda: LearnedSignatures(channels, "knn"), "logreg": lambda: LearnedSignatures(channels, "logreg")}
    rows_out = []
    for set_name, (rows, labels, pumps) in sets.items():
        for name, mk in makers.items():
            def fit(excl=None):
                keep = [i for i, p in enumerate(pumps) if p != excl]
                return mk().fit([rows[i] for i in keep], [labels[i] for i in keep])
            full = fit()
            lopo = {p: fit(p) for p in tr}
            ptr = predictions_frame(run_pipeline(tr, ("train",), classifier=lambda a, z, t: lopo[a].predict_proba(z), with_what_if=False)[1])
            pva = predictions_frame(run_pipeline(va, ("validation",), classifier=lambda a, z, t: full.predict_proba(z), with_what_if=False)[1])
            r = ev.score_splits(pd.concat([ptr, pva], ignore_index=True))
            rows_out.append({"training_rows": set_name, "model": name, "train": round(r["train"]["composite_score"], 1), "validation": round(r["validation"]["composite_score"], 1),
                             "type_recall_train": round(r["train"]["fault_type_recall"], 2), "type_recall_val": round(r["validation"]["fault_type_recall"], 2),
                             "class_recall_train": round(r["train"]["class_recall"], 2), "FA_day_train": round(r["train"]["false_alerts_per_asset_day"], 2)})
            print(rows_out[-1], flush=True)
    pd.DataFrame(rows_out).to_csv(ROOT / "results" / "signatures.csv", index=False)


if __name__ == "__main__":
    main()
