#!/usr/bin/env python3
"""E02: Gaussian naive Bayes for machine fault types. Trained on TRAIN events only; validation scored once.

Train-pump alerts use leave-one-pump-out models (the pump's own events are not in its model).
Writes results/e02_trainval_predictions.csv, results/e02_reliability.csv/.png, results/e02_model.json."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import numpy as np
import pandas as pd
from assist import _ref  # noqa: F401
from assist.config import _asset_from_dict, assets_from_csv, load_machine
from assist.detect.benign import benign_minutes
from assist.detect.physics import detect_physics, physics_alerts
from assist.detect.zscore import DEFAULT_K, DEFAULT_PERSIST, detect, to_alerts_with_rules, z_frame
from assist.evidence.bayes import GaussianNB
from assist.ingest.features import load_assets, load_features
from assist.ingest.quality import sensor_alerts, trust_frame
from assist.pipeline import predictions_frame
from assist.recommend.actions import add_actions
from assist.splits import pumps_in
from retrofit_sim.baselines import CH
import eval as ev

GAP = FLAT = 30
TAIL, HS = 60, 2.2     # D06 settings chosen on train
MAX_PER_EVENT = 200


def training_rows(det_train: pd.DataFrame, events: pd.DataFrame):
    """z-vectors of flagged minutes inside labelled machine events of train pumps."""
    rows, labels, pumps = [], [], []
    ev_m = events[(events.event_class == "MACHINE") & events.observable].copy()
    for c in ("onset_utc", "end_utc"):
        ev_m[c] = pd.to_datetime(ev_m[c], utc=True)
    for aid, g in det_train.groupby("asset_id"):
        flagged = g[g.flag & g.z_all.map(bool)]
        es = ev_m[ev_m.asset_id == aid]
        for _, e in es.iterrows():
            end = e.end_utc if pd.notna(e.end_utc) else pd.Timestamp.max.tz_localize("UTC")
            m = flagged[(flagged.window_start_utc >= e.onset_utc) & (flagged.window_start_utc <= end)]
            others = es[es.event_id != e.event_id]
            if len(others):   # drop minutes shared with another machine event (ambiguous label)
                amb = np.zeros(len(m), bool)
                for _, o in others.iterrows():
                    oe = o.end_utc if pd.notna(o.end_utc) else pd.Timestamp.max.tz_localize("UTC")
                    amb |= ((m.window_start_utc >= o.onset_utc) & (m.window_start_utc <= oe)).to_numpy()
                m = m[~amb]
            if len(m) > MAX_PER_EVENT:
                m = m.iloc[np.linspace(0, len(m) - 1, MAX_PER_EVENT).astype(int)]
            rows += list(m.z_all); labels += [e.fault_type] * len(m); pumps += [aid] * len(m)
    return rows, labels, pumps


def main():
    assets, machine = load_assets(), load_machine()
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(ROOT.parent / "data" / "full" / "assets.csv")}
    tr, va = pumps_in("train"), pumps_in("validation")
    ev_tr = pd.read_csv(ROOT.parent / "data" / "full" / "events_train.csv")
    ev_va = pd.read_csv(ROOT.parent / "data" / "full" / "events_validation.csv")
    data = {}
    for name, ids in (("train", tr), ("validation", va)):
        f = load_features(ids, splits=(name,))
        zf = z_frame(f, assets)
        ben = benign_minutes(ids, tail_min=TAIL, hs_limit=HS)
        det = detect(f, assets, DEFAULT_K, DEFAULT_PERSIST, trust_gate=True, zf=zf, benign=ben)
        data[name] = (f, zf, trust_frame(zf, FLAT), ben, det)
    rows, labels, pumps = training_rows(data["train"][4], ev_tr)
    print("training rows", len(rows), pd.Series(labels).value_counts().to_dict())
    channels = list(CH.values())
    full = GaussianNB(channels).fit(rows, labels)
    full.save(ROOT / "results" / "e02_model.json")
    lopo = {p: GaussianNB(channels).fit([r for r, q in zip(rows, pumps) if q != p], [l for l, q in zip(labels, pumps) if q != p])
            if any(q != p for q in pumps) else full for p in tr}

    def preds(name, classifier, gate=False):
        f, zf, tf, ben, det = data[name]
        a = add_actions(to_alerts_with_rules(det, cfg, machine, DEFAULT_K, classifier=classifier, gate=gate), machine)
        a += physics_alerts(detect_physics(f, assets, machine, 2.0, 30, benign=ben), machine)
        a += sensor_alerts(zf, tf, machine, GAP, FLAT)
        return a

    evs = pd.concat([ev_tr, ev_va])
    for c in ("onset_utc", "end_utc"):
        evs[c] = pd.to_datetime(evs[c], utc=True)

    def truth(aid, t):
        w = evs[(evs.asset_id == aid) & (evs.onset_utc <= t) & ((evs.end_utc.isna()) | (evs.end_utc >= t))]
        return set(w.fault_type)

    # Calibrate: temperature T on TRAIN leave-one-pump-out alerts only (Brier score of top probability vs correct type)
    calls = []
    def record(a, z, t):
        calls.append((a, z, t))
        return lopo[a].predict_proba(z)
    preds("train", record)
    from assist.evidence.rules import classify as _c
    best = (9.0, 1.0)
    for T in (1, 2, 5, 10, 20, 50, 100, 300):
        err = []
        for a, z, t in calls:
            d0 = _c(z, cfg[a], machine, DEFAULT_K)
            if d0["event_class"] != "MACHINE":
                continue
            allowed = set(__import__("assist.config", fromlist=["x"]).usable_symptoms(machine, cfg[a]))
            pr = {k: v for k, v in lopo[a].predict_proba_tempered(z, T).items() if k in allowed}
            if not pr:
                continue
            tot = sum(pr.values()); k_, p_ = max(pr.items(), key=lambda kv: kv[1]); p_ /= tot
            err.append((p_ - float(k_ in truth(a, t))) ** 2)
        b = float(np.mean(err)) if err else 9.0
        print(f"T={T}: Brier {b:.3f} (n={len(err)})")
        if b < best[0]:
            best = (b, float(T))
    T = best[1]
    print("temperature chosen on train:", T)
    out = {}
    for label, cls_tr, cls_va, gate in (("E01 rules", None, None, False),
                                  ("E02 naive Bayes (raw)", lambda a, z, t: lopo[a].predict_proba(z), lambda a, z, t: full.predict_proba(z), False),
                                  ("E02 naive Bayes (calibrated)", lambda a, z, t: lopo[a].predict_proba_tempered(z, T), lambda a, z, t: full.predict_proba_tempered(z, T), False),
                                  ("E02 + rule gate", lambda a, z, t: lopo[a].predict_proba_tempered(z, T), lambda a, z, t: full.predict_proba_tempered(z, T), True)):
        a_tr, a_va = preds("train", cls_tr, gate), preds("validation", cls_va, gate)
        p = pd.concat([predictions_frame(a_tr), predictions_frame(a_va)], ignore_index=True)
        s = ev.score_splits(p)
        out[label] = (p, a_tr + a_va)
        print(f"{label}: train {s['train']['composite_score']:.1f} type recall {s['train']['fault_type_recall']:.2f} | "
              f"validation {s['validation']['composite_score']:.1f} type recall {s['validation']['fault_type_recall']:.2f}")
    out["E02 naive Bayes (calibrated)"][0].to_csv(ROOT / "results" / "e02_trainval_predictions.csv", index=False)
    out["E02 + rule gate"][0].to_csv(ROOT / "results" / "e02_gate_trainval_predictions.csv", index=False)
    # How many E02 machine alerts name a type none of whose symptoms match the evidence?
    from assist.evidence.rules import rule_scores
    for label in ("E02 naive Bayes (calibrated)", "E02 + rule gate"):
        bad = tot = 0
        for a in out[label][1]:
            if a.event_class == "MACHINE" and a.fault_type not in ("UNKNOWN", "IMPELLER_WEAR") and a.evidence:
                tot += 1
                z = {e.channel: e.z for e in a.evidence}
                bad += rule_scores(z, cfg[a.asset_id], machine, DEFAULT_K).get(a.fault_type, 0.0) == 0.0
        print(f"{label}: {bad} of {tot} typed machine alerts have evidence that matches none of the type's symptoms")

    # Reliability: stated probability vs observed correct rate for machine alerts (train via leave-one-pump-out + validation)
    recs = []
    for a in out["E02 naive Bayes (calibrated)"][1]:
        if a.event_class != "MACHINE" or a.fault_type == "UNKNOWN" or a.probability == 0 or not a.top_types:
            continue
        t = pd.Timestamp(a.alert_time_utc)
        w = evs[(evs.asset_id == a.asset_id) & (evs.onset_utc <= t) & ((evs.end_utc.isna()) | (evs.end_utc >= t))]
        recs.append({"p": a.probability, "correct": bool((w.fault_type == a.fault_type).any())})
    r = pd.DataFrame(recs)
    if len(r):
        r["bin"] = pd.cut(r.p, [0, 0.4, 0.55, 0.7, 0.85, 1.0])
        rel = r.groupby("bin", observed=True).agg(n=("p", "size"), stated=("p", "mean"), observed=("correct", "mean")).reset_index()
        rel.to_csv(ROOT / "results" / "e02_reliability.csv", index=False)
        print(rel.to_string(index=False))
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(rel.stated, rel.observed, "o-")
        for _, q in rel.iterrows():
            ax.annotate(f"n={int(q.n)}", (q.stated, q.observed), textcoords="offset points", xytext=(5, -10), fontsize=8)
        ax.set_xlabel("stated probability"); ax.set_ylabel("observed correct rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("E02 calibrated reliability")
        fig.tight_layout(); fig.savefig(ROOT / "results" / "e02_reliability.png", dpi=130)


if __name__ == "__main__":
    main()
