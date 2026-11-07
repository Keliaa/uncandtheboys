#!/usr/bin/env python3
"""T05: leave-one-site-out. Train pumps form four sites: Vessel A (01, 02, 03), Plant C (04, 05), Plant D (06), Vessel B (07).

For each held-out site: limits (D01 k, D04 limit, T02 step) are chosen on the other three sites, the naive Bayes model (E02)
is trained on their labelled events, and the held-out site is scored once. Validation and test pumps are not used.
The 'frozen' column is the frozen system (settings chosen on all train pumps, so the held-out site was seen) for comparison."""
import itertools, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import numpy as np
import pandas as pd
from assist import _ref  # noqa: F401
from assist.config import _asset_from_dict, assets_from_csv, load_machine
from assist.detect.benign import benign_minutes
from assist.detect.maintenance import lockout_restarts, maintenance_alerts, restart_steps
from assist.detect.physics import detect_physics, physics_alerts
from assist.detect.zscore import DEFAULT_PERSIST, detect, to_alerts_with_rules, z_frame
from assist.evidence.bayes import GaussianNB
from assist.ingest.features import load_assets, load_features
from assist.ingest.quality import sensor_alerts, trust_frame
from assist.pipeline import BENIGN_TAIL, FLAT_MIN, GAP_PERSIST, HS_LIMIT, PHYS_PERSIST, predictions_frame
from assist.recommend.actions import add_actions
from assist.splits import pumps_in
from retrofit_sim.baselines import CH
from tune_e02 import training_rows
import eval as ev

SITES = {"Vessel A": ["PUMP-01", "PUMP-02", "PUMP-03"], "Plant C": ["PUMP-04", "PUMP-05"], "Plant D": ["PUMP-06"], "Vessel B": ["PUMP-07"]}
K_GRID, PHYS_GRID, STEP_GRID = [6.0, 8.0, 10.0], [2.0, 3.0, 5.0], [4.0, 8.0, 12.0]
FROZEN = (8.0, 2.0, 8.0)


def main():
    assets, machine = load_assets(), load_machine()
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(ROOT.parent / "data" / "full" / "assets.csv")}
    tr = pumps_in("train")
    assert sorted(p for v in SITES.values() for p in v) == sorted(tr)
    events = pd.read_csv(ROOT.parent / "data" / "full" / "events_train.csv")
    f = load_features(tr, splits=("train",))
    zf = z_frame(f, assets)
    tf = trust_frame(zf, FLAT_MIN)
    ben = benign_minutes(tr, tail_min=BENIGN_TAIL, hs_limit=HS_LIMIT)
    channels = list(CH.values())
    print("precomputing detectors", flush=True)
    dets = {k: detect(f, assets, k, DEFAULT_PERSIST, trust_gate=True, zf=zf, benign=ben) for k in K_GRID}
    phys = {lim: add_actions(physics_alerts(detect_physics(f, assets, machine, lim, PHYS_PERSIST, benign=ben), machine), machine) for lim in PHYS_GRID}
    steps = restart_steps(zf, lockout_restarts(zf))
    maint = {z: add_actions(maintenance_alerts(steps, cfg, machine, z), machine) for z in STEP_GRID}
    sens = sensor_alerts(zf, tf, machine, GAP_PERSIST, FLAT_MIN)

    def alerts_for(site_ids, model, k, lim, step):
        clf = None if model is None else (lambda a, z, t: model.predict_proba(z))
        d = dets[k]
        d = d[d.asset_id.isin(site_ids)]
        a = add_actions(to_alerts_with_rules(d, cfg, machine, k, classifier=clf, gate=True), machine)
        keep = lambda L: [x for x in L if x.asset_id in site_ids]
        return a + keep(phys[lim]) + keep(sens) + keep(maint[step])

    def score(alerts, ids):
        p = predictions_frame(alerts)
        return ev.score_splits(p, assets_subset=ids)["train"]["composite_score"] if len(p) else 0.0

    def fit_model(ids):
        d = dets[8.0]
        rows, labels, _ = training_rows(d[d.asset_id.isin(ids)], events[events.asset_id.isin(ids)])
        return GaussianNB(channels).fit(rows, labels)

    b1 = pd.read_csv(ROOT.parent / "baseline-reproduction" / "results" / "b1_trainval_predictions.csv")   # B1 (k=6, persist 15), tuned on train
    full_model = fit_model(tr)
    frozen_all = alerts_for(tr, full_model, *FROZEN)
    print("frozen system, all train pumps: %.1f" % score(frozen_all, tr), flush=True)
    rows = []
    for site, held in SITES.items():
        rest = [p for p in tr if p not in held]
        model = fit_model(rest)
        best = None
        for k, lim, st in itertools.product(K_GRID, PHYS_GRID, STEP_GRID):
            s = score(alerts_for(rest, model, k, lim, st), rest)
            if best is None or s > best[0] + 1e-9:
                best = (s, k, lim, st)
        s_out = score(alerts_for(held, model, *best[1:]), held)
        s_frozen = score([a for a in frozen_all if a.asset_id in held], held)
        rows.append({"held_out_site": site, "pumps": ",".join(held), "limits_chosen(k,D04,T02)": f"{best[1]:g},{best[2]:g},{best[3]:g}",
                     "score_on_other_sites": round(best[0], 1), "score_held_out": round(s_out, 1), "frozen_system_same_pumps": round(s_frozen, 1),
                     "B1_baseline_same_pumps": round(ev.score_splits(b1, assets_subset=held)["train"]["composite_score"], 1)})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    mean = {"held_out_site": "mean", "pumps": "", "limits_chosen(k,D04,T02)": "", "score_on_other_sites": round(t.score_on_other_sites.mean(), 1),
            "score_held_out": round(t.score_held_out.mean(), 1), "frozen_system_same_pumps": round(t.frozen_system_same_pumps.mean(), 1),
            "B1_baseline_same_pumps": round(t.B1_baseline_same_pumps.mean(), 1)}
    t = pd.concat([t, pd.DataFrame([mean])], ignore_index=True)
    t.to_csv(ROOT / "results" / "t05_sites.csv", index=False)
    md = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    (ROOT / "results" / "t05_sites.md").write_text("\n".join(md) + "\n")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
