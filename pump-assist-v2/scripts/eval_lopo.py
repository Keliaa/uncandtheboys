#!/usr/bin/env python3
"""Leave-one-pump-out for pump-assist-v2, without and with the derived-channel-aware sensor diagnosis (B4 idea #6).

For each of the 9 labelled pumps: the naive Bayes model (E02) is trained on the other 8 pumps' labelled events, the settings (D01 limit k,
D04 head limit) are chosen on those 8 (mean of the train and validation composites there), and the held-out pump is scored alone. Everything
is computed once per setting for all 9 pumps and then filtered (pumps are detected independently). No generator values are used.
Writes results/lopo_v2.csv, results/lopo_v2_events.csv. Test pumps are never run."""
import copy, itertools, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.config import _asset_from_dict, assets_from_csv, load_machine
from assist.detect.benign import benign_minutes
from assist.detect.maintenance import lockout_restarts, maintenance_alerts, restart_steps
from assist.detect.physics import detect_physics, physics_alerts
from assist.detect.zscore import DEFAULT_PERSIST, detect, to_alerts_with_rules, z_frame
from assist.evidence.bayes import GaussianNB
from assist.evidence.derived import explain_by_transmitter
from assist.evidence.episodes import manage_episodes
from assist.evidence.sensor import refine_sensor_types
from assist.ingest.features import load_assets, load_features
from assist.ingest.quality import sensor_alerts, trust_frame
from assist.pipeline import BENIGN_TAIL, FLAT_MIN, GAP_PERSIST, HS_LIMIT, PHYS_PERSIST, predictions_frame
from assist.recommend.actions import add_actions
from retrofit_sim.baselines import CH
from tune_e02 import training_rows
import eval as ev

DEV = [f"PUMP-0{i}" for i in range(1, 10)]
K_GRID, LIM_GRID, STEP, QUIET = [6.0, 8.0, 10.0], [2.0, 3.0, 5.0], 8.0, 1440


def main():
    assets, machine = load_assets(), load_machine()
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(ROOT.parent / "data" / "full" / "assets.csv")}
    events = pd.concat([pd.read_csv(ROOT.parent / "data" / "full" / "events_train.csv"), pd.read_csv(ROOT.parent / "data" / "full" / "events_validation.csv")])
    f = load_features(DEV, splits=("train", "validation"))
    zf = z_frame(f, assets)
    tf = trust_frame(zf, FLAT_MIN)
    ben = benign_minutes(DEV, tail_min=BENIGN_TAIL, hs_limit=HS_LIMIT)
    print("precomputing", flush=True)
    dets = {k: detect(f, assets, k, DEFAULT_PERSIST, trust_gate=True, zf=zf, benign=ben) for k in K_GRID}
    phys = {lim: add_actions(physics_alerts(detect_physics(f, assets, machine, lim, PHYS_PERSIST, benign=ben), machine), machine) for lim in LIM_GRID}
    maint = add_actions(maintenance_alerts(restart_steps(zf, lockout_restarts(zf)), cfg, machine, STEP), machine)
    sens = sensor_alerts(zf, tf, machine, GAP_PERSIST, FLAT_MIN)
    channels = list(CH.values())

    def fit_model(ids):
        d = dets[8.0]
        rows, labels, _ = training_rows(d[d.asset_id.isin(ids)], events[events.asset_id.isin(ids)])
        return GaussianNB(channels).fit(rows, labels)

    def alerts_for(ids, model, k, lim, derived):
        d = dets[k][dets[k].asset_id.isin(ids)]
        a = add_actions(to_alerts_with_rules(d, cfg, machine, k, classifier=lambda p, z, t: model.predict_proba(z), gate=True), machine)
        a = refine_sensor_types(a, zf, machine)
        a += copy.deepcopy([x for x in phys[lim] if x.asset_id in ids]) + copy.deepcopy([x for x in sens if x.asset_id in ids]) \
            + copy.deepcopy([x for x in maint if x.asset_id in ids])
        if derived:
            a = explain_by_transmitter(a, zf, assets, machine)
        return manage_episodes(a, d, quiet_min=QUIET)

    def pooled(alerts, ids):
        p = predictions_frame(alerts)
        return ev.score_splits(p, assets_subset=ids)

    rows, events_out, held_alerts = [], [], {}
    models = {h: fit_model([p for p in DEV if p != h]) for h in DEV}
    for variant, derived in (("v2 without derived-channel diagnosis", False), ("v2 with derived-channel diagnosis", True)):
        for held in DEV:
            rest = [p for p in DEV if p != held]
            best = None
            for k, lim in itertools.product(K_GRID, LIM_GRID):
                s = pooled(alerts_for(rest, models[held], k, lim, derived), rest)["mean"]
                if best is None or s > best[0] + 1e-9:
                    best = (s, k, lim)
            al = alerts_for([held], models[held], best[1], best[2], derived)
            held_alerts.setdefault(variant, []).extend(al)
            r_all = pooled(al, [held])
            r = r_all["train"] if "train" in r_all else r_all["validation"]
            rows.append({"variant": variant, "held_out": held, "limits(k,D04)": f"{best[1]:g},{best[2]:g}", "score_on_other_pumps": round(best[0], 1),
                         "composite": round(r["composite_score"], 1), "recall": round(r["event_recall"], 2), "type_recall": round(r["fault_type_recall"], 2),
                         "class_recall": round(r["class_recall"], 2), "FA_per_day": round(r["false_alerts_per_asset_day"], 2), "events": r["events_scored"],
                         "early_warning_rate": r["early_warning_rate"], "median_lead_h": r["median_lead_time_h"]})
            for e in r["per_event"]:
                events_out.append({"variant": variant, "held_out": held, **{kk: e.get(kk) for kk in ("event_id", "fault_type", "detected", "type_correct", "class_correct", "lead_time_to_deadline_h", "timeliness")}})
            print(rows[-1], flush=True)
    for variant, al in held_alerts.items():
        predictions_frame(al).to_csv(ROOT / "results" / ("lopo_v2_with_derived_predictions.csv" if "with derived" in variant else "lopo_v2_without_derived_predictions.csv"), index=False)
    d = pd.DataFrame(rows)
    means = d.groupby("variant")[["composite", "recall", "type_recall", "class_recall", "FA_per_day"]].mean().round(2).reset_index().assign(held_out="mean")
    pd.concat([d, means], ignore_index=True).to_csv(ROOT / "results" / "lopo_v2.csv", index=False)
    pd.DataFrame(events_out).to_csv(ROOT / "results" / "lopo_v2_events.csv", index=False)
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
