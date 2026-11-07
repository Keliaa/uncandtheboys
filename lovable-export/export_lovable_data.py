#!/usr/bin/env python3
"""Build every data file the Lovable operator cockpit reads, validate it, and optionally push it.

    python lovable-export/export_lovable_data.py --out ../cockpit/public/data            # build + validate
    python lovable-export/export_lovable_data.py --out ../cockpit/public/data --push     # ... and commit + push

Writes manifest.json, the three scene bundles (operator-demo-replay/1, exactly as operator-demo/prepare_demo.py builds
them, plus plant_alarms), fleet.json (fleet-replay/1) and impact.json (impact/1). The contract is in README.md.
B4 runs unchanged with its frozen parameters; regenerated alerts must equal the frozen prediction files.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from common import pdm  # noqa: E402

BUNDLE_LABEL = "Recorded replay · synthetic pump data"
FLEET_LABEL = "Fleet replay · synthetic pump data"
IMPACT_LABEL = "Impact estimates · synthetic data, editable assumptions"
ASSUMPTIONS = {"labour_rate_eur_h": 80, "downtime_cost_eur_h": 1500, "repair_unplanned_eur": 9000,
               "repair_planned_eur": 4000, "unplanned_stop_h": 10, "planned_stop_h": 3, "false_alert_check_eur": 120,
               "source": "organiser business_case.py defaults + team labour rate"}
MAX_SIMULATED = 32


class ExportError(RuntimeError):
    pass


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------ pump runs
class Pump:
    def __init__(self, a, inp, pred, f, alarms, split, fz=None):
        self.a, self.inp, self.pred, self.f, self.alarms, self.split, self.fz = a, inp, pred, f, alarms, split, fz
        self.state = None


def _alarms_for(alarms: pd.DataFrame, aid: str) -> pd.DataFrame:
    return alarms[alarms.asset_id == aid][["tag", "alarm_type", "raised_utc", "cleared_utc"]].reset_index(drop=True)


def run_benchmark(data: Path, params: dict) -> list[Pump]:
    b4 = C.b4_module()
    assets = pd.read_csv(data / "assets.csv")
    alarms = pd.read_csv(data / "alarms.csv")
    tv = pd.read_csv(C.FROZEN_TRAINVAL)
    te = pd.read_csv(C.FROZEN_TEST)
    out = []
    for _, a in assets.iterrows():
        aid = a.asset_id
        inp = b4.load_inputs(str(data), [aid])
        pred, f = C.run_b4(inp, params)
        if a.split == "test":
            cols = list(te.columns)
            x = pred[cols].reset_index(drop=True).astype(str)
            y = te[te.asset_id == aid][cols].reset_index(drop=True).astype(str)
            if not x.equals(y):
                raise ExportError(f"{aid}: regenerated B4 alerts differ from {C.FROZEN_TEST.name}")
            fz = None
        else:
            try:
                fz = pdm.check_against_frozen(pred, tv, aid)
            except pdm.PrepError as e:
                raise ExportError(str(e))
        log(f"  {aid} ({a.split}): {len(pred)} B4 alerts, identical to the frozen predictions")
        out.append(Pump(a, inp, pred, f, _alarms_for(alarms, aid), a.split, fz))
    return out


def stress_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "public" / "assets.csv").exists())


def sim_id(seed_dir: Path, aid: str) -> str:
    return f"SIM-{seed_dir.name.split('_')[-1]}-{aid.split('-')[-1]}"


def run_stress(root: Path, params: dict, limit: int) -> tuple[list[Pump], dict]:
    """Up to `limit` simulated pumps (renamed SIM-<seed>-<nn>); returns pumps and raw B4 predictions per seed dir."""
    b4 = C.b4_module()
    pumps, raw = [], {}
    for d in stress_dirs(root):
        pub = d / "public"
        assets = pd.read_csv(pub / "assets.csv")
        alarms = pd.read_csv(pub / "alarms.csv") if (pub / "alarms.csv").exists() else pd.DataFrame(
            columns=["asset_id", "tag", "alarm_type", "raised_utc", "cleared_utc"])
        preds = []
        for _, a in assets.iterrows():
            inp = b4.load_inputs(str(pub), [a.asset_id])
            pred, f = C.run_b4(inp, params)
            preds.append(pred)
            if len(pumps) < limit:
                new = sim_id(d, a.asset_id)
                a2 = a.copy()
                a2["asset_id"] = new
                a2["site"] = f"Simulated site {d.name.split('_')[-1]} (fictional)"
                p2 = pred.copy()
                p2["asset_id"] = new
                pumps.append(Pump(a2, inp, p2, f, _alarms_for(alarms, a.asset_id), "simulated"))
        raw[d] = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
        log(f"  {d.name}: {len(assets)} simulated pumps, {len(raw[d])} B4 alerts")
    return pumps, raw


# ------------------------------------------------------------------------------------------------ scenes
def build_scenes(pumps: dict, data: Path, params: dict) -> list[tuple[str, dict]]:
    """The operator-demo scenes, selected and validated by prepare_demo's own deterministic rules."""
    b4 = C.b4_module()
    benign = pd.read_csv(data / "benign_windows_train_validation.csv")
    events = pd.concat([pd.read_csv(data / n) for n in ("events_train.csv", "events_validation.csv")],
                       ignore_index=True)        # used only to validate the scene choice, never exported
    P = {**b4.PARAMS, **params}
    out = []
    for sc in pdm.SCENES:
        p = pumps[sc["asset"]]
        mon = pdm.monitoring_frame(b4, p.f, p.a, P)
        cal_end = C.utc(p.a.window_start_utc) + pd.Timedelta(hours=params["cal_hours"])
        if sc["kind"] == "quiet":
            win, _ = pdm.select_quiet_window(benign, p.a.asset_id, p.pred, p.f, mon, cal_end)
            start, end, focus = win["start"], win["end"], None
            extra = {"context_note": f"Recorded operating change: {win['note']} "
                                     f"({C.utc(win['label_start_utc']).strftime('%H:%M')}–"
                                     f"{C.utc(win['label_end_utc']).strftime('%H:%M')} UTC)."}
        else:
            pdm.validate_alert_scene(sc, p.pred, events)
            focus = C.utc(sc["alert_time"])
            start = focus - pd.Timedelta(minutes=sc["lead_min"])
            end = focus + pd.Timedelta(minutes=sc["tail_min"])
            extra = {}
        b = pdm.build_bundle(b4, sc, p.a, p.f, mon, p.inp["ctx"], p.fz, start, end, focus, extra)
        lb = C.utc(b["scene"]["lookback_start_utc"])
        al = p.alarms
        r = pd.to_datetime(al.raised_utc, utc=True)
        rows = []
        for _, x in al[((r >= lb) & (r <= end)).values].sort_values("raised_utc").iterrows():
            cl = x.cleared_utc if isinstance(x.cleared_utc, str) and C.utc(x.cleared_utc) <= end else None
            rows.append({"tag": x.tag, "alarm_type": x.alarm_type, "raised_utc": C.iso(x.raised_utc),
                         "cleared_utc": None if cl is None else C.iso(cl)})   # clearing after the scene end is unknown
        b["plant_alarms"] = rows
        b["label"] = BUNDLE_LABEL
        out.append((f"scene_{sc['chapter']}_{sc['id']}.json", C.clean(b)))
        log(f"  scene {sc['chapter']} {sc['id']}: {len(b['minutes']['t'])} minutes, {len(b['alerts'])} alerts, "
            f"{len(rows)} plant alarms")
    return out


# ------------------------------------------------------------------------------------------------ fleet
FLEET_BUDGET = 2_500_000


def build_fleet(pumps: list[Pump], params: dict) -> tuple[dict, list[Pump]]:
    """All benchmark pumps plus simulated pumps in order; simulated pumps are dropped from the end (deterministically)
    until the file fits the size budget. Returns the document and the pumps it contains."""
    assets = []
    for p in pumps:
        p.state = C.PumpState(p.a, p.f, p.pred, p.alarms, params)
        snaps = [C.snapshot(p.state, o) for o in range(0, C.WEEK_MIN + 1, C.SNAPSHOT_MIN)]
        assets.append({"asset_id": p.a.asset_id, "site": p.a.site, "system": p.a.system, "control": p.a.control,
                       "instrumentation_tier": p.a.instrumentation_tier, "split": p.split, "snapshots": snaps})
    kept = list(pumps)
    while True:
        doc = C.clean({"format": "fleet-replay/1", "label": FLEET_LABEL, "aligned": "monitoring_day",
                       "assets": assets[:len(kept)]})
        if len(C.dumps(doc).encode()) < FLEET_BUDGET or kept[-1].split != "simulated":
            return doc, kept
        kept.pop()


# ------------------------------------------------------------------------------------------------ impact
def classify_maintenance(work_type: str, obs: str) -> str:
    """no_finding / light / justified / other (lubrication is always 'other': it is always needed)."""
    o = str(obs).strip()
    if work_type == "ROUTINE_INSPECTION" and "no abnormal findings" in o.lower():
        return "no_finding"
    if work_type == "STRAINER_CLEANING":
        return {"Clean.": "no_finding", "Light debris.": "light", "Moderate fouling.": "justified"}.get(o, "other")
    if work_type == "VIBRATION_ROUTE_MEASUREMENT" and "zone A/B" in o:
        return "no_finding"
    if work_type == "ALIGNMENT_CHECK" and o == "Within tolerance.":
        return "no_finding"
    if work_type == "COUPLING_INSPECTION" and o.endswith("checked, OK."):
        return "no_finding"
    if work_type == "INSTRUMENT_CHECK" and o.endswith("loop-checked, OK."):
        return "no_finding"
    return "other"


def maintenance_summary(data: Path) -> dict:
    m = pd.read_csv(data / "maintenance.csv")
    s, e = pd.to_datetime(m.start_utc, utc=True), pd.to_datetime(m.end_utc, utc=True)
    m["hours"] = (e - s) / pd.Timedelta(hours=1)
    m["cls"] = [classify_maintenance(w, o) for w, o in zip(m.work_type, m.operator_observation)]
    m["planned_b"] = m.planned.astype(str).str.lower() == "true"
    nf = m.cls == "no_finding"
    years = (e.max() - s.min()) / pd.Timedelta(days=365.25)
    fleet = {"records": len(m), "hours": m.hours.sum(), "no_finding_records": int(nf.sum()),
             "no_finding_hours": m.hours[nf].sum(), "no_finding_share_hours": m.hours[nf].sum() / m.hours.sum(),
             "light_finding_records": int((m.cls == "light").sum()),
             "justified_records": int((m.cls == "justified").sum()),
             "corrective_records": int((~m.planned_b).sum())}
    by_type = [{"work_type": w, "records": len(g), "hours": g.hours.sum(),
                "no_finding_records": int((g.cls == "no_finding").sum()),
                "no_finding_hours": g.hours[g.cls == "no_finding"].sum()} for w, g in m.groupby("work_type")]
    by_pump = [{"asset_id": a, "records": len(g), "hours": g.hours.sum(),
                "no_finding_hours": g.hours[g.cls == "no_finding"].sum(),
                "no_finding_eur_per_year": g.hours[g.cls == "no_finding"].sum() * ASSUMPTIONS["labour_rate_eur_h"] / years}
               for a, g in m.groupby("asset_id")]
    st = m[m.work_type == "STRAINER_CLEANING"]
    strainer = {"cleanings": len(st), "found_clean": int((st.cls == "no_finding").sum()),
                "found_light": int((st.cls == "light").sum()), "found_moderate": int((st.cls == "justified").sum())}
    return {"period": {"start_utc": C.iso(s.min()), "end_utc": C.iso(e.max())}, "fleet": fleet,
            "by_work_type": by_type, "by_pump": by_pump, "strainer": strainer}


def _eval_module():
    spec = importlib.util.spec_from_file_location("b4_eval", C.B4_DIR / "scripts" / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics(r: dict) -> dict:
    return {"composite": r["composite_score"], "event_recall": r["event_recall"],
            "fault_type_recall": r["fault_type_recall"], "class_recall": r["class_recall"],
            "false_excess_alerts_per_asset_day": r["false_alerts_per_asset_day"],
            "median_lead_time_to_deadline_h": r.get("median_lead_time_h"),
            "median_delay_after_detectable_h": r.get("median_delay_after_detectable_h"), "events": r["events_scored"]}


def _pooled(reports: list) -> dict:
    """Mean over stress seeds (as in RESULTS.md); medians pooled over all scored events."""
    keys = ["composite", "event_recall", "fault_type_recall", "class_recall", "false_excess_alerts_per_asset_day"]
    ms = [_metrics(r) for r in reports]
    out = {k: float(np.mean([m[k] for m in ms])) for k in keys}
    pe = pd.DataFrame([e for r in reports for e in r["per_event"]])
    lead = pe["lead_time_to_deadline_h"].dropna() if len(pe) else pd.Series(dtype=float)
    dly = pe["delay_after_detectable_h"].dropna() if len(pe) else pd.Series(dtype=float)
    out["median_lead_time_to_deadline_h"] = float(lead.median()) if len(lead) else None
    out["median_delay_after_detectable_h"] = float(dly.median()) if len(dly) else None
    out["events"] = int(sum(m["events"] for m in ms))
    out["fleets"] = len(reports)
    return out


def detector_summary(data: Path, stress_raw: dict) -> dict:
    ev = _eval_module()
    res = {}
    for name, pred in (("b1", pd.read_csv(C.B1_TRAINVAL)), ("b4", pd.read_csv(C.FROZEN_TRAINVAL))):
        s = ev.score_splits(pred, str(data))
        res[name] = {"train": _metrics(s["train"]), "validation": _metrics(s["validation"])}
    if stress_raw:
        b4r, b1r = [], []
        for d, pred in stress_raw.items():
            b4r.append(ev.score_stress_dir(pred, str(d)))
            b1r.append(ev.score_stress_dir(ev.b1_predictions(str(d / "public")), str(d)))
        res["b4"]["simulated"] = _pooled(b4r)
        res["b1"]["simulated"] = _pooled(b1r)
    return res


def tier_ablation() -> list:
    t = pd.read_csv(C.TIER_ABLATION)
    return [{"tier": r.tier, "composite": r.composite, "event_recall": r.recall, "fault_type_recall": r.type_rec}
            for _, r in t.iterrows()]


def exposure(pumps: list[Pump]) -> list:
    out = []
    for p in pumps:
        end = (C.utc(p.a.window_end_utc) - C.utc(p.a.window_start_utc)) / pd.Timedelta(minutes=1)
        hrs = sum(e - s for s, e in C.open_intervals(p.state, end)) / 60
        out.append({"asset_id": p.a.asset_id, "open_alert_hours": hrs, "alerts": len(p.pred),
                    "machine_alerts": int((p.pred.event_class == "MACHINE").sum()),
                    "sensor_alerts": int((p.pred.event_class == "SENSOR").sum())})
    return out


# ------------------------------------------------------------------------------------------------ main
def export(out: Path, data: Path, stress_root: Path, max_sim: int) -> dict:
    t0 = time.time()
    sha, commit_ts = C.git_head()
    params = C.frozen_params()
    log("running B4 (frozen parameters) on the benchmark pumps ...")
    bench = run_benchmark(data, params)
    log(f"running B4 on simulated pumps from {stress_root} ...")
    sim, stress_raw = run_stress(stress_root, params, max_sim)
    by_id = {p.a.asset_id: p for p in bench}
    log("building scene bundles ...")
    scenes = build_scenes(by_id, data, params)
    log("building fleet.json ...")
    fleet, in_fleet = build_fleet(bench + sim, params)
    n_sim = sum(p.split == "simulated" for p in in_fleet)
    log(f"  fleet: {len(in_fleet)} pumps ({n_sim} of {len(sim)} simulated kept within the size budget)")
    log("building impact.json (scoring B1 and B4) ...")
    impact = C.clean({"format": "impact/1", "label": IMPACT_LABEL, "assumptions": ASSUMPTIONS,
                      "maintenance": maintenance_summary(data), "detector": detector_summary(data, stress_raw),
                      "tier_ablation": tier_ablation(), "per_pump_exposure": exposure(in_fleet)})
    manifest = {"format": pdm.BUNDLE_FORMAT, "label": BUNDLE_LABEL, "generated_utc": commit_ts, "source_commit": sha,
                "scenes": [{"chapter": b["scene"]["chapter"], "id": n[:-5], "file": n} for n, b in scenes],
                "fleet": "fleet.json", "impact": "impact.json"}
    files = {"manifest.json": json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
             **{n: C.dumps(b) for n, b in scenes}, "fleet.json": C.dumps(fleet), "impact.json": C.dumps(impact)}
    out.mkdir(parents=True, exist_ok=True)
    for n, txt in files.items():
        (out / n).write_text(txt, encoding="utf-8")
    log(f"wrote {len(files)} files to {out} in {time.time() - t0:.0f} s (source commit {sha})")
    return {"sha": sha, "files": list(files)}


def push(out: Path, sha: str) -> None:
    repo = Path(subprocess.run(["git", "-C", str(out), "rev-parse", "--show-toplevel"], capture_output=True,
                               text=True, check=True).stdout.strip())
    rel = out.resolve().relative_to(repo.resolve())
    if rel.as_posix() != "public/data":
        raise ExportError(f"refusing to push: --out must be <cockpit repo>/public/data, got {rel}")
    g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout
    g("pull", "--ff-only")
    g("add", "--", "public/data")
    if not g("diff", "--cached", "--name-only").strip():
        log("cockpit data unchanged; nothing to push")
        return
    other = [x for x in g("diff", "--cached", "--name-only").split() if not x.startswith("public/data/")]
    if other:
        raise ExportError(f"refusing to commit files outside public/data: {other}")
    g("commit", "-m", f"data: refresh cockpit exports ({sha})")
    g("push")                                             # plain fast-forward push, never forced
    log(f"pushed to {g('remote', 'get-url', 'origin').strip()} ({g('rev-parse', '--short', 'HEAD').strip()})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(C.REPO.parent / "cockpit" / "public" / "data"))
    p.add_argument("--data", default=str(C.DATA_DIR))
    p.add_argument("--stress-root", default=str(C.STRESS_ROOT),
                   help="directory holding stress_<seed>/ fleets (physics-context-detector/build by default)")
    p.add_argument("--max-simulated", type=int, default=MAX_SIMULATED)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--push", action="store_true", help="commit public/data in the cockpit repo and push")
    a = p.parse_args(argv)
    out = Path(a.out).resolve()
    res = export(out, Path(a.data).resolve(), Path(a.stress_root), a.max_simulated)
    if not a.no_validate:
        import validate
        if validate.main([str(out), "--data", a.data, "--stress-root", a.stress_root]) != 0:
            raise ExportError("validation failed; not pushing")
    if a.push:
        push(out, res["sha"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ExportError, pdm.PrepError) as e:
        print(f"\nEXPORT FAILED: {e}", file=sys.stderr)
        sys.exit(2)
