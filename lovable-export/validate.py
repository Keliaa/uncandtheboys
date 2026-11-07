#!/usr/bin/env python3
"""Contract checks for the Lovable cockpit data files (run automatically at the end of the export).

    python lovable-export/validate.py ../cockpit/public/data [--no-causal]

Exit code 0 only if every check passes. Prints a summary table. The causal check re-runs B4 on inputs truncated at 20
randomly sampled snapshot times (train, validation and simulated pumps; never the test pumps) and requires the
recomputed snapshot to equal the exported one.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

BUDGET = {"scene": 1_000_000, "fleet.json": 2_500_000, "impact.json": 300_000}
LEVELS = {"WARNING", "CAUTION", "ADVISORY", None}
SPLITS = {"train", "validation", "test", "simulated"}
EVENT_CLASSES = {"MACHINE", "SENSOR"}
LABEL_KEYS = {"event_id", "onset_utc", "detectable_utc", "deadline_utc", "true_fault_type", "label_event_ids",
              "label_onset_utc"}
EVENT_ID_RE = re.compile(r"\b(?:PUMP|STRESS)-\d\d-E\d\d\b")
CAUSAL_SAMPLES = 20


def budget_for(name: str) -> int:
    if name.startswith("fleet"):
        return BUDGET["fleet.json"]
    if name.startswith("impact"):
        return BUDGET["impact.json"]
    return BUDGET["scene"]


class Checker:
    def __init__(self):
        self.errors = []

    def check(self, cond, msg):
        if not cond:
            self.errors.append(msg)
        return bool(cond)


def _reject_constant(x):
    raise ValueError(f"non-JSON constant {x}")


def load(path: Path, ck: Checker):
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except Exception as e:  # noqa: BLE001
        ck.check(False, f"{path.name}: does not parse ({e})")
        return None


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield path + "." + k, k, v
            yield from walk(v, path + "." + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")


def generic(name: str, doc, ck: Checker):
    ck.check(isinstance(doc.get("label"), str) and "synthetic" in doc["label"].lower(),
             f"{name}: label must say the data is synthetic")
    for p, k, v in walk(doc):
        if k in LABEL_KEYS:
            ck.check(False, f"{name}: label field {p}")
        if isinstance(v, float):
            ck.check(math.isfinite(v), f"{name}: non-finite number at {p}")
            ck.check(round(v, 3) == v, f"{name}: float not rounded to 3 decimals at {p}")
        if isinstance(v, str):
            ck.check(C.EM_DASH not in v, f"{name}: em dash at {p}")
            ck.check(not EVENT_ID_RE.search(v), f"{name}: event id in text at {p}")


def has(d, keys, where, ck):
    miss = [k for k in keys if not isinstance(d, dict) or k not in d]
    return ck.check(not miss, f"{where}: missing keys {miss}")


def check_bundle(name, b, ck):
    if not has(b, ["format", "label", "scene", "asset", "minutes", "channels", "alerts", "plant_alarms"], name, ck):
        return
    ck.check(b["format"] == "operator-demo-replay/1", f"{name}: format")
    has(b["scene"], ["id", "chapter", "kind", "title", "question", "start_utc", "end_utc", "focus_utc",
                     "minutes_per_tick", "main_channels", "lookback_start_utc"], f"{name}.scene", ck)
    has(b["asset"], ["asset_id", "site", "system", "control", "instrumentation_tier", "monitoring_start_utc",
                     "calibration_end_utc", "measurements"], f"{name}.asset", ck)
    m = b["minutes"]
    keys = ["t", "rows", "status", "reporting", "missing", "mode", "schedule", "demand_pct", "supply_temp_c"]
    if has(m, keys, f"{name}.minutes", ck):
        n = len(m["t"])
        for k in keys:
            ck.check(len(m[k]) == n, f"{name}: minutes.{k} has {len(m[k])} values, t has {n}")
        ck.check(all(s in C.STATUSES for s in m["status"]), f"{name}: unknown minutes.status")
        t = pd.to_datetime(pd.Series(m["t"]), utc=True)
        ck.check(t.is_monotonic_increasing, f"{name}: minutes.t not increasing")
        end = C.utc(b["scene"]["end_utc"])
        ck.check(bool((t + pd.Timedelta(minutes=1) <= end).all()), f"{name}: minutes after the scene end")
        for k, ch in b["channels"].items():
            ck.check(len(ch.get("measured", [])) == n and len(ch.get("expected", [])) == n,
                     f"{name}: channel {k} length differs from minutes.t")
        for a in b["alerts"]:
            has(a, ["asset_id", "alert_time_utc", "event_class", "fault_type", "evidence", "recommended_action",
                    "evidence_channels", "source_ref"], f"{name}.alert", ck)
            ck.check(a.get("event_class") in EVENT_CLASSES, f"{name}: alert event_class")
            ck.check(C.utc(a["alert_time_utc"]) <= end, f"{name}: alert after the scene end")
        for a in b["plant_alarms"]:
            has(a, ["tag", "alarm_type", "raised_utc", "cleared_utc"], f"{name}.plant_alarm", ck)
            ck.check(C.utc(a["raised_utc"]) <= end and (a["cleared_utc"] is None or C.utc(a["cleared_utc"]) <= end),
                     f"{name}: plant alarm information after the scene end")


def check_fleet(f, ck):
    if not has(f, ["format", "label", "aligned", "assets"], "fleet", ck):
        return
    ck.check(f["format"] == "fleet-replay/1", "fleet: format")
    ck.check(f["aligned"] == "monitoring_day", "fleet: aligned")
    ids = [a.get("asset_id") for a in f["assets"]]
    ck.check(len(ids) == len(set(ids)), "fleet: duplicate asset_id")
    for a in f["assets"]:
        w = f"fleet.{a.get('asset_id')}"
        if not has(a, ["asset_id", "site", "system", "control", "instrumentation_tier", "split", "snapshots"], w, ck):
            continue
        ck.check(a["split"] in SPLITS, f"{w}: split {a['split']}")
        if a["split"] == "simulated":
            ck.check("fictional" in a["site"].lower(), f"{w}: simulated site name must be clearly fictional")
        offs = [s["offset_min"] for s in a["snapshots"]]
        ck.check(all(isinstance(o, int) for o in offs), f"{w}: offset_min must be int")
        ck.check(all(x < y for x, y in zip(offs, offs[1:])), f"{w}: offset_min not increasing")
        for s in a["snapshots"]:
            if not has(s, ["offset_min", "deviation", "deviation_channel", "level", "status", "alert"], w, ck):
                break
            ck.check(s["status"] in C.STATUSES, f"{w}: status {s['status']}")
            ck.check(s["level"] in LEVELS, f"{w}: level {s['level']}")
            ck.check(s["deviation"] is None or isinstance(s["deviation"], (int, float)), f"{w}: deviation type")
            ck.check((s["deviation"] is None) == (s["deviation_channel"] is None), f"{w}: deviation/channel pairing")
            if s["status"] == "calibrating":
                ck.check(s["deviation"] is None, f"{w}: deviation while calibrating at {s['offset_min']}")
            if s["alert"] is not None:
                has(s["alert"], list(C.ALERT_KEYS), f"{w}.alert", ck)
                ck.check(s["level"] is not None, f"{w}: open alert without level at {s['offset_min']}")
                ck.check(s["alert"]["event_class"] in EVENT_CLASSES, f"{w}: alert event_class")
            elif s["level"] is not None:
                ck.check(s["level"] == "WARNING", f"{w}: level {s['level']} without an open alert")


def check_impact(d, ck):
    if not has(d, ["format", "label", "assumptions", "maintenance", "detector", "tier_ablation", "per_pump_exposure"],
               "impact", ck):
        return
    ck.check(d["format"] == "impact/1", "impact: format")
    has(d["maintenance"], ["period", "pumps", "fleet", "by_work_type", "strainer"], "impact.maintenance", ck)
    has(d["assumptions"], ["labour_rate_eur_h", "downtime_cost_eur_h", "unplanned_stop_h", "planned_stop_h",
                           "repair_unplanned_eur", "repair_planned_eur", "developing_faults_per_year",
                           "false_alert_check_eur", "operating_cost_eur_year", "installed_cost_eur",
                           "history_years_per_pump", "source"], "impact.assumptions", ck)
    for w in d["maintenance"]["by_work_type"]:
        has(w, ["work_type", "label", "records", "hours", "no_finding_records", "no_finding_hours"], "impact.work_type", ck)
    has(d["maintenance"]["strainer"], ["cleanings", "found_clean", "found_light", "found_moderate", "other"],
        "impact.strainer", ck)
    ck.check(isinstance(d.get("tiers"), list) and d["tiers"], "impact: tiers missing")
    for t in d.get("tiers", []):
        has(t, ["tier", "label", "hardware_eur", "sees", "composite"], "impact.tier", ck)
    ck.check(isinstance(d["detector"].get("unseen_note"), str), "impact.detector.unseen_note missing")
    has(d["maintenance"]["fleet"], ["records", "hours", "no_finding_records", "no_finding_hours",
                                    "no_finding_share_hours", "light_finding_records", "justified_records",
                                    "corrective_records"], "impact.maintenance.fleet", ck)
    for det in ("b1", "b4"):
        x = d["detector"].get(det, {})
        if has(x, ["name", "composite", "fault_type_recall", "event_recall", "false_alerts_per_pump_day"],
               f"impact.detector.{det}", ck):
            ck.check({"train", "validation", "unseen_simulated"} <= set(x["composite"]),
                     f"impact.detector.{det}.composite needs train/validation/unseen_simulated")


# ---------------------------------------------------------------------------------------------- causality
def _source(aid: str, data: Path, stress_root: Path):
    if aid.startswith("SIM-"):
        _, seed, nn = aid.split("-")
        pub = stress_root / f"stress_{seed}" / "public"
        return pub, f"STRESS-{nn}"
    return data, aid


def causal_check(fleet: dict, data: Path, stress_root: Path, ck: Checker, n: int = CAUSAL_SAMPLES,
                 alerts_file: Path | None = None) -> int:
    """B4 fleet: alerts come from the truncated B4 run. Other engines (alerts_file): their published alerts up to
    the cut, with telemetry, expected values and monitoring state from the truncated B4 run."""
    b4 = C.b4_module()
    params = C.frozen_params()
    pool = [(a["asset_id"], s) for a in fleet["assets"] if a["split"] != "test" for s in a["snapshots"]
            if s["offset_min"] >= 6 * 60]
    rng = np.random.default_rng(2026)
    picks = [pool[i] for i in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))]
    done = 0
    for aid, snap in picks:
        src, src_id = _source(aid, data, stress_root)
        assets = pd.read_csv(src / "assets.csv")
        a = assets[assets.asset_id == src_id].iloc[0].copy()
        alarms = pd.read_csv(src / "alarms.csv") if (src / "alarms.csv").exists() else pd.DataFrame(
            columns=["asset_id", "tag", "alarm_type", "raised_utc", "cleared_utc"])
        alarms = alarms[alarms.asset_id == src_id][["tag", "alarm_type", "raised_utc", "cleared_utc"]]
        cut = C.utc(a.window_start_utc) + pd.Timedelta(minutes=snap["offset_min"])
        inp = C.truncate_inputs(b4.load_inputs(str(src), [src_id]), cut)
        pred, f = C.run_b4(inp, params)
        if alerts_file is not None:
            al = pd.read_csv(alerts_file)
            al = al[al.asset_id == src_id]
            pred = al[pd.to_datetime(al.alert_time_utc, utc=True) <= cut]
        pred = pred.assign(asset_id=aid)
        a["asset_id"] = aid
        st = C.PumpState(a, f, pred, C.truncate_alarms(alarms, cut), params)
        again = C.clean(C.snapshot(st, snap["offset_min"]))
        ck.check(again == snap, f"causality: {aid} offset {snap['offset_min']} differs when recomputed from data "
                                f"<= {C.iso(cut)}: exported {snap} vs {again}")
        done += 1
    return done


# ---------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dir")
    p.add_argument("--data", default=str(C.DATA_DIR))
    p.add_argument("--stress-root", default=str(C.STRESS_ROOT))
    p.add_argument("--no-causal", action="store_true")
    a = p.parse_args(argv)
    d = Path(a.dir)
    ck = Checker()
    man = load(d / "manifest.json", ck) if ck.check((d / "manifest.json").exists(), "manifest.json missing") else None
    rows, docs = [], {}
    if man:
        has(man, ["format", "label", "generated_utc", "source_commit", "scenes", "fleet", "impact"], "manifest", ck)
        generic("manifest.json", man, ck)
        names = [s["file"] for s in man.get("scenes", [])] + [man.get("fleet"), man.get("impact")]
        fleets = {man.get("fleet")}
        for e in man.get("engines", []):
            if has(e, ["id", "label", "alerts_source", "scenes", "fleet", "impact"], "manifest.engine", ck):
                names += [s["file"] for s in e["scenes"]] + [e["fleet"], e["impact"]]
                fleets.add(e["fleet"])
                ck.check(len(e["scenes"]) == len(man["scenes"]), f"engine {e['id']}: scene count differs")
        names = list(dict.fromkeys(names))
        for s in man.get("scenes", []):
            has(s, ["chapter", "id", "file"], "manifest.scene", ck)
        for n in names:
            if not ck.check(n and (d / n).exists(), f"manifest lists {n}, which does not exist"):
                continue
            size = (d / n).stat().st_size
            budget = budget_for(n)
            ck.check(size < budget, f"{n}: {size} bytes over budget {budget}")
            doc = load(d / n, ck)
            if doc is None:
                continue
            docs[n] = doc
            generic(n, doc, ck)
            rows.append((n, size))
            if n in fleets:
                check_fleet(doc, ck)
            elif n == man.get("impact"):
                check_impact(doc, ck)
            else:
                check_bundle(n, doc, ck)
    fleet = docs.get(man.get("fleet")) if man else None
    causal = 0
    if fleet and not a.no_causal and not ck.errors:
        causal = causal_check(fleet, Path(a.data), Path(a.stress_root), ck)
        for e in man.get("engines", []):
            if e["fleet"] != man.get("fleet") and e["fleet"] in docs:
                causal += causal_check(docs[e["fleet"]], Path(a.data), Path(a.stress_root), ck, n=10,
                                       alerts_file=C.REPO / e["alerts_source"])

    print("\n| file | size KB | budget KB |\n|---|---:|---:|")
    for n, s in rows:
        print(f"| {n} | {s / 1024:.0f} | {budget_for(n) / 1024:.0f} |")
    for e in (man or {}).get("engines", []):
        fd = docs.get(e["fleet"])
        if fd:
            lv = pd.Series([s["level"] or "none" for x in fd["assets"] for s in x["snapshots"]]).value_counts()
            print(f"engine {e['id']} ({e['fleet']}, {len(fd['assets'])} pumps): "
                  + ", ".join(f"{k} {v}" for k, v in lv.items()))
    if fleet:
        sp = pd.Series([x["split"] for x in fleet["assets"]]).value_counts()
        snaps = [s for x in fleet["assets"] for s in x["snapshots"]]
        lv = pd.Series([s["level"] or "none" for s in snaps]).value_counts()
        last = pd.Series([x["snapshots"][-1]["level"] or "none" for x in fleet["assets"]]).value_counts()
        print(f"\npumps: {len(fleet['assets'])} ({', '.join(f'{k} {v}' for k, v in sp.items())}); "
              f"snapshots: {len(snaps)}")
        print("snapshots by level: " + ", ".join(f"{k} {v}" for k, v in lv.items()))
        print("pumps by level at end of week: " + ", ".join(f"{k} {v}" for k, v in last.items()))
    if man:
        print(f"scenes: {', '.join(s['id'] for s in man['scenes'])}; source commit {man.get('source_commit')}")
    imp = docs.get(man.get("impact")) if man else None
    if imp:
        mf = imp["maintenance"]["fleet"]
        print(f"maintenance: {mf['records']} records, {mf['no_finding_records']} with no finding, "
              f"{mf['no_finding_share_hours']:.1%} of hours, {mf['no_finding_hours']:.0f} h no-finding work")
        for det in [d for d in ("b1", "b4", "v2") if d in imp["detector"]]:
            c = imp["detector"][det]["composite"]
            print(f"{det.upper()} composite: train {c['train']}, validation {c['validation']}, "
                  f"unseen simulated {c['unseen_simulated']}")
    print(f"causality re-check: {causal} snapshots recomputed from truncated data")
    if ck.errors:
        print(f"\nVALIDATION FAILED ({len(ck.errors)} problems):")
        for e in ck.errors[:40]:
            print("  - " + e)
        return 1
    print("\nvalidation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
