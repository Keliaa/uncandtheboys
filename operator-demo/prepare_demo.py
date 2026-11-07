#!/usr/bin/env python3
"""Offline preparation of the operator-demo replay bundles.

    python prepare_demo.py            (from operator-demo/, or with any working directory)

* reads the shared development data (data/full, read only) and imports the B4 detector unchanged;
* runs B4 with its frozen parameters over each demo pump's whole monitoring window (from monitoring start, so the
  24-hour calibration is never restarted at a scene boundary) and checks the alerts are identical to the frozen
  benchmark output physics-context-detector/results/b4_trainval_predictions.csv;
* selects the quiet "normal change" window deterministically from the labelled benign load changes and validates the
  two alert scenes against the event labels (labels are used for selection/validation only and are written to
  bundles/prep_report.json, which the app never reads);
* writes compact replay bundles (completed one-minute observations, expected values, monitoring status, alerts up to
  the scene end) to operator-demo/bundles/.

Fails visibly (non-zero exit, nothing written) when a scene cannot be prepared honestly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from demo import paths  # noqa: E402
from demo.replay import BUNDLE_FORMAT  # noqa: E402

LOOKBACK_MIN = 24 * 60          # history included before each scene (charts / drawer), never after the scene end
QUIET_PAD_MIN = 20              # replay padding around the labelled load-change window
B4_COLS = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action",
           "detector"]

SCENES = [
    {"id": "normal_change", "chapter": 1, "kind": "quiet", "asset": "PUMP-08", "title": "Normal change",
     "question": "Does a normal operating change create a task?", "minutes_per_tick": 2.0,
     "main_channels": ["bt", "v", "pd", "pw"]},
    {"id": "worth_checking", "chapter": 2, "kind": "alert", "asset": "PUMP-08", "title": "Worth checking",
     "question": "Does a persistent concern produce an understandable check?",
     "alert_time": "2025-09-10T08:23:00Z", "fault_type": "SUCTION_STRAINER_CLOGGING", "lead_min": 45, "tail_min": 20,
     "minutes_per_tick": 1.5, "main_channels": ["ps", "pd", "pw", "n"]},
    {"id": "verify_reading", "chapter": 3, "kind": "alert", "asset": "PUMP-09", "title": "Verify the reading",
     "question": "Does an unreliable measurement lead to instrument verification?",
     "alert_time": "2026-02-11T00:09:00Z", "fault_type": "SENSOR_DRIFT", "lead_min": 45, "tail_min": 20,
     "minutes_per_tick": 1.5, "main_channels": ["i", "pw", "q", "pd"]},
]


class PrepError(RuntimeError):
    pass


def utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def iso(t) -> str:
    return utc(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def import_b4():
    sys.path.insert(0, str(paths.CHALLENGE_SRC))
    sys.path.insert(0, str(paths.B4_DIR))
    import b4  # noqa: E402  (unchanged detector module)
    return b4


# ---------------------------------------------------------------------------------------------------- B4 run
def run_detector(b4, data_dir: Path, asset_id: str, params: dict):
    inp = b4.load_inputs(str(data_dir), [asset_id])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        # numpy matmul floating-point warnings from the ridge fits; output is verified against the frozen CSV
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pred, frames = b4.run_b4(inp, params, return_frames=True)
    f, channels = frames[asset_id]
    return inp, pred, f, channels


def check_against_frozen(pred: pd.DataFrame, frozen: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    """Alerts must be identical to the frozen benchmark output; returns the frozen rows with their row numbers."""
    fz = frozen[frozen.asset_id == asset_id]
    a = pred[B4_COLS].reset_index(drop=True).astype(str)
    b = fz[B4_COLS].reset_index(drop=True).astype(str)
    if not a.equals(b):
        raise PrepError(f"{asset_id}: regenerated B4 alerts differ from the frozen predictions "
                        f"({len(a)} vs {len(b)} rows). Refusing to build bundles from a changed detector.")
    return fz


# ---------------------------------------------------------------------------------------------- monitoring
def monitoring_frame(b4, f: pd.DataFrame, a: pd.Series, params: dict) -> pd.DataFrame:
    """Per-minute monitoring status from the detector's own state flags (no labels)."""
    n = len(f)
    idx = np.arange(n)
    cal_end = int(params["cal_hours"] * 60)
    rows = f["rows"].values
    run = f["run"].values.astype(bool)
    steady = f["steady"].values.astype(bool)
    status = np.where(rows == 0, "unavailable",
                      np.where(idx < cal_end, "calibrating",
                               np.where(~run, "not_running", np.where(~steady, "settling", "active"))))
    avail, sparse = b4.static_channels(a)
    keys = [k for k in b4.CH if k in avail and b4.CH[k] + "__count" in f]
    rep = np.zeros(n, dtype=int)
    missing = [[] for _ in range(n)]
    for k in keys:
        cnt = f[b4.CH[k] + "__count"].fillna(0).values > 0
        if k in sparse:      # battery wireless sensors update about every 5 min
            cnt = pd.Series(cnt.astype(float)).rolling(7, min_periods=1).max().values > 0
        rep += cnt
        for j in np.where(~cnt)[0]:
            missing[j].append(k)
    out = pd.DataFrame({"status": status, "reporting": np.where(rows == 0, 0, rep),
                        "missing": [",".join(m) if rows[j] > 0 else ",".join(keys) for j, m in enumerate(missing)]})
    out.attrs["keys"] = keys
    return out


def display_series(b4, f: pd.DataFrame, k: str, sparse: set) -> tuple[np.ndarray, np.ndarray]:
    """Measured one-minute value and B4's context-expected value (expected shown only while running)."""
    col = b4.CH[k] + "__mean"
    raw = f[col].values.astype(float) if col in f else np.full(len(f), np.nan)
    meas = pd.Series(raw).ffill(limit=6).values if k in sparse else raw
    if "x_" + k in f and "e_" + k in f:
        exp = f["x_" + k].values - f["e_" + k].values
        exp = np.where(f["run"].values.astype(bool), exp, np.nan)
    else:
        exp = np.full(len(f), np.nan)
    return meas, exp


def _num(v, nd=3):
    out = []
    for x in v:
        if x is None or (isinstance(x, float) and not math.isfinite(x)) or (isinstance(x, (np.floating,)) and not np.isfinite(x)):
            out.append(None)
        else:
            out.append(round(float(x), nd))
    return out


# ------------------------------------------------------------------------------------------- scene selection
def select_quiet_window(benign: pd.DataFrame, asset_id: str, pred: pd.DataFrame, f: pd.DataFrame,
                        mon: pd.DataFrame, cal_end: pd.Timestamp) -> tuple[dict, list]:
    """Earliest labelled LOAD_CHANGE window after calibration, before the first B4 alert, with monitoring active
    throughout and no B4 alert overlapping the replay window."""
    times = pd.to_datetime(pred["alert_time_utc"], utc=True)
    first_alert = times.min() if len(times) else None
    w = benign[(benign.asset_id == asset_id) & (benign.benign_type == "LOAD_CHANGE")].copy()
    w["s"] = pd.to_datetime(w.start_utc, utc=True)
    w["e"] = pd.to_datetime(w.end_utc, utc=True)
    w = w.sort_values(["s", "e"])
    t = f["t"]
    log = []
    for _, r in w.iterrows():
        s0, e0 = r.s.floor("1min"), r.e.ceil("1min")
        start, end = s0 - pd.Timedelta(minutes=QUIET_PAD_MIN), e0 + pd.Timedelta(minutes=QUIET_PAD_MIN)
        reasons = []
        if r.s < cal_end:
            reasons.append("inside the initial 24-hour calibration period")
        if first_alert is not None and r.e >= first_alert:
            reasons.append("not before the pump's first B4 alert")
        if ((times >= start) & (times <= end)).any():
            reasons.append("a B4 alert overlaps the replay window")
        m = ((t >= start) & (t < end)).values
        st = mon.status.values[m]
        if m.sum() != int((end - start) / pd.Timedelta(minutes=1)):
            reasons.append("replay window not fully inside the monitoring window")
        elif (st != "active").any():
            bad = sorted(set(st[st != "active"]))
            reasons.append(f"monitoring not available throughout ({', '.join(bad)})")
        log.append({"start_utc": r.start_utc, "end_utc": r.end_utc, "note": r.note,
                    "qualifies": not reasons, "reasons": reasons})
        if not reasons:
            return {"label_start_utc": r.start_utc, "label_end_utc": r.end_utc, "note": r.note,
                    "start": start, "end": end}, log
    raise PrepError(f"No qualifying normal-change window for {asset_id}; not fabricating a quiet example.\n"
                    + "\n".join(f"  {x['start_utc']}..{x['end_utc']} ({x['note']}): {'; '.join(x['reasons'])}"
                                for x in log))


def validate_alert_scene(sc: dict, pred: pd.DataFrame, events: pd.DataFrame) -> dict:
    t = utc(sc["alert_time"])
    times = pd.to_datetime(pred["alert_time_utc"], utc=True)
    hit = pred[(times == t) & (pred.fault_type == sc["fault_type"])]
    if len(hit) != 1:
        raise PrepError(f"{sc['asset']}: expected exactly one {sc['fault_type']} B4 alert at {iso(t)}, found "
                        f"{len(hit)}. Alerts: {list(zip(pred.alert_time_utc, pred.fault_type))}")
    ev = events[(events.asset_id == sc["asset"]) & (events.fault_type == sc["fault_type"])].copy()
    on = pd.to_datetime(ev.onset_utc, utc=True)
    en = pd.to_datetime(ev.end_utc, utc=True)
    ok = ev[(on <= t) & (en.isna() | (en >= t))]
    if not len(ok):
        raise PrepError(f"{sc['asset']}: the {sc['fault_type']} alert at {iso(t)} is not inside a labelled event of "
                        f"that type; the scene would not show what it claims.")
    return {"alert_time_utc": iso(t), "fault_type": sc["fault_type"], "label_event_ids": list(ok.event_id),
            "label_onset_utc": list(ok.onset_utc)}


# ---------------------------------------------------------------------------------------------- bundle build
def evidence_channels(b4, alert: pd.Series, avail: list) -> list:
    from demo.adapter import check_for, parse_evidence
    ad = alert.to_dict()
    chk = check_for(ad)
    keys = []
    if chk.channel:
        keys = [chk.channel] + [c for c in b4.COUPLED.get(chk.channel, []) if c in b4.CH]
    else:
        for o in parse_evidence(ad["evidence"]):
            k = {"rx": "n", "hr": "pd"}.get(o.key, o.key)
            if k and k in b4.CH:
                keys.append(k)
        keys += {"SUCTION_STRAINER_CLOGGING": ["ps", "pd", "pw"], "DISCHARGE_RESTRICTION": ["pd", "q", "pw"],
                 "BEARING_WEAR": ["c", "a", "bt"], "MISALIGNMENT": ["v", "bt"], "IMBALANCE": ["v"],
                 "IMPELLER_WEAR": ["pd", "q", "pw"], "CAVITATION": ["ps", "pd", "a"],
                 "AIR_ENTRAINMENT": ["q", "pd", "pw"]}.get(ad["fault_type"], [])
    seen = []
    for k in keys:
        if k in avail and k not in seen:
            seen.append(k)
    return seen[:4]


def build_bundle(b4, sc: dict, a: pd.Series, f: pd.DataFrame, mon: pd.DataFrame, ctx: pd.DataFrame,
                 frozen_rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, focus, extra: dict) -> dict:
    ws = utc(a.window_start_utc)
    lb = max(start - pd.Timedelta(minutes=LOOKBACK_MIN), ws)
    t = f["t"]
    # the minute [t, t+1) is complete at t + 1 min: include only minutes complete by the scene end
    m = ((t >= lb) & (t + pd.Timedelta(minutes=1) <= end)).values
    g = f.loc[m].reset_index(drop=True)
    mg = mon.loc[m].reset_index(drop=True)
    c = ctx.copy()
    c["t"] = pd.to_datetime(c.timestamp_utc, utc=True)
    c = c.set_index("t").reindex(g["t"])
    avail, sparse = b4.static_channels(a)
    keys = mon.attrs["keys"]
    chans = {}
    for k in keys:
        meas, exp = display_series(b4, g, k, sparse)
        chans[k] = {"measured": _num(meas), "expected": _num(exp)}
    alerts = []
    times = pd.to_datetime(frozen_rows.alert_time_utc, utc=True)
    for row_no, r in frozen_rows[(times <= end).values].iterrows():
        d = {k: (None if (isinstance(r[k], float) and not math.isfinite(r[k])) else r[k]) for k in B4_COLS}
        d["confidence"] = float(d["confidence"])
        d["source_ref"] = f"B4 {r.asset_id} {r.alert_time_utc} {r.fault_type}"
        d["source_file"] = "physics-context-detector/results/b4_trainval_predictions.csv"
        d["source_row"] = int(row_no) + 2          # 1-based line number in the CSV (header is line 1)
        d["evidence_channels"] = evidence_channels(b4, r, keys)
        alerts.append(d)
    return {
        "format": BUNDLE_FORMAT,
        "label": "Recorded replay · synthetic pump data",
        "scene": {"id": sc["id"], "chapter": sc["chapter"], "kind": sc["kind"], "title": sc["title"],
                  "question": sc["question"], "start_utc": iso(start), "end_utc": iso(end),
                  "focus_utc": iso(focus) if focus is not None else None,
                  "minutes_per_tick": sc["minutes_per_tick"], "main_channels": [k for k in sc["main_channels"]
                                                                                 if k in keys],
                  "lookback_start_utc": iso(lb), **extra},
        "asset": {"asset_id": a.asset_id, "site": a.site, "system": a.system, "control": a.control,
                  "instrumentation_tier": a.instrumentation_tier, "monitoring_start_utc": iso(ws),
                  "calibration_end_utc": iso(ws + pd.Timedelta(hours=24)),
                  "measurements": keys, "measurements_total": len(keys)},
        "minutes": {"t": [iso(x) for x in g["t"]], "rows": [int(x) for x in g["rows"].values],
                    "status": list(mg.status), "reporting": [int(x) for x in mg.reporting],
                    "missing": list(mg.missing), "mode": list(g["operating_mode"].astype(str)),
                    "schedule": [None if (isinstance(x, float) or x is None) else str(x)
                                 for x in c["schedule_state"].values],
                    "demand_pct": _num(c["duty_demand_pct"].values, 1),
                    "supply_temp_c": _num(c["supply_water_temp_c"].values, 1)},
        "channels": chans,
        "alerts": alerts,
    }


# ----------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(paths.DATA_DIR), help="shared data directory (read only)")
    p.add_argument("--out", default=str(paths.BUNDLE_DIR), help="bundle directory (must be inside operator-demo)")
    args = p.parse_args(argv)
    out = Path(args.out).resolve()
    if not paths.inside_demo(out) or out == paths.DEMO_DIR.resolve():
        p.error("--out must be a directory inside operator-demo/")
    data = Path(args.data).resolve()
    t0 = time.time()
    b4 = import_b4()
    params = json.loads(paths.B4_FROZEN_PARAMS.read_text())
    frozen = pd.read_csv(paths.B4_FROZEN_PREDICTIONS)
    assets = pd.read_csv(data / "assets.csv")
    benign = pd.read_csv(data / "benign_windows_train_validation.csv")
    events = pd.concat([pd.read_csv(data / n) for n in ("events_train.csv", "events_validation.csv")
                        if (data / n).exists()], ignore_index=True)
    inputs_before = {str(x): sha256(x) for x in (paths.B4_FROZEN_PARAMS, paths.B4_FROZEN_PREDICTIONS)}

    report = {"generated_utc": iso(pd.Timestamp.now(tz="UTC")), "b4_params": str(paths.B4_FROZEN_PARAMS.name),
              "b4_module_sha256": sha256(paths.B4_DIR / "b4.py"), "frozen_inputs_sha256": inputs_before,
              "note": "Presenter/validation report. Contains label information; the operator app never reads it.",
              "scenes": []}
    cache = {}
    bundles = []
    for sc in SCENES:
        aid = sc["asset"]
        if aid not in cache:
            print(f"running B4 (frozen parameters) on {aid} from its monitoring start ...", flush=True)
            inp, pred, f, channels = run_detector(b4, data, aid, params)
            fz = check_against_frozen(pred, frozen, aid)
            a = assets[assets.asset_id == aid].iloc[0]
            mon = monitoring_frame(b4, f, a, {**b4.PARAMS, **params})
            cache[aid] = (inp, pred, f, a, mon, fz)
            print(f"  {len(pred)} alerts, identical to frozen predictions", flush=True)
        inp, pred, f, a, mon, fz = cache[aid]
        cal_end = utc(a.window_start_utc) + pd.Timedelta(hours=params["cal_hours"])
        if sc["kind"] == "quiet":
            win, log = select_quiet_window(benign, aid, pred, f, mon, cal_end)
            start, end, focus = win["start"], win["end"], None
            extra = {"context_note": f"Recorded operating change: {win['note']} "
                                     f"({utc(win['label_start_utc']).strftime('%H:%M')}–"
                                     f"{utc(win['label_end_utc']).strftime('%H:%M')} UTC)."}
            rep = {"id": sc["id"], "selection_rule": "earliest labelled LOAD_CHANGE window after the 24 h calibration, "
                   "before the first B4 alert, monitoring active throughout, no B4 alert in the replay window",
                   "selected": {k: v for k, v in win.items() if k not in ("start", "end")}, "candidates": log}
        else:
            rep = {"id": sc["id"], **validate_alert_scene(sc, pred, events)}
            focus = utc(sc["alert_time"])
            start = focus - pd.Timedelta(minutes=sc["lead_min"])
            end = focus + pd.Timedelta(minutes=sc["tail_min"])
            extra = {}
        bundle = build_bundle(b4, sc, a, f, mon, inp["ctx"], fz, start, end, focus, extra)
        rep.update({"start_utc": iso(start), "end_utc": iso(end), "alerts_in_bundle": len(bundle["alerts"]),
                    "minutes_in_bundle": len(bundle["minutes"]["t"])})
        report["scenes"].append(rep)
        bundles.append((f"scene_{sc['chapter']}_{sc['id']}.json", bundle))
        print(f"  chapter {sc['chapter']} '{sc['title']}': {iso(start)} .. {iso(end)}, "
              f"{len(bundle['alerts'])} alerts up to scene end", flush=True)

    inputs_after = {str(x): sha256(x) for x in (paths.B4_FROZEN_PARAMS, paths.B4_FROZEN_PREDICTIONS)}
    if inputs_after != inputs_before:
        raise PrepError("frozen benchmark files changed during preparation")
    # write atomically: build in a sibling temp directory inside operator-demo, then swap
    tmp = out.parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, b in bundles:
        (tmp / name).write_text(json.dumps(b, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    manifest = {"format": BUNDLE_FORMAT, "label": "Recorded replay · synthetic pump data",
                "generated_utc": report["generated_utc"],
                "scenes": [{"chapter": b["scene"]["chapter"], "id": b["scene"]["id"], "file": n,
                            "sha256": hashlib.sha256((tmp / n).read_bytes()).hexdigest()} for n, b in bundles]}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (tmp / "prep_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    print(f"bundles written to {out} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PrepError as e:
        print(f"\nPREPARATION FAILED: {e}", file=sys.stderr)
        sys.exit(2)
