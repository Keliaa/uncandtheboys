"""Unit tests for the shared export rules and the maintenance classification.

Run from the repository root:  python -m pytest -q lovable-export/tests
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import export_lovable_data as X  # noqa: E402
import validate as V  # noqa: E402


# ------------------------------------------------------------------------------- sigma and deviation
def test_calibration_sigma_matches_cockpit_rule():
    off = np.arange(100, dtype=float)
    r = np.sin(off)
    resid = {"a": r, "b": np.full(100, np.nan)}
    sig = C.calibration_sigmas(resid, off, cal_min=40)
    assert sig["a"] == pytest.approx(np.std(r[:40], ddof=1))
    assert sig["b"] is None                                   # fewer than 10 calibration points


def test_sigma_floor():
    off = np.arange(50, dtype=float)
    sig = C.calibration_sigmas({"k": np.zeros(50)}, off, cal_min=30)
    assert sig["k"] == C.SIGMA_FLOOR


def test_deviation_takes_channel_max_and_respects_staleness():
    r1 = np.array([0.0, 1.0, 2.0, np.nan, np.nan])
    r2 = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
    z, k = C.deviation_at({"x": r1, "y": r2}, {"x": 1.0, "y": 0.1}, idx=4)
    assert (k, z) == ("y", pytest.approx(5.0))
    z, k = C.deviation_at({"x": r1, "y": r2}, {"x": 0.1, "y": 1.0}, idx=4)
    assert (k, z) == ("x", pytest.approx(20.0))                # last finite x residual (2 min old) is used
    stale = np.r_[1.0, np.full(40, np.nan)]
    assert C.deviation_at({"x": stale}, {"x": 1.0}, idx=40) == (None, None)   # older than 30 min


def test_deviation_fallback_uses_only_past_residuals():
    r = np.array([1.0, -1.0, 1.0, 100.0])
    z_past, _ = C.deviation_at({"x": r}, {"x": None}, idx=2)
    assert z_past == pytest.approx(1.0 / np.std(r[:3], ddof=1))


# ------------------------------------------------------------------------------- open-alert rule
def _alerts(rows):
    df = pd.DataFrame(rows, columns=["asset_id", "alert_time_utc", "event_class", "fault_type", "evidence",
                                     "recommended_action"])
    return C.alert_table(df, pd.Timestamp("2025-01-01", tz="UTC"))


def test_open_rule_hold_refresh_and_lockout():
    al = _alerts([["P", "2025-01-01T01:00:00Z", "MACHINE", "X", "e", "act A"],
                  ["P", "2025-01-01T05:00:00Z", "MACHINE", "X", "e2", "act A"],
                  ["P", "2025-01-01T02:00:00Z", "SENSOR", "S", "e", "act B"]])
    none = np.array([])
    assert len(C.open_alerts(al, none, 30)) == 0                          # before any alert
    assert len(C.open_alerts(al, none, 70)) == 1
    assert len(C.open_alerts(al, none, 130)) == 2
    # group A is refreshed by its 05:00 alert: open until 17:00; group B closes at 14:00
    o = C.open_alerts(al, none, 15 * 60)
    assert list(o.recommended_action) == ["act A"] and o.iloc[0].evidence == "e2"
    assert len(C.open_alerts(al, none, 17 * 60)) == 0
    # a LOCKED_OUT minute completed after the latest alert closes the group
    o = C.open_alerts(al, np.array([6 * 60 + 1.0]), 7 * 60)
    assert list(o.recommended_action) == []
    sev = C.most_severe(C.open_alerts(al, none, 130))
    assert sev["event_class"] == "MACHINE"


def test_open_intervals_match_snapshot_rule():
    class S:
        pass
    st = S()
    st.alerts = _alerts([["P", "2025-01-01T01:00:00Z", "MACHINE", "X", "e", "a"],
                         ["P", "2025-01-01T05:00:00Z", "MACHINE", "X", "e", "a"]])
    st.lock_off = np.array([])
    iv = C.open_intervals(st, end_min=7 * 24 * 60)
    assert iv == [(60.0, 17 * 60.0)]
    for t in range(0, 24 * 60, 7):
        inside = any(s <= t < e for s, e in iv)
        assert inside == (len(C.open_alerts(st.alerts, st.lock_off, t)) > 0)


def test_plant_alarm_active_and_truncation():
    al = pd.DataFrame({"tag": ["x"], "alarm_type": ["LO"], "raised_utc": ["2025-01-01T01:00:00Z"],
                       "cleared_utc": ["2025-01-01T02:00:00Z"]})
    t = lambda s: pd.Timestamp(s, tz="UTC")
    assert not C.alarm_active(al, t("2025-01-01T00:59"))
    assert C.alarm_active(al, t("2025-01-01T01:30"))
    assert not C.alarm_active(al, t("2025-01-01T02:00"))
    tr = C.truncate_alarms(al, t("2025-01-01T01:30"))
    assert tr.cleared_utc.isna().all()                         # clearing is not known yet at the cut


# ------------------------------------------------------------------------------- JSON hygiene
def test_clean_rounds_and_removes_nonfinite_and_em_dash():
    out = C.clean({"a": 1.23456, "b": float("nan"), "c": [np.float64(2.0004), np.inf], "d": "x — y",
                   "e": np.int64(3), "f": -0.0001})
    assert out == {"a": 1.235, "b": None, "c": [2.0, None], "d": "x - y", "e": 3, "f": 0.0}


# ------------------------------------------------------------------------------- maintenance
@pytest.mark.parametrize("wt,obs,cls", [
    ("ROUTINE_INSPECTION", "No abnormal findings.", "no_finding"),
    ("ROUTINE_INSPECTION", "Routine inspection, no abnormal findings.", "no_finding"),
    ("STRAINER_CLEANING", "Clean.", "no_finding"),
    ("STRAINER_CLEANING", "Light debris.", "light"),
    ("STRAINER_CLEANING", "Moderate fouling.", "justified"),
    ("STRAINER_CLEANING", "Low suction pressure noted; strainer cleaned.", "other"),
    ("VIBRATION_ROUTE_MEASUREMENT", "Portable reading 2.6 mm/s RMS, zone A/B.", "no_finding"),
    ("ALIGNMENT_CHECK", "Within tolerance.", "no_finding"),
    ("COUPLING_INSPECTION", "Coupling guard removed and coupling checked, OK.", "no_finding"),
    ("INSTRUMENT_CHECK", "Transmitters loop-checked, OK.", "no_finding"),
    ("LUBRICATION", "Bearings regreased.", "other"),
])
def test_classify_maintenance(wt, obs, cls):
    assert X.classify_maintenance(wt, obs) == cls


@pytest.mark.skipif(not (C.DATA_DIR / "maintenance.csv").exists(), reason="shared data not present")
def test_maintenance_known_totals():
    m = X.maintenance_summary(C.DATA_DIR)
    f = m["fleet"]
    assert f["records"] == 741
    assert f["no_finding_records"] == 458
    assert abs(f["no_finding_share_hours"] - 0.62) <= 0.01
    assert m["strainer"]["cleanings"] == 301 and m["strainer"]["found_clean"] == 109


# ------------------------------------------------------------------------------- exported files
EXPORT_DIR = Path(os.environ.get("LOVABLE_EXPORT_DIR", str(C.REPO.parent / "cockpit" / "public" / "data")))


@pytest.mark.skipif(not (EXPORT_DIR / "manifest.json").exists(), reason="no exported files to validate")
def test_exported_files_pass_contract():
    assert V.main([str(EXPORT_DIR), "--no-causal"]) == 0


def test_validator_rejects_em_dash_nan_and_labels(tmp_path):
    ck = V.Checker()
    V.generic("x.json", {"label": "synthetic", "a": "a — b", "event_id": "PUMP-01-E01", "v": 1.23456}, ck)
    msgs = " ".join(ck.errors)
    assert "em dash" in msgs and "label field" in msgs and "event id" in msgs and "rounded" in msgs
    (tmp_path / "bad.json").write_text('{"x": NaN}')
    ck = V.Checker()
    assert V.load(tmp_path / "bad.json", ck) is None and ck.errors


# ------------------------------------------------------------------------------- second engine (v2)
def test_channels_in_text_reads_v2_wording():
    avail = ["v", "a", "ps", "pd", "i", "pw"]
    assert X.channels_in_text("vibration rms mm s deviates alone (z = +18.1)", avail) == ["v"]
    assert X.channels_in_text("suction pressure is low (z = -8.4); motor power up", avail) == ["ps", "pw"]
    assert X.channels_in_text("nothing named here", avail) == []


def test_select_engine_alert_needs_class_inside_labelled_event():
    ev = pd.DataFrame({"asset_id": ["P"], "fault_type": ["SENSOR_DRIFT"], "onset_utc": ["2025-01-02T00:00:00Z"],
                       "end_utc": ["2025-01-03T00:00:00Z"]})
    pred = pd.DataFrame({"alert_time_utc": ["2025-01-01T12:00:00Z", "2025-01-02T05:00:00Z", "2025-01-02T06:00:00Z"],
                         "event_class": ["SENSOR", "MACHINE", "SENSOR"]})
    sc = {"asset": "P", "fault_type": "SENSOR_DRIFT", "title": "t"}
    assert X.select_engine_alert(sc, pred, ev) == pd.Timestamp("2025-01-02T06:00:00Z")
    with pytest.raises(X.ExportError):
        X.select_engine_alert(sc, pred.iloc[:2], ev)


@pytest.mark.skipif(not X.V2_TRAINVAL.exists(), reason="pump-assist-v2 results not present")
def test_v2_alert_files_have_submission_columns():
    for f in (X.V2_TRAINVAL, X.V2_TEST):
        cols = set(pd.read_csv(f).columns)
        assert {"asset_id", "alert_time_utc", "event_class", "fault_type", "evidence", "recommended_action"} <= cols
