"""Replay timing: the operator view never shows information from after the replay clock."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from demo import channels as chn
from demo.adapter import build_cards, split_cards
from demo.paths import B4_FROZEN_PREDICTIONS, BUNDLE_DIR
from demo.replay import MINUTE, ts

from conftest import alert, make_scene


def test_three_chapters_in_order(scenes):
    assert [s.chapter for s in sorted(scenes.values(), key=lambda s: s.chapter)] == [1, 2, 3]
    assert set(scenes) == {"normal_change", "worth_checking", "verify_reading"}


def test_bundles_contain_nothing_after_scene_end(scenes):
    for s in scenes.values():
        assert (s.minutes.available_at <= s.end).all(), s.id
        if len(s.alerts):
            assert (s.alerts.time <= s.end).all(), s.id


def test_bundles_carry_no_label_information(scenes):
    """Labels select/validate scenes only; they live in prep_report.json, which the app never reads."""
    for s in scenes.values():
        blob = json.dumps(s.raw).lower()
        for word in ("event_id", "onset", "label_", "events_train", "events_validation"):
            assert word not in blob, (s.id, word)


@pytest.mark.parametrize("sid", ["normal_change", "worth_checking", "verify_reading"])
def test_view_never_reveals_future(scenes, sid):
    s = scenes[sid]
    rng = np.random.default_rng(0)
    span = (s.end - s.start) / MINUTE
    clocks = [s.start, s.end] + [s.start + pd.Timedelta(minutes=float(x)) for x in rng.uniform(0, span, 40)]
    if s.focus is not None:
        clocks += [s.focus - pd.Timedelta(seconds=1), s.focus]
    for c in clocks:
        v = s.view(c)
        assert (v.minutes.available_at <= c).all()
        assert (v.minutes.t + MINUTE <= c).all()
        if len(v.alerts):
            assert (v.alerts.time <= c).all()


def test_seeking_back_and_forth_is_consistent(scenes):
    s = scenes["worth_checking"]
    late = s.view(s.end)
    early = s.view(s.focus - pd.Timedelta(minutes=10))
    again = s.view(s.end)
    assert len(early.alerts) < len(late.alerts)
    assert len(again.alerts) == len(late.alerts) and len(again.minutes) == len(late.minutes)


def test_quiet_scene_has_no_task_and_active_monitoring(scenes):
    s = scenes["normal_change"]
    assert len(s.alerts) == 0
    t = s.start
    while t <= s.end:
        v = s.view(t)
        assert build_cards(v.alerts, s.asset, [], s.start) == []
        assert v.monitoring.status == "active", (t, v.monitoring.headline)
        t += MINUTE
    # it is after calibration and the demand really changes during the window
    assert s.start >= s.calibration_end
    w = s.minutes[(s.minutes.t >= s.start) & (s.minutes.t < s.end)]
    assert w.demand_pct.max() - w.demand_pct.min() >= 10


@pytest.mark.parametrize("sid,when,fault", [("worth_checking", "2025-09-10T08:23:00Z", "SUCTION_STRAINER_CLOGGING"),
                                           ("verify_reading", "2026-02-11T00:09:00Z", "SENSOR_DRIFT")])
def test_alert_appears_exactly_at_recorded_time(scenes, sid, when, fault):
    s = scenes[sid]
    t = ts(when)
    before = s.view(t - pd.Timedelta(seconds=1))
    at = s.view(t)
    assert fault not in set(before.alerts.fault_type)
    assert fault in set(at.alerts.fault_type)
    cur, _ = split_cards(build_cards(at.alerts, s.asset, [], s.start), s.start)
    assert any(o.fault_type == fault and o.time == t for c in cur for o in c.occurrences)


def test_bundle_alerts_match_frozen_b4_rows(scenes):
    frozen = pd.read_csv(B4_FROZEN_PREDICTIONS)
    cols = ["asset_id", "alert_time_utc", "event_class", "fault_type", "evidence", "recommended_action", "detector"]
    for s in scenes.values():
        for _, a in s.alerts.iterrows():
            row = frozen.iloc[int(a.source_row) - 2]
            for c in cols:
                assert str(row[c]) == str(a[c]), (s.id, c)
            assert float(row.confidence) == pytest.approx(float(a.confidence))
        # every frozen alert of this pump up to the scene end is in the bundle (none dropped)
        fz = frozen[(frozen.asset_id == s.asset["asset_id"]) &
                    (pd.to_datetime(frozen.alert_time_utc, utc=True) <= s.end)]
        assert len(fz) == len(s.alerts), s.id


def test_advance_stops_at_first_new_alert(scenes):
    s = scenes["worth_checking"]
    c = s.focus - pd.Timedelta(minutes=1, seconds=30)
    new, trig = s.advance(c, 30)
    assert new == s.focus and len(trig) == 1
    new2, trig2 = s.advance(new, 1.5)          # continuing does not re-trigger the same alert
    assert len(trig2) == 0 and new2 > new


def test_advance_clamps_to_scene_end(scenes):
    s = scenes["normal_change"]
    new, trig = s.advance(s.end - MINUTE, 100)
    assert new == s.end and s.at_end(new) and len(trig) == 0


def test_existing_items_shown_separately(scenes):
    s = scenes["verify_reading"]
    v = s.view(s.end)
    cur, ex = split_cards(build_cards(v.alerts, s.asset, [], s.start), s.start)
    assert len(ex) >= 1 and all(c.first.time <= s.start for c in ex)
    assert any(c.title == "Verify the motor-current reading" for c in cur)
    assert sum(len(c.occurrences) for c in cur + ex) == len(v.alerts)


# -------------------------------------------------------------------------------- monitoring availability
def test_no_data_is_unavailable_not_healthy():
    s = make_scene(["active"] * 5)
    v = s.view(s.start)          # nothing completed yet
    assert v.monitoring.status == "unavailable" and not v.monitoring.available
    assert "does not mean the pump is fine" in v.monitoring.detail


def test_data_gap_is_unavailable():
    s = make_scene(["active"] * 5 + [None] * 10)
    v = s.view(s.start + pd.Timedelta(minutes=12))
    assert v.monitoring.status == "unavailable"
    assert build_cards(v.alerts, s.asset) == []      # empty queue, but monitoring explicitly unavailable


def test_unavailable_status_rows():
    s = make_scene(["active"] * 3 + ["unavailable"] * 3)
    assert s.view(s.end).monitoring.status == "unavailable"


def test_calibration_is_not_healthy():
    s = make_scene(["calibrating"] * 10)
    m = s.view(s.end).monitoring
    assert m.status == "calibrating" and not m.available
    assert "not a healthy state" in m.detail


def test_not_running_and_settling_are_not_active():
    assert make_scene(["not_running"] * 3).view(make_scene(["not_running"] * 3).end).monitoring.status == "not_running"
    s = make_scene(["settling"] * 3)
    assert s.view(s.end).monitoring.status == "settling" and not s.view(s.end).monitoring.available


def test_missing_measurement_is_reported():
    s = make_scene(["active"] * 3)
    s.minutes.loc[:, "missing"] = "i"
    m = s.view(s.end).monitoring
    assert m.status == "active" and m.missing == ["i"] and "motor current" in m.detail


# ------------------------------------------------------------------------------------- B4 name coupling
def test_channel_names_match_b4():
    import sys
    from demo.paths import B4_DIR, CHALLENGE_SRC
    sys.path.insert(0, str(CHALLENGE_SRC))
    sys.path.insert(0, str(B4_DIR))
    import b4
    assert chn.B4_NAME == b4.NAME
    assert chn.B4_UNIT == b4.UNIT
    assert chn.COLUMNS == {k: v for k, v in b4.CH.items() if k in chn.COLUMNS}


def test_manifest_hashes_match(scenes):
    import hashlib
    man = json.loads((BUNDLE_DIR / "manifest.json").read_text())
    for e in man["scenes"]:
        assert hashlib.sha256((BUNDLE_DIR / e["file"]).read_bytes()).hexdigest() == e["sha256"]
