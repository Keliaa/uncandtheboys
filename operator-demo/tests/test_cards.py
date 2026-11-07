"""Action cards: grouping, explanations supported by the source alert, and the operator workflow."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from demo.adapter import (MONITORING_CONTINUES, OUTCOMES, build_cards, check_for, explain, parse_evidence,
                          split_cards)
from demo.interactions import InteractionStore
from demo.paths import B4_FROZEN_PREDICTIONS

from conftest import alert, make_scene


def frame(*alerts):
    df = pd.DataFrame(list(alerts))
    df["time"] = pd.to_datetime(df.alert_time_utc, utc=True)
    return df


ASSET = {"asset_id": "PUMP-X", "site": "Ship", "system": "Seawater"}


# ------------------------------------------------------------------------------------------------ grouping
def test_repeated_alerts_same_check_grouped():
    df = frame(alert("2025-01-01T01:00:00Z"), alert("2025-01-01T03:00:00Z"), alert("2025-01-01T02:00:00Z"))
    cards = build_cards(df, ASSET)
    assert len(cards) == 1
    c = cards[0]
    assert [str(o.time) for o in c.occurrences] == sorted(str(o.time) for o in c.occurrences)
    assert c.first.time == pd.Timestamp("2025-01-01T01:00:00Z") and c.latest.time == pd.Timestamp("2025-01-01T03:00Z")
    assert len(c.occurrences) == 3


def test_different_check_is_separate_card():
    df = frame(alert("2025-01-01T01:00:00Z"),
               alert("2025-01-01T02:00:00Z", fault="BEARING_WEAR",
                     evidence="crest factor 5.1 (healthy ~3.2)", action="Check the bearing."))
    cards = build_cards(df, ASSET)
    assert len(cards) == 2
    assert {c.title for c in cards} == {"Check the suction strainer", "Check the pump bearing"}


def test_sensor_checks_grouped_per_channel():
    ev_i = ("motor current deviates alone: +3.97 A from the context-expected value (z=+4.5); the physically coupled "
            "channels do not show the change a real machine fault would cause. Pattern: gradual ramp.")
    ev_q = ev_i.replace("motor current", "flow").replace(" A ", " m3/h ")
    df = frame(alert("2025-01-01T01:00:00Z", "SENSOR_DRIFT", evidence=ev_i),
               alert("2025-01-01T02:00:00Z", "SENSOR_CALIBRATION_SHIFT", evidence=ev_i.replace("gradual ramp",
                                                                                               "step change")),
               alert("2025-01-01T03:00:00Z", "SENSOR_DRIFT", evidence=ev_q))
    cards = build_cards(df, ASSET)
    titles = sorted(c.title for c in cards)
    assert titles == ["Verify the flow reading", "Verify the motor-current reading"]
    assert sum(len(c.occurrences) for c in cards) == 3       # every underlying alert retained


def test_different_pumps_never_merge():
    df = frame(alert("2025-01-01T01:00:00Z"), alert("2025-01-01T02:00:00Z", asset="PUMP-Y"))
    assert len(build_cards(df, ASSET)) == 2


# --------------------------------------------------------------------------------------------- explanations
NUM = re.compile(r"\d+(?:\.\d+)?")


def _frozen_alerts():
    return pd.read_csv(B4_FROZEN_PREDICTIONS).fillna("")


def test_every_explanation_sentence_comes_from_its_evidence():
    """Each observation used in 'Why this check?' was read from an exact span of the B4 evidence text, and any
    number it shows is within rounding of a number in that span."""
    for _, a in _frozen_alerts().iterrows():
        ev = str(a.evidence)
        _, used = explain(a.to_dict())
        for o in used:
            assert o.span in ev, (a.alert_time_utc, o.span)
            src = [float(x) for x in NUM.findall(o.span)]
            for shown in NUM.findall(o.sentence):
                v = float(shown)
                if v in (3.2, 4.0) and "typical" in o.sentence:      # reference levels quoted from the evidence
                    continue
                assert any(abs(v - x) <= max(0.051, 0.006 * abs(x)) for x in src), (o.sentence, o.span)


def test_explanations_hide_scores_and_confidence():
    for _, a in _frozen_alerts().iterrows():
        why, _ = explain(a.to_dict())
        assert "z=" not in why and "confidence" not in why.lower() and "probability" not in why.lower()


def test_next_step_is_b4_action_verbatim():
    for _, a in _frozen_alerts().head(50).iterrows():
        df = frame({**a.to_dict(), "source_ref": "r", "alert_time_utc": a.alert_time_utc})
        c = build_cards(df, {"asset_id": a.asset_id})[0]
        assert c.next_step == a.recommended_action


def test_scene_explanations(scenes):
    s2 = scenes["worth_checking"]
    c2 = [c for c in build_cards(s2.view(s2.end).alerts, s2.asset) if c.title == "Check the suction strainer"][0]
    assert "Suction pressure is about 4.9 kPa lower than expected" in c2.why
    s3 = scenes["verify_reading"]
    c3 = [c for c in build_cards(s3.view(s3.end).alerts, s3.asset)
          if c.title == "Verify the motor-current reading"][0]
    assert "have not shown that change" in c3.why
    assert "measurement rather than the machine" in c3.why
    assert "do not act on the machine" in c3.next_step


def test_unparsed_evidence_still_explains():
    why, used = explain({"fault_type": "UNKNOWN", "evidence": "something new", "recommended_action": "x"})
    assert used == [] and "sustained difference" in why


# -------------------------------------------------------------------------------------------------- workflow
@pytest.fixture
def store(tmp_path):
    return InteractionStore(tmp_path / "log.jsonl")


def _scene_with(alerts):
    return make_scene(["active"] * 120, alerts=alerts)


def test_new_acknowledged_recorded(store):
    s = _scene_with([alert("2025-01-01T00:30:00Z")])
    t1 = pd.Timestamp("2025-01-01T00:40:00Z")
    cards = build_cards(s.view(t1).alerts, s.asset, store.visible_actions(s.id, t1), s.start)
    assert cards[0].status == "new"
    store.acknowledge(s.id, cards[0].card_id, t1, ["r"])
    t2 = t1 + pd.Timedelta(minutes=5)
    c = build_cards(s.view(t2).alerts, s.asset, store.visible_actions(s.id, t2), s.start)[0]
    assert c.status == "acknowledged" and "Monitoring continues" in c.status_note()
    store.record_check(s.id, c.card_id, t2, "Passed to maintenance", "  see log ", ["r"])
    t3 = t2 + pd.Timedelta(minutes=1)
    c = build_cards(s.view(t3).alerts, s.asset, store.visible_actions(s.id, t3), s.start)[0]
    assert c.status == "check_recorded"
    assert MONITORING_CONTINUES in c.status_note() and "no work order was created" in c.status_note()
    assert c.actions[-1]["notes"] == "see log"


def test_nothing_unusual_keeps_evidence_and_monitoring(store):
    s = _scene_with([alert("2025-01-01T00:30:00Z"), alert("2025-01-01T01:30:00Z")])
    t = pd.Timestamp("2025-01-01T00:45:00Z")
    cid = build_cards(s.view(t).alerts, s.asset)[0].card_id
    store.record_check(s.id, cid, t, "Nothing unusual found", "", ["r"])
    v = s.view(t + pd.Timedelta(minutes=1))
    mon_before = v.monitoring.status
    c = build_cards(v.alerts, s.asset, store.visible_actions(s.id, v.clock), s.start)[0]
    assert c.status == "check_recorded" and len(c.occurrences) == 1      # card and evidence still shown
    assert mon_before == "active"
    # the detector keeps raising it: the later alert is shown and re-opens the task
    later = s.view(pd.Timestamp("2025-01-01T01:31:00Z"))
    c2 = build_cards(later.alerts, s.asset, store.visible_actions(s.id, later.clock), s.start)[0]
    assert len(c2.occurrences) == 2 and c2.status == "new" and c2.updated_after_action
    assert later.monitoring.status == "active"


def test_seeking_back_hides_later_actions(store):
    s = _scene_with([alert("2025-01-01T00:30:00Z")])
    t = pd.Timestamp("2025-01-01T00:50:00Z")
    cid = build_cards(s.view(t).alerts, s.asset)[0].card_id
    store.acknowledge(s.id, cid, t, ["r"])
    assert store.visible_actions(s.id, t - pd.Timedelta(minutes=1)) == []
    assert len(store.visible_actions(s.id, t)) == 1


def test_restart_and_reset(store):
    store.acknowledge("a", "c1", "2025-01-01T00:00:00Z", [])
    store.acknowledge("b", "c2", "2025-01-01T00:00:00Z", [])
    store.restart_scene("a")
    assert store.visible_actions("a", "2030-01-01") == []
    assert len(store.visible_actions("b", "2030-01-01")) == 1
    store.reset()
    assert store.all() == []


def test_unknown_outcome_rejected(store):
    with pytest.raises(ValueError):
        store.record_check("a", "c", "2025-01-01", "Fixed it", "", [])
    assert set(OUTCOMES) == {"Concern observed", "Nothing unusual found", "Unable to verify", "Passed to maintenance"}


def test_store_refuses_paths_outside_demo():
    with pytest.raises(ValueError):
        InteractionStore("/tmp/elsewhere/log.jsonl")


def test_torn_log_line_ignored(tmp_path):
    p = tmp_path / "log.jsonl"
    st = InteractionStore(p)
    st.acknowledge("a", "c", "2025-01-01", [])
    with open(p, "a") as fh:
        fh.write('{"broken": ')
    assert len(st.all()) == 1


def test_split_keeps_everything():
    s = _scene_with([alert("2024-12-31T20:00:00Z"), alert("2025-01-01T00:30:00Z", fault="BEARING_WEAR",
                                                          evidence="crest factor 5.1 (healthy ~3.2)")])
    cards = build_cards(s.view(s.end).alerts, s.asset, [], s.start)
    cur, ex = split_cards(cards, s.start)
    assert [c.title for c in ex] == ["Check the suction strainer"]
    assert [c.title for c in cur] == ["Check the pump bearing"]
