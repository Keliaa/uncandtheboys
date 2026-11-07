"""Headless walkthrough of the Streamlit app (streamlit.testing): every chapter, every control."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from demo.paths import INTERACTION_LOG  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def md(at) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


def btn(at, label):
    return [b for b in at.button if b.label == label or b.label.endswith(label)][0]


@pytest.fixture
def at(scenes):
    assert INTERACTION_LOG.name == "pytest_interactions.jsonl"
    a = AppTest.from_file(APP, default_timeout=60).run()
    btn(a, "Reset demo (clears all recorded actions)").click().run()
    assert not a.exception
    return a


def test_quiet_chapter(at):
    text = md(at)
    assert "Recorded replay · synthetic pump data" in text
    assert "No new checks recommended" in text and "not a statement that the pump is fault-free" in text
    assert "Monitoring active" in text
    assert not [b for b in at.button if b.label == "Acknowledge"]
    at.slider[0].set_value(at.slider[0].max).run()           # seek to the end of the load change
    assert not at.exception
    assert "No new checks recommended" in md(at)
    assert "healthy" not in md(at).lower().replace("not a healthy state", "")


def test_worth_checking_flow(at):
    btn(at, "Next scene").click().run()
    assert at.session_state.idx == 1
    assert "different pump" not in md(at) and "same pump" in md(at)
    at.slider[0].set_value(at.slider[0].max).run()           # seeking stops at the new check
    assert "08:23:00" in md(at) and "Check the suction strainer" in md(at)
    btn(at, "Acknowledge").click().run()
    assert "Acknowledged at 2025-09-10 08:23 UTC" in md(at)
    at.radio(key=[r.key for r in at.radio if r.key and r.key.startswith("out_")][0]).set_value(
        "Nothing unusual found").run()
    btn(at, "Save check").click().run()
    text = md(at)
    assert "Check recorded; monitoring continues." in text and "Check the suction strainer" in text
    assert "Monitoring active" in text
    btn(at, "Show evidence").click().run()
    assert not at.exception
    text = md(at)
    assert "Suspected fault type" in text and "Suction strainer clogging" in text
    assert len(at.get("plotly_chart")) >= 2


def test_verify_reading_and_existing_items(at):
    btn(at, "Next scene").click().run()
    btn(at, "Next scene").click().run()
    assert at.session_state.idx == 2
    assert "to a different pump" in md(at)
    assert "Existing items" in md(at)                         # earlier misalignment concern is not hidden
    at.slider[0].set_value(at.slider[0].max).run()
    text = md(at)
    assert "Verify the motor-current reading" in text and "measurement rather than the machine" in text


def test_restart_clears_chapter_actions(at):
    btn(at, "Next scene").click().run()
    at.slider[0].set_value(at.slider[0].max).run()
    btn(at, "Acknowledge").click().run()
    btn(at, "Restart").click().run()
    assert "No new checks recommended" in md(at)                # back before the alert
    at.slider[0].set_value(at.slider[0].max).run()
    assert [b for b in at.button if b.label == "Acknowledge"]   # task is new again after restart


def test_play_advances_and_pauses_at_new_check(at, monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    btn(at, "Next scene").click().run()
    s = at.session_state
    s.clocks["worth_checking"] = "2025-09-10T08:21:00+00:00"
    btn(at, "Play").click().run()
    assert not at.exception
    assert s.playing is False                                  # paused automatically
    assert s.clocks["worth_checking"].startswith("2025-09-10T08:23:00")
