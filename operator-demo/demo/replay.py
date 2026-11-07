"""Replay bundles and the replay clock.

Everything shown to the operator goes through `Scene.view(clock)`, which returns only information that was available
at `clock`:

* a one-minute aggregate for the minute [t, t + 1 min) becomes visible at t + 1 min (when the minute is complete);
* a B4 alert becomes visible at its recorded publication time `alert_time_utc`.

Bundles contain nothing after the scene end, so even a bug in a view cannot reveal later evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import channels as chn

BUNDLE_FORMAT = "operator-demo-replay/1"
MINUTE = pd.Timedelta(minutes=1)

STATUS_ORDER = ("unavailable", "calibrating", "not_running", "settling", "active")


def ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def fmt(t, seconds: bool = False) -> str:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "—"
    return ts(t).strftime("%Y-%m-%d %H:%M:%S UTC" if seconds else "%Y-%m-%d %H:%M UTC")


def fmt_hm(t) -> str:
    return ts(t).strftime("%H:%M UTC")


@dataclass
class MonitoringState:
    """Monitoring availability at the replay clock. Independent of tasks and operator actions."""
    status: str                    # one of STATUS_ORDER
    headline: str
    detail: str
    since: pd.Timestamp | None
    reporting: int = 0
    total: int = 0
    missing: list = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.status == "active"


@dataclass
class View:
    clock: pd.Timestamp
    minutes: pd.DataFrame          # completed minutes only
    alerts: pd.DataFrame           # alerts published at or before the clock
    monitoring: MonitoringState


class Scene:
    def __init__(self, data: dict, path: Path | None = None):
        if data.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"unsupported bundle format {data.get('format')!r}")
        self.raw = data
        self.path = path
        self.meta = data["scene"]
        self.asset = data["asset"]
        self.id = self.meta["id"]
        self.chapter = int(self.meta["chapter"])
        self.start = ts(self.meta["start_utc"])
        self.end = ts(self.meta["end_utc"])
        self.focus = ts(self.meta["focus_utc"]) if self.meta.get("focus_utc") else None
        self.speed = float(self.meta.get("minutes_per_tick", 1.0))
        self.calibration_end = ts(self.asset["calibration_end_utc"])
        self.monitoring_start = ts(self.asset["monitoring_start_utc"])
        m = data["minutes"]
        df = pd.DataFrame({k: v for k, v in m.items()})
        df["t"] = pd.to_datetime(df["t"], utc=True)
        df["available_at"] = df["t"] + MINUTE
        for key, ch in data["channels"].items():
            df[f"m_{key}"] = np.array(ch["measured"], dtype=float)
            df[f"e_{key}"] = np.array(ch["expected"], dtype=float)
        self.minutes = df.sort_values("t").reset_index(drop=True)
        self.channel_keys = list(data["channels"].keys())
        al = pd.DataFrame(data["alerts"])
        if len(al):
            al["time"] = pd.to_datetime(al["alert_time_utc"], utc=True)
            al = al.sort_values(["time", "source_ref"]).reset_index(drop=True)
        else:
            al = pd.DataFrame(columns=["time", "source_ref", "alert_time_utc"])
        self.alerts = al

    # ------------------------------------------------------------------ clock
    def clamp(self, clock) -> pd.Timestamp:
        c = ts(clock)
        return min(max(c, self.start), self.end)

    def advance(self, clock, minutes: float) -> tuple[pd.Timestamp, pd.DataFrame]:
        """Move the clock forward. Stops exactly at the first alert published in (clock, target] so the
        presenter sees each new action appear; returns the alerts that triggered the stop (empty if none)."""
        c = self.clamp(clock)
        target = self.clamp(c + pd.Timedelta(minutes=minutes))
        if len(self.alerts):
            new = self.alerts[(self.alerts.time > c) & (self.alerts.time <= target)]
            if len(new):
                first = new.time.min()
                return first, new[new.time == first]
        return target, self.alerts.iloc[0:0]

    def at_end(self, clock) -> bool:
        return ts(clock) >= self.end

    # ------------------------------------------------------------------ views
    def view(self, clock) -> View:
        c = ts(clock)
        mins = self.minutes[self.minutes.available_at <= c]
        alerts = self.alerts[self.alerts.time <= c] if len(self.alerts) else self.alerts
        return View(c, mins, alerts, monitoring_state(self, mins, c))

    def channel_label(self, key: str) -> str:
        u = chn.unit(key)
        return f"{chn.name(key)} ({u})" if u else chn.name(key)


def monitoring_state(scene: Scene, mins: pd.DataFrame, clock: pd.Timestamp) -> MonitoringState:
    total = int(scene.asset.get("measurements_total", 0))
    if not len(mins):
        return MonitoringState("unavailable", "Monitoring unavailable",
                               "No data received yet in this replay. An empty action list does not mean the pump "
                               "is fine.", None, 0, total)
    last = mins.iloc[-1]
    if last.available_at < clock - MINUTE:
        # the latest completed minute is missing entirely (data never arrived)
        return MonitoringState("unavailable", "Monitoring unavailable",
                               f"No data received since {fmt(last.available_at)}. An empty action list does not "
                               f"mean the pump is fine.", last.available_at, 0, total)
    status = str(last.status)
    since = _status_since(mins)
    missing = [k for k in str(last.get("missing", "") or "").split(",") if k]
    reporting = int(last.get("reporting", 0))
    if status == "unavailable":
        good = mins[mins.status != "unavailable"]
        lastdata = fmt(good.iloc[-1].available_at) if len(good) else "the start of this replay"
        return MonitoringState(status, "Monitoring unavailable",
                               f"No data received since {lastdata}. Condition checks cannot run without data; an "
                               f"empty action list does not mean the pump is fine.", since, 0, total, missing)
    if status == "calibrating":
        return MonitoringState(status, "Learning normal behaviour — checks limited",
                               f"Monitoring started {fmt(scene.monitoring_start)}. Until the 24-hour learning period "
                               f"ends at {fmt(scene.calibration_end)}, comparisons with expected values are "
                               f"incomplete. This is not a healthy state.", since, reporting, total, missing)
    if status == "not_running":
        mode = str(last.get("mode", "")).replace("_", " ").lower() or "not running"
        return MonitoringState(status, f"Pump not running ({mode})",
                               "Condition checks need the pump to run steadily; they resume after the next start.",
                               since, reporting, total, missing)
    if status == "settling":
        return MonitoringState(status, "Checks paused while the pump settles after a start",
                               f"Running since {fmt(since)}; comparisons resume after about 15 minutes of steady "
                               f"running.", since, reporting, total, missing)
    detail = "Readings are being compared with what is expected for the current operating conditions."
    if missing:
        names = ", ".join(chn.lower_name(k) for k in missing)
        detail += f" Not reporting this minute: {names} — checks that rely on them are limited."
    return MonitoringState("active", "Monitoring active", detail, since, reporting, total, missing)


def _status_since(mins: pd.DataFrame) -> pd.Timestamp | None:
    s = mins.status.values
    cur = s[-1]
    j = len(s) - 1
    while j > 0 and s[j - 1] == cur:
        j -= 1
    return mins.t.iloc[j]


# ---------------------------------------------------------------------------------------------------- loading
def load_scene(path: Path) -> Scene:
    with open(path, encoding="utf-8") as fh:
        return Scene(json.load(fh), Path(path))


def load_manifest(bundle_dir: Path) -> dict:
    p = Path(bundle_dir) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"No replay bundles found in {bundle_dir}. Run: python prepare_demo.py")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_scenes(bundle_dir: Path) -> list[Scene]:
    man = load_manifest(bundle_dir)
    scenes = [load_scene(Path(bundle_dir) / s["file"]) for s in man["scenes"]]
    return sorted(scenes, key=lambda s: s.chapter)
