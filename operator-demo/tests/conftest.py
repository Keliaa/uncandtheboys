from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# app tests must never touch the presenter's interaction log
os.environ.setdefault("OPERATOR_DEMO_LOG", "pytest_interactions.jsonl")

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))

from demo.paths import BUNDLE_DIR  # noqa: E402
from demo.replay import BUNDLE_FORMAT, Scene, load_scenes  # noqa: E402


@pytest.fixture(scope="session")
def scenes():
    if not (BUNDLE_DIR / "manifest.json").exists():
        pytest.skip("replay bundles not prepared (run: python prepare_demo.py)")
    return {s.id: s for s in load_scenes(BUNDLE_DIR)}


def make_scene(statuses, alerts=(), start="2025-01-01T00:00:00Z", cal_hours=24, monitoring_start=None,
               channel_value=1.0) -> Scene:
    """Small synthetic bundle: one minute per entry in `statuses` (None = no row for that minute at all)."""
    t0 = pd.Timestamp(start)
    ts_, st_ = [], []
    for j, s in enumerate(statuses):
        if s is None:
            continue
        ts_.append((t0 + pd.Timedelta(minutes=j)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        st_.append(s)
    n = len(ts_)
    ms = pd.Timestamp(monitoring_start) if monitoring_start else t0
    end = t0 + pd.Timedelta(minutes=len(statuses))
    return Scene({
        "format": BUNDLE_FORMAT,
        "scene": {"id": "synthetic", "chapter": 1, "kind": "test", "title": "t", "start_utc": t0.isoformat(),
                  "end_utc": end.isoformat(), "minutes_per_tick": 1.0, "main_channels": ["i"]},
        "asset": {"asset_id": "PUMP-X", "site": "Ship", "system": "Seawater", "monitoring_start_utc": ms.isoformat(),
                  "calibration_end_utc": (ms + pd.Timedelta(hours=cal_hours)).isoformat(), "measurements": ["i"],
                  "measurements_total": 1},
        "minutes": {"t": ts_, "rows": [60] * n, "status": st_, "reporting": [1] * n, "missing": [""] * n,
                    "mode": ["RUNNING"] * n, "schedule": [None] * n, "demand_pct": [50.0] * n,
                    "supply_temp_c": [10.0] * n},
        "channels": {"i": {"measured": [channel_value] * n, "expected": [channel_value] * n}},
        "alerts": list(alerts),
    })


def alert(time, fault="SUCTION_STRAINER_CLOGGING", asset="PUMP-X", evidence=None, action="Inspect the strainer."):
    ev = evidence or ("suction pressure 4.86 kPa below the value expected for this operating point (z=-4.4). "
                      "Context: demand 52 %, supply water 14.3 degC.")
    return {"asset_id": asset, "alert_time_utc": time, "event_class": "MACHINE", "fault_type": fault,
            "confidence": 0.7, "evidence": ev, "recommended_action": action, "detector": "test",
            "source_ref": f"B4 {asset} {time} {fault}", "source_file": "x.csv", "source_row": 2,
            "evidence_channels": ["ps"]}
