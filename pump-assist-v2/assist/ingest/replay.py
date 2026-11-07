"""I02: live replay stream. One JSON message per second -> 1-minute rows in the same format as features_1min.parquet.

The replay tool (original-challenge retrofit_sim.replay) reads stored telemetry and emits {"topic", "payload"} messages; it only reads.
`MinuteBuffer` collects the messages of one minute and gives mean, std (n-1), min, max and count per channel, the most frequent
operating mode, the running fraction, the OR of the quality flags and the number of rows. A minute is complete (and handed to the
detector) only after the first message of the next minute arrives, so the detector never sees a half-filled minute.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from ..splits import DATA_FULL

CHANNELS = ["motor_speed_rpm", "motor_current_a", "motor_power_kw", "suction_pressure_kpa", "discharge_pressure_kpa", "flow_m3_h",
            "fluid_temp_c", "bearing_temp_c", "motor_winding_temp_c", "vibration_rms_mm_s", "vibration_accel_rms_g",
            "vibration_crest_factor", "ambient_temp_c", "valve_position_pct"]


def replay_messages(asset_id: str, data: Path | str = DATA_FULL, start: str | None = None, end: str | None = None) -> Iterator[dict]:
    """The messages the replay tool would print (read only)."""
    from retrofit_sim.replay import messages
    yield from messages(str(data), asset_id, start, end)


class MinuteBuffer:
    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self._minute: pd.Timestamp | None = None
        self._vals: dict[str, list[float]] = {}
        self._modes: Counter = Counter()
        self._flags = 0
        self._n = 0
        self._seen: set[str] = set()      # channels that have carried data so far (a column exists from then on, as in the file)

    def _flush(self) -> dict | None:
        if self._minute is None:
            return None
        row: dict = {"asset_id": self.asset_id, "window_start_utc": self._minute}
        for c in CHANNELS:
            if c not in self._seen:
                continue
            v = np.asarray(self._vals.get(c, []), dtype=float)
            n = len(v)
            row[f"{c}__mean"] = float(v.mean()) if n else np.nan
            row[f"{c}__std"] = float(v.std(ddof=1)) if n > 1 else np.nan
            row[f"{c}__min"] = float(v.min()) if n else np.nan
            row[f"{c}__max"] = float(v.max()) if n else np.nan
            row[f"{c}__count"] = n
        row["operating_mode"] = self._modes.most_common(1)[0][0]
        row["running_fraction"] = self._modes.get("RUNNING", 0) / self._n
        row["quality_any"] = self._flags
        row["rows"] = self._n
        self._vals, self._modes, self._flags, self._n = {}, Counter(), 0, 0
        return row

    def add(self, message: dict) -> dict | None:
        """Add one message. Returns the completed row of the previous minute when this message starts a new minute."""
        p = message["payload"]
        ts = pd.Timestamp(p["timestamp_utc"]).tz_convert("UTC")
        minute = ts.floor("min")
        done = None
        if self._minute is not None and minute != self._minute:
            done = self._flush()
        self._minute = minute
        for c in CHANNELS:
            if c in p:
                self._vals.setdefault(c, []).append(float(p[c]))
                self._seen.add(c)
        self._modes[p.get("operating_mode", "STOPPED")] += 1
        self._flags |= int(p.get("quality_flags", 0))
        self._n += 1
        return done

    def flush(self) -> dict | None:
        """End of stream: the last (possibly incomplete) minute."""
        return self._flush()


def minute_rows(messages: Iterable[dict], asset_id: str, speed: float | None = None) -> Iterator[dict]:
    """Messages -> completed 1-minute rows. speed = simulated seconds per wall-clock second (None: as fast as possible)."""
    buf = MinuteBuffer(asset_id)
    for m in messages:
        row = buf.add(m)
        if row is not None:
            if speed:
                time.sleep(60.0 / speed)
            yield row
    last = buf.flush()
    if last is not None:
        yield last
