"""E03: sensor fault or machine fault.

Rule 1 (in evidence.rules.classify): one channel deviates and all its related channels (machine file `related_channels`)
stay quiet -> class SENSOR. How quiet is `sensor_alone` x the alert limit.
Rule 2 (Q01, ingest.quality): frozen value, noise and missing rows are SENSOR faults.
Rule 3 (this module): the shape of the deviation separates a drift from a calibration shift. A sensor that jumps within
`STEP_MAX_MIN` minutes is a shift; one that moves slowly over hours is a drift. A machine fault that leaves a related channel
unmoved can look like either, which is why rule 1 is only a rule-out, not proof.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import CH  # noqa: E402

STEP_MAX_MIN = 60
LOOKBACK_H = 48
HIGH_Z = 8.0
LOW_Z = 1.0
_INV = {v: k for k, v in CH.items()}


def rise_minutes(zf: pd.DataFrame, asset_id: str, channel: str, when: pd.Timestamp, lookback_h: float = LOOKBACK_H,
                 high: float = HIGH_Z, low: float = LOW_Z) -> float | None:
    """Minutes from the last time |z| was below `low` to the first time it exceeded `high` (5-min median, same sign as at `when`).
    None when the channel never was quiet in the look-back window (pre-existing deviation) or never reached `high`. Uses only data up to `when`."""
    key = _INV.get(channel)
    if key is None or f"z_{key}" not in zf:
        return None
    g = zf[zf.asset_id == asset_id].sort_values("window_start_utc").set_index("window_start_utc")
    s = g[f"z_{key}"].rolling(5, min_periods=1).median()
    w = s[(s.index <= when) & (s.index > when - pd.Timedelta(hours=lookback_h))].dropna()
    if w.empty:
        return None
    a = np.abs(w * np.sign(w.iloc[-1]))
    hi = a[a > high]
    if hi.empty:
        return None
    t_hi = hi.index.min()
    quiet = a[(a.index < t_hi) & (a < low)]
    return None if quiet.empty else (t_hi - quiet.index.max()).total_seconds() / 60.0


def refine_sensor_types(alerts, zf: pd.DataFrame, machine: dict, step_max_min: float = STEP_MAX_MIN):
    """A SENSOR_CALIBRATION_SHIFT that built up slowly (rise time above `step_max_min`) becomes SENSOR_DRIFT."""
    from ..alert import Evidence
    from ..config import action_for
    for a in alerts:
        if a.event_class != "SENSOR" or a.fault_type != "SENSOR_CALIBRATION_SHIFT" or not a.z_at_alert:
            continue
        ch, _ = max(a.z_at_alert.items(), key=lambda kv: abs(kv[1]))
        r = rise_minutes(zf, a.asset_id, ch, pd.Timestamp(a.alert_time_utc))
        if r is not None and r > step_max_min:
            a.fault_type = "SENSOR_DRIFT"
            a.recommended_action = action_for(machine, "SENSOR_DRIFT")["text"]
            a.evidence.append(Evidence(ch, a.z_at_alert[ch], f"{ch.replace('_', ' ')} moved slowly over {r / 60:.0f} h (drift, not a step)"))
    return alerts
