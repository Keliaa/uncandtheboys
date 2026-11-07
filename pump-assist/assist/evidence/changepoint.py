"""D08: change-point detection. When a channel's CUSUM alarms, find the minute where the change started.

One-sided CUSUM of the channel's z-score (same z as D01, a residual against the commissioning baseline):
    S_t = max(0, S_{t-1} + sign * z_t - kappa)
The change started right after the last minute where S was zero before the alarm (the classical change-time estimate). Only data up
to the alert time is used. A pre-existing deviation (S never zero in the look-back window) gives no change time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import CH  # noqa: E402

from .sensor import STEP_MAX_MIN

KAPPA = 2.0
ALARM = 40.0            # CUSUM level counted as an alarm (B2 uses 60 with the same kappa; this is a descriptive estimate, not a detector)
LOOKBACK_H = 48
_INV = {v: k for k, v in CH.items()}


def change_time(zf: pd.DataFrame, asset_id: str, channel: str, when: pd.Timestamp, kappa: float = KAPPA, alarm: float = ALARM,
                lookback_h: float = LOOKBACK_H) -> pd.Timestamp | None:
    key = _INV.get(channel)
    if key is None or f"z_{key}" not in zf:
        return None
    g = zf[zf.asset_id == asset_id].sort_values("window_start_utc").set_index("window_start_utc")
    z = g[f"z_{key}"]
    z = z[(z.index <= when) & (z.index > when - pd.Timedelta(hours=lookback_h))].dropna()
    if len(z) < 10:
        return None
    sign = np.sign(z.iloc[-5:].median())
    if sign == 0:
        return None
    x = (sign * z.to_numpy()).clip(-30, 30)
    s = np.zeros(len(x))
    for i in range(len(x)):
        s[i] = max(0.0, (s[i - 1] if i else 0.0) + x[i] - kappa)
    if s[-1] < alarm:
        return None
    zero = np.flatnonzero(s == 0.0)
    if len(zero) == 0:
        return None                    # deviation already present at the start of the window
    return z.index[min(zero[-1] + 1, len(z) - 1)]


def add_change_times(alerts, zf: pd.DataFrame):
    """Append 'changed at <time>' to the evidence text of every channel with a detectable change."""
    for a in alerts:
        if not a.z_at_alert:
            continue
        when = pd.Timestamp(a.alert_time_utc)
        for e in a.evidence:
            if e.channel not in a.z_at_alert:
                continue
            t = change_time(zf, a.asset_id, e.channel, when)
            if t is None or "changed at" in e.text or "building since" in e.text:
                continue
            from .sensor import rise_minutes      # E03 shape: a jump is a step, a slow build-up is a drift or wear
            r = rise_minutes(zf, a.asset_id, e.channel, when)
            if r is not None and r > STEP_MAX_MIN:
                e.text += f" (deviation building since {t:%Y-%m-%d %H:%M} UTC)"
            else:
                e.text += f" (changed at {t:%Y-%m-%d %H:%M} UTC)"
    return alerts
