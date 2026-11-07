"""T02: new baseline after maintenance.

After a repair the machine is different, and faults also appear after maintenance (misalignment after a bearing
change, a sensor offset after a valve repair). For every lockout (LOCKED_OUT -> RUNNING):
  1. the end of the lockout is the restart time,
  2. a new baseline is learned from the first `learn_h` hours of steady running after it (the old baseline is the
     start value, so the step is the new level measured in units of the old baseline's spread),
  3. a step above `step_z` on any channel (vibration, pressure, ...) raises an alert, diagnosed with the same rules (E01).
The alert time is the end of the learning window, so only data up to then is used. Restarts in the first 24 h of a pump's
data are skipped (the old baseline is still being formed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import BASELINE_H, CH  # noqa: E402

LEARN_H = 2.0
STEP_Z = 8.0          # same as the D01 alert limit; chosen on train pumps (4 and 6 gave false alerts, 12 missed steps)
MIN_LOCK_MIN = 30
MIN_STEADY_MIN = 45   # steady minutes needed in the learning window
# Only channels whose level should not depend on the duty after a restart: vibration and pressures (the worklist step).
STEP_CHANNELS = {"vibration_rms_mm_s", "vibration_accel_rms_g", "vibration_crest_factor", "suction_pressure_kpa", "discharge_pressure_kpa"}


def lockout_restarts(zf: pd.DataFrame, min_lock: int = MIN_LOCK_MIN) -> pd.DataFrame:
    """asset_id, lock_start, restart (first minute after the lockout). Uses operating_mode only."""
    rows = []
    for aid, g in zf.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc")
        lk = (g["operating_mode"] == "LOCKED_OUT").to_numpy()
        t = g["window_start_utc"].to_numpy()
        i = 0
        while i < len(g):
            if lk[i]:
                j = i
                while j < len(g) and lk[j]:
                    j += 1
                if j - i >= min_lock and j < len(g):
                    rows.append({"asset_id": aid, "lock_start": pd.Timestamp(t[i]), "restart": pd.Timestamp(t[j])})
                i = j
            else:
                i += 1
    return pd.DataFrame(rows, columns=["asset_id", "lock_start", "restart"])


def restart_steps(zf: pd.DataFrame, restarts: pd.DataFrame, learn_h: float = LEARN_H) -> pd.DataFrame:
    """Per restart: window end (alert time) and the step of every channel (median z in the learning window)."""
    keys = [k for k in CH if f"z_{k}" in zf]
    rows = []
    for _, r in restarts.iterrows():
        g = zf[zf.asset_id == r.asset_id]
        t0 = g["window_start_utc"].min()
        if r.restart < t0 + pd.Timedelta(hours=BASELINE_H):
            continue
        w = g[(g.window_start_utc >= r.restart) & (g.window_start_utc < r.restart + pd.Timedelta(hours=learn_h, minutes=10)) & g.steady]
        if len(w) < MIN_STEADY_MIN:
            continue
        rows.append({"asset_id": r.asset_id, "restart": r.restart, "window_end": w.window_start_utc.max() + pd.Timedelta(minutes=1),
                     "steps": {CH[k]: float(w[f"z_{k}"].median()) for k in keys if CH[k] in STEP_CHANNELS and w[f"z_{k}"].notna().any()}})
    return pd.DataFrame(rows, columns=["asset_id", "restart", "window_end", "steps"])


def maintenance_alerts(steps: pd.DataFrame, assets_cfg: dict, machine: dict, step_z: float = STEP_Z):
    from ..alert import AlertRecord, Evidence
    from ..evidence.rules import classify
    alerts = []
    for _, r in steps.iterrows():
        z = r.steps
        if not z or max(abs(v) for v in z.values()) < step_z:
            continue
        d = classify(z, assets_cfg[r.asset_id], machine, 2 * step_z)      # classifier limit = 2 x step limit (same ratio as D01)
        top = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:3]
        d["evidence"] = [Evidence(c, v, f"After the maintenance stop the new baseline of {c.replace('_', ' ')} is {v:+.1f} spreads from the old one")
                         for c, v in top if abs(v) >= step_z] or d["evidence"]
        alerts.append(AlertRecord(asset_id=r.asset_id, alert_time_utc=r.window_end.strftime("%Y-%m-%dT%H:%M:%SZ"), source="t02", **d))
    return alerts
