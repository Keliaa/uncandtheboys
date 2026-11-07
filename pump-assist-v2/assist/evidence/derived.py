"""Derived-channel-aware sensor diagnosis (idea from B4, EXPERIMENTS #6; uses telemetry and the definition of the derived channel only).

The head residual of D04 is a derived channel: head = (discharge - suction pressure) / (rho g). If the suction transmitter drifts by
d kPa, the measured head moves by -d/(rho g) even though the pump is fine, so a "head below the pump curve" alert (IMPELLER_WEAR) can be a
sensor fault. Test it: the residual implied by each pressure transmitter's own shift (measured against its commissioning baseline) is
compared with the observed residual. If one transmitter explains it, and the machine channels (current, speed, power, flow) and the other
pressure are quiet, the alert is not independent evidence of a hydraulic fault: it becomes a SENSOR alert on that transmitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim import physics as P  # noqa: E402
from retrofit_sim.baselines import BASELINE_H  # noqa: E402

G = 9.81
WINDOW_MIN = 60
TOL = 0.35                  # |observed - implied| <= TOL * |observed| counts as explained
QUIET = {"motor_current_a": "i", "motor_speed_rpm": "n", "motor_power_kw": "pw", "flow_m3_h": "q"}


def channel_shift(zf: pd.DataFrame, asset_id: str, channel: str, when: pd.Timestamp, window_min: int = WINDOW_MIN) -> float | None:
    """Mean of (value - commissioning baseline of its speed regime) over the last `window_min` steady minutes, in native units."""
    m = f"{channel}__mean"
    g = zf[zf.asset_id == asset_id]
    if m not in g or g[m].notna().sum() == 0:
        return None
    t0 = g["window_start_utc"].min()
    base = g[(g["window_start_utc"] < t0 + pd.Timedelta(hours=BASELINE_H)) & g["steady"]]
    if base[m].notna().sum() < 30:
        return None
    med = base.groupby("regime")[m].median()
    mu = g["regime"].map(med).fillna(base[m].median())
    w = g[(g["window_start_utc"] <= when) & (g["window_start_utc"] > when - pd.Timedelta(minutes=window_min)) & g["steady"]]
    if len(w) < 10:
        return None
    d = (w[m] - mu.loc[w.index]).dropna()
    return float(d.mean()) if len(d) >= 10 else None


def _density(zf: pd.DataFrame, asset_id: str, when: pd.Timestamp, fluid: str) -> float:
    import re
    g = zf[(zf.asset_id == asset_id) & (zf.window_start_utc <= when)].tail(WINDOW_MIN)
    t = float(g["fluid_temp_c__mean"].mean()) if "fluid_temp_c__mean" in g and g["fluid_temp_c__mean"].notna().any() else 20.0
    m = re.search(r"([\d.]+)\s*PSU", str(fluid), flags=re.I)
    return float(P.water_density(t, float(m.group(1)) if m else 0.0))


def explain_by_transmitter(alerts, zf: pd.DataFrame, assets: pd.DataFrame, machine: dict, k: float = 8.0, tol: float = TOL,
                           dedupe_h: float = 72.0):
    """Convert head-residual (D04) alerts that one pressure transmitter explains into SENSOR alerts."""
    from ..alert import Evidence
    from ..config import action_for
    from .sensor import STEP_MAX_MIN, rise_minutes
    a_ix = assets.set_index("asset_id")
    for a in alerts:
        if a.source != "d04" or a.event_class != "MACHINE" or not a.evidence:
            continue
        when = pd.Timestamp(a.alert_time_utc)
        R = a.evidence[0].z                                   # observed head residual, % of rated head (negative = below the curve)
        H = float(a_ix.loc[a.asset_id].pump_rated_head_m)
        rho = _density(zf, a.asset_id, when, a_ix.loc[a.asset_id].fluid)
        g = zf[(zf.asset_id == a.asset_id) & (zf.window_start_utc <= when)].tail(1)
        if g.empty:
            continue
        from retrofit_sim.baselines import CH
        zlast = {CH[kk]: float(g[f"z_{kk}"].iloc[0]) for kk in CH if f"z_{kk}" in g and np.isfinite(g[f"z_{kk}"].iloc[0])}
        for ch, sign, other in (("suction_pressure_kpa", -1.0, "discharge_pressure_kpa"), ("discharge_pressure_kpa", +1.0, "suction_pressure_kpa")):
            d = channel_shift(zf, a.asset_id, ch, when)
            if d is None:
                continue
            implied = sign * d * 1000.0 / (rho * G) / H * 100.0
            same_sign = implied * R > 0
            if not (same_sign and abs(R - implied) <= tol * abs(R)):
                continue
            quiet = all(abs(zlast.get(c, 0.0)) < k / 3 for c in list(QUIET) + [other])
            if not quiet:
                continue
            r = rise_minutes(zf, a.asset_id, ch, when)
            ftype = "SENSOR_DRIFT" if (r is not None and r > STEP_MAX_MIN) else "SENSOR_CALIBRATION_SHIFT"
            a.event_class, a.fault_type, a.source = "SENSOR", ftype, "d04-explained"
            a.recommended_action = action_for(machine, ftype)["text"]
            a.what_if = None
            a.evidence = [Evidence(ch, float(d), f"The head residual ({R:+.1f} % of rated head) is explained by the {ch.replace('_', ' ')} transmitter alone "
                                                  f"(it reads {d:+.1f} kPa from its baseline, implying {implied:+.1f} %); current, speed, power, flow and the other pressure are normal")]
            break
    # One sensor problem is one alert: a converted alert is dropped when the same transmitter already raised a SENSOR alert on that pump
    # within `dedupe_h` hours (the same stretch of a drifting transmitter).
    out, seen = [], {}
    for a in sorted(alerts, key=lambda x: x.alert_time_utc):
        if a.source == "d04-explained":
            key = (a.asset_id, a.evidence[0].channel)
            t = pd.Timestamp(a.alert_time_utc)
            prior = [pd.Timestamp(b.alert_time_utc) for b in alerts if b is not a and b.asset_id == a.asset_id and b.event_class == "SENSOR" and b.evidence
                     and b.evidence[0].channel == a.evidence[0].channel and pd.Timestamp(b.alert_time_utc) < t
                     and t - pd.Timestamp(b.alert_time_utc) < pd.Timedelta(hours=dedupe_h)]
            if prior or (key in seen and t - seen[key] < pd.Timedelta(hours=dedupe_h)):
                continue
            seen[key] = t
        out.append(a)
    return out
