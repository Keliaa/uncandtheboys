"""D04: physics residual from the pump curve. No training data is needed to compute it.

Measured head H = (p_discharge - p_suction) / (rho g). If flow is not measured, flow Q is inferred from the
pump's system curve H = H_static + K Q^2 (K from the design point). Then
    expected head  He = pump_head(Q, r)                (reference physics.py, affinity scaling built in)
    residual       = (H - He) / H_bep * 100            [% of rated head]
Impeller wear lowers head at the same flow, so a persistent negative residual points to wear.
Only the alert limit and persistence are chosen on train pumps (scripts/tune_d04.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim import physics as P  # noqa: E402

from .benign import benign_lookup

G = 9.81


def head_residual(features: pd.DataFrame, assets: pd.DataFrame, machine: dict) -> pd.DataFrame:
    """Per pump-minute residual (% of rated head) on steady running minutes; NaN elsewhere."""
    ph = machine.get("physics", {})
    a_ix = assets.set_index("asset_id")
    out = []
    for aid, g in features.groupby("asset_id", sort=False):
        A = a_ix.loc[aid]
        g = g.sort_values("window_start_utc").reset_index(drop=True)
        steady = (g["operating_mode"] == "RUNNING") & (g["running_fraction"] >= 0.99)
        run = steady.astype(int).groupby((~steady).cumsum()).cumsum()
        ok = run >= ph.get("steady_min_minutes", 10)
        col = lambda c: g[c] if c in g else pd.Series(np.nan, index=g.index)
        sal = ph.get("brackish_salinity_psu", 6.0) if "brackish" in str(A.fluid).lower() else 0.0
        rho = P.water_density(col("fluid_temp_c__mean").fillna(20.0), sal)
        n = col("motor_speed_rpm__mean")
        r = n / A.motor_rated_rpm if n.notna().any() else pd.Series(1.0, index=g.index)
        ps = col("suction_pressure_kpa__mean")
        if not ps.notna().any():
            ps = pd.Series(ph.get("assumed_suction_kpa_by_profile", {}).get(A.profile, 0.0), index=g.index)
        H = (col("discharge_pressure_kpa__mean") - ps) * 1000.0 / (rho * G)
        Hb, Qb = A.pump_rated_head_m, A.pump_rated_flow_m3_h / 3600.0
        h0 = A.pump_shutoff_head_m / A.pump_rated_head_m
        if col("flow_m3_h__mean").notna().any():
            Q = col("flow_m3_h__mean") / 3600.0
        else:
            K = (A.design_head_m - A.static_head_m) / (A.design_flow_m3_h / 3600.0) ** 2
            Q = np.sqrt(np.maximum(H - A.static_head_m, 0.0) / K)
        He = P.pump_head(Q, r, Hb, Qb, h0)
        res = ((H - He) / Hb * 100.0).where(ok)
        out.append(pd.DataFrame({"asset_id": aid, "window_start_utc": g["window_start_utc"], "steady": ok,
                                 "residual_pct": res}))
    return pd.concat(out, ignore_index=True)


def detect_physics(features: pd.DataFrame, assets: pd.DataFrame, machine: dict,
                   limit_pct: float = 5.0, persist: int = 60, benign: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flag when the EWMA of the residual stays below -limit_pct for `persist` steady minutes."""
    span = machine.get("physics", {}).get("ewma_span_min", 30)
    res = head_residual(features, assets, machine)
    parts = []
    for aid, g in res.groupby("asset_id", sort=False):
        g = g.reset_index(drop=True)
        s = g.loc[g.steady, "residual_pct"]
        ew = s.ewm(span=span, adjust=False).mean()
        flag = (ew < -limit_pct) & ~pd.Series(benign_lookup(benign, aid, g.loc[ew.index, 'window_start_utc']), index=ew.index)
        run = flag.astype(int).groupby((~flag).cumsum()).cumsum()
        g["ewma_pct"] = np.nan
        g["run"] = 0
        g.loc[ew.index, "ewma_pct"] = ew
        g.loc[ew.index, "run"] = run
        g["flag"] = g["run"] > 0
        g["sustained"] = g["run"] >= persist
        g["available_utc"] = g["window_start_utc"] + pd.Timedelta(minutes=1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def physics_alerts(det: pd.DataFrame, machine: dict, refractory: pd.Timedelta = pd.Timedelta(hours=12)):
    """AlertRecords for sustained negative head residual: IMPELLER_WEAR (head below the pump curve)."""
    from ..alert import AlertRecord, Evidence
    from ..config import action_for
    from ..evidence.rules import certainty_label
    alerts = []
    for aid, g in det.groupby("asset_id", sort=False):
        last = None
        starts = g[g.sustained & ~g.sustained.shift(fill_value=False)]
        for _, r in starts.iterrows():
            if last is not None and r.available_utc - last < refractory:
                continue
            last = r.available_utc
            p = float(min(0.9, max(0.1, abs(r.ewma_pct) / 20.0)))
            alerts.append(AlertRecord(
                asset_id=aid, alert_time_utc=r.available_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), event_class="MACHINE",
                fault_type="IMPELLER_WEAR", probability=p, certainty=certainty_label(p, machine),
                evidence=[Evidence("discharge_pressure_kpa", float(r.ewma_pct),
                                   f"Head is {abs(r.ewma_pct):.1f} % of rated head below the pump curve")],
                recommended_action=action_for(machine, "IMPELLER_WEAR")["text"]))
    return alerts
