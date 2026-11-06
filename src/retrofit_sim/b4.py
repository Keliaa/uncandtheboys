"""B4: physics- and context-aware causal detector with stable, controller-aware diagnosis.

Pipeline per asset (every step is causal: a value at minute t uses only rows with timestamp <= t plus assets.csv):

1. 1-min grid (features_1min + operating_context); running / steady / post-lockout flags.
2. Expected values for every channel:
   * physics (pump curve + system curve from assets.csv, documented control laws) where it is reliable:
     required speed of VFD temperature-controlled pumps, pump-curve head ratio where flow is metered;
   * a small ridge regression on exogenous operating context (duty demand, supply-water temperature, physics speed,
     ambient) fitted on the asset's own steady minutes, expanding during the first `cal_hours` and then frozen.
     Physics biases are learned the same way but clipped to a calibration envelope, so faults that are present at
     window start are not absorbed into "normal".
3. Robust z per channel; fast (5 min) and slow (60 min) smoothed scores.
4. Detectors with their own episode tracking: gateway loss, per-channel instrument faults (stuck / noise / single-
   channel drift or calibration step), bearing (acceleration + crest factor, fleet-level crest prior), velocity
   family (imbalance / misalignment), hydraulic family (clogging / restriction / impeller wear / cavitation / air).
5. Diagnosis from the whole persistence window (mean residual vector, which channel deviated first, step vs ramp),
   per control strategy. An alert is raised only when the diagnosed type has been stable for `stable_min` minutes
   (or `max_wait_min` has passed), one fault type per alert, alerts on one asset >= 6 min apart.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

G = 9.80665
CH = {"v": "vibration_rms_mm_s", "a": "vibration_accel_rms_g", "c": "vibration_crest_factor", "bt": "bearing_temp_c",
      "ps": "suction_pressure_kpa", "pd": "discharge_pressure_kpa", "i": "motor_current_a", "pw": "motor_power_kw",
      "q": "flow_m3_h", "n": "motor_speed_rpm", "wt": "motor_winding_temp_c", "tf": "fluid_temp_c", "vp": "valve_position_pct"}
NAME = {"v": "vibration velocity", "a": "vibration acceleration", "c": "vibration crest factor", "bt": "bearing temperature",
        "ps": "suction pressure", "pd": "discharge pressure", "i": "motor current", "pw": "motor power", "q": "flow",
        "n": "motor speed", "wt": "winding temperature", "tf": "fluid temperature", "vp": "valve position",
        "rx": "speed vs physics-required speed", "hr": "head vs pump curve"}
HYD = ["rx", "n", "i", "pw", "ps", "pd", "q", "vp", "hr"]
VIB = ["v", "a", "c"]
TEMP = ["bt", "wt"]
# channels that physically co-move with each channel (a deviation of one channel alone points at the instrument)
COUPLED = {"pd": ["i", "pw", "q", "n", "ps", "rx", "vp"], "ps": ["pd", "q", "n", "i", "pw", "rx", "vp"],
           "q": ["pd", "i", "pw", "n", "vp", "ps", "rx"], "i": ["pw", "pd", "q", "n", "rx", "wt"],
           "pw": ["i", "pd", "q", "n", "rx"], "bt": ["v", "a", "c", "wt"], "wt": ["i", "pw"], "tf": ["bt"],
           "n": ["pw", "i", "pd", "q"], "vp": ["q", "pd", "i", "pw"]}
SENSOR_CH = ["pd", "ps", "q", "i", "pw", "bt", "wt", "tf", "n", "vp"]
ABS_FLOOR = {"v": 0.05, "a": 0.01, "c": 0.06, "bt": 0.4, "ps": 1.0, "pd": 2.0, "i": 0.3, "pw": 0.3, "q": 2.0,
             "n": 4.0, "wt": 0.6, "tf": 0.3, "vp": 0.5}
REL_FLOOR = {"v": 0.03, "a": 0.03, "c": 0.015, "bt": 0.0, "ps": 0.0, "pd": 0.006, "i": 0.006, "pw": 0.008, "q": 0.008,
             "n": 0.002, "wt": 0.0, "tf": 0.0, "vp": 0.01}

PARAMS = {
    "cal_hours": 24.0,          # expanding calibration period of the context model, then frozen
    "min_cal_min": 90,          # minutes of steady data before residual detectors may fire
    "refit_every_min": 60,
    "steady_min": 15,           # minutes after a start before a minute counts as steady
    "k_fast": 6.0, "persist_fast": 10,
    "k_slow": 3.5, "persist_slow": 30, "slow_span": 60,
    "stable_min": 10,           # diagnosed type must be stable this long before the first alert
    "max_wait_min": 60,
    "close_after_min": 90,      # an episode ends after this many quiet minutes
    "reminder_h": 0.0,          # 0 = no reminders
    "gap_min": 3,
    "stuck_min": 20, "stuck_sparse_samples": 9, "stuck_nat_mult": 1.5,
    "noise_ratio": 3.0, "noise_persist": 8,
    "crest_abs": 3.75,          # fleet prior: healthy crest factor ~3.2 (bearing damage / cavitation raise it)
    "rx_env": (0.0, 0.025),     # plausible calibration bias of measured / physics speed ratio - 1
    "hr_env": (-0.03, 0.03),
    "single_ratio": 0.25,
    "k_strong": 18.0,           # very strong deviations count after 3 min
    "cav_ps_z": 6.0,
    "sensor_margin": 4.0,       # a single-instrument explanation must beat the best machine signature by this chi2 margin
    "unexplained": 0.6,
    "air_hs": 2.0, "air_lvl": 1.6,
    "post_maint_min": 720,
    "step_rise_min": 20,
    "lockout_reset_min": 10 ** 9, "sensor_realert_min": 720,   # lockout reset tested: hurts (routine lockouts)
    "snap_valid_min": 270, "snap_env": 4.5, "snap_v2": 0.55, "snap_r21": 0.9, "snap_v1": 1.1,
    "abs_scale_fixed": {"i": 0.012, "pw": 0.015, "pd": 0.012, "q": 0.02},   # fleet uncertainty of absolute physics
    "abs_scale_vfd": {"i": 0.04, "pw": 0.04, "pd": 0.025, "q": 0.03},
    "k_abs": 3.5, "abs_unexplained": 0.3,
    "ref_span": 360,            # slow reference used to isolate escalations on top of an ongoing hydraulic fault
    "no_drift_ch": ("wt", "tf"),     # thermal channels with strong unmodelled dynamics: no single-channel drift calls
    "cal_inflate_min": 360,     # residual scale x sqrt(1 + 360 / calibration minutes): early fits are uncertain
    "settle_min": {"bt": 90, "wt": 150, "tf": 20},   # thermal channels need time after a start to settle
    "close_after_type": {"CAVITATION": 25, "AIR_ENTRAINMENT": 25},
    "escalation_types": ("CAVITATION", "AIR_ENTRAINMENT"),
    "close_machine_min": 720,   # steady-running quiet minutes before a machine episode ends
    "close_sensor_min": 240,       # coupled channels must stay below this fraction of k for a single-channel sensor call
}


# ----------------------------------------------------------------------------------------------------------- loading
def load_inputs(data_dir: str, asset_ids) -> dict:
    """Read only the requested assets (test pumps are never loaded unless explicitly requested)."""
    ids = sorted(set(asset_ids))
    assets = pd.read_csv(os.path.join(data_dir, "assets.csv"))
    assets = assets[assets.asset_id.isin(ids)].reset_index(drop=True)
    feat = pd.read_parquet(os.path.join(data_dir, "features_1min.parquet"), filters=[("asset_id", "in", ids)])
    ctx = pd.read_csv(os.path.join(data_dir, "operating_context.csv"))
    ctx = ctx[ctx.asset_id.isin(ids)].reset_index(drop=True)
    sp = os.path.join(data_dir, "vibration_snapshots.parquet")
    snaps = pd.read_parquet(sp, filters=[("asset_id", "in", ids)]) if os.path.exists(sp) else None
    return {"assets": assets, "feat": feat, "ctx": ctx, "snaps": snaps}


# --------------------------------------------------------------------------------------------- static channel info
WIRELESS = ("v", "a", "c", "bt")


def static_channels(a: pd.Series) -> tuple[set, set]:
    """Available channels and sparse (battery wireless, ~5-min) channels from assets.csv only (static, causal)."""
    cols = set(str(a.channels_available).split("|")) if isinstance(a.channels_available, str) else set(CH.values())
    avail = {k for k, c in CH.items() if c in cols}
    wireless = (not pd.isna(getattr(a, "wireless_update_s", np.nan))) or a.instrumentation_tier == "T1_RETROFIT"
    sparse = {k for k in WIRELESS if k in avail} if wireless else set()
    return avail, sparse


# ------------------------------------------------------------------------------------------------------ frame build
def build_frame(feat: pd.DataFrame, ctx: pd.DataFrame, a: pd.Series) -> pd.DataFrame:
    ws, we = pd.Timestamp(a.window_start_utc), pd.Timestamp(a.window_end_utc)
    grid = pd.date_range(ws, we - pd.Timedelta(minutes=1), freq="1min", tz="UTC") if ws.tzinfo is None else \
        pd.date_range(ws, we - pd.Timedelta(minutes=1), freq="1min")
    f = feat.copy()
    f["t"] = pd.to_datetime(f["window_start_utc"], utc=True)
    f = f.drop(columns=["window_start_utc", "asset_id"]).set_index("t").reindex(grid)
    f.index.name = "t"
    f["rows"] = f["rows"].fillna(0)
    f["operating_mode"] = f["operating_mode"].fillna("NO_DATA")
    f["running_fraction"] = f["running_fraction"].fillna(0.0)
    c = ctx.copy()
    c["t"] = pd.to_datetime(c["timestamp_utc"], utc=True)
    c = c.set_index("t")[["duty_demand_pct", "supply_water_temp_c", "ambient_temp_c", "sea_state_hs_m",
                          "suction_static_head_m"]].reindex(grid).ffill()
    c.columns = ["ctx_dem", "ctx_tsup", "ctx_amb", "ctx_hs", "ctx_lvl"]
    f = f.join(c).reset_index()
    return f


def _runs(mask: np.ndarray) -> np.ndarray:
    """Length of the current run of True values ending at each index."""
    out = np.zeros(len(mask), dtype=np.int64)
    c = 0
    for j, x in enumerate(mask):
        c = c + 1 if x else 0
        out[j] = c
    return out


def _since(mask: np.ndarray) -> np.ndarray:
    """Minutes since the last True (large if never)."""
    out = np.full(len(mask), 10 ** 6, dtype=np.int64)
    last = None
    for j, x in enumerate(mask):
        if x:
            last = j
        if last is not None:
            out[j] = j - last
    return out


def ewm(x: np.ndarray, span: float) -> np.ndarray:
    """Causal EWMA that skips NaN (holds the last value)."""
    alpha = 2.0 / (span + 1.0)
    out = np.full(len(x), np.nan)
    s = np.nan
    for j, v in enumerate(x):
        if not np.isnan(v):
            s = v if np.isnan(s) else s + alpha * (v - s)
        out[j] = s
    return out


def ewm_seg(x: np.ndarray, span: float, reset: np.ndarray) -> np.ndarray:
    """Causal EWMA over non-NaN values that restarts after every reset (e.g. a stop or lockout)."""
    alpha = 2.0 / (span + 1.0)
    out = np.full(len(x), np.nan)
    s = np.nan
    for j, v in enumerate(x):
        if reset[j]:
            s = np.nan
        if not np.isnan(v):
            s = v if np.isnan(s) else s + alpha * (v - s)
        out[j] = s
    return out


def ffill_limit(x: np.ndarray, limit: int) -> np.ndarray:
    return pd.Series(x).ffill(limit=limit).values


# ---------------------------------------------------------------------------------------------------------- physics
def physics_columns(f: pd.DataFrame, a: pd.Series) -> pd.DataFrame:
    """Exogenous drivers and physics expectations from assets.csv and the documented control laws."""
    Qb = a.pump_rated_flow_m3_h / 3600.0
    Hb = a.pump_rated_head_m
    h0 = a.pump_shutoff_head_m / Hb
    Hst = a.static_head_m
    Qd = a.design_flow_m3_h / 3600.0
    Kd = (a.design_head_m - Hst) / Qd ** 2
    A = Hb * h0
    B = Hb * (h0 - 1.0) / Qb ** 2
    dem = np.clip(f["ctx_dem"].values / 100.0, 0.0, 1.3)
    tsup = f["ctx_tsup"].values
    ctrl = a.control
    rmin = a.min_speed_ratio if not pd.isna(a.min_speed_ratio) else 0.5
    if ctrl == "VFD_TEMPERATURE":
        need = dem * (36.0 - 25.0) / np.maximum(36.0 - tsup, 3.0)
        qreq = np.clip(need, 0.40, 1.15) * Qd
        r_exp = np.clip(np.sqrt((Hst + (B + Kd) * qreq ** 2) / A), rmin, 1.0)
        u = r_exp
    elif ctrl == "VFD_PRESSURE":
        u = np.clip(0.35 + 0.65 * dem, 0.2, 1.0)
        r_exp = np.full(len(f), np.nan)
    elif ctrl == "FLOW_VALVE":
        u = np.clip(dem, 0.2, 1.1)
        r_exp = np.full(len(f), np.nan)
    else:
        u = np.full(len(f), 1.0)
        r_exp = np.full(len(f), np.nan)
    f["u"] = u
    f["r_exp"] = r_exp
    avail, _ = static_channels(a)
    sp = f.get("motor_speed_rpm__mean")
    r_meas = (sp.values / a.motor_rated_rpm) if (sp is not None and "n" in avail) else np.where(f["run"].values, 1.0, 0.0)
    f["r_meas"] = r_meas
    f["rx"] = r_meas / r_exp - 1.0 if ctrl == "VFD_TEMPERATURE" else np.nan
    # pump-curve head ratio where suction, discharge and flow are measured (speed from the drive or rated)
    hr = np.full(len(f), np.nan)
    if all(k in avail and f.get(CH[k] + "__mean") is not None for k in ("ps", "pd", "q")):
        rho = 1000.0
        H_meas = (f[CH["pd"] + "__mean"].values - f[CH["ps"] + "__mean"].values) * 1000.0 / (rho * G)
        Q = f[CH["q"] + "__mean"].values / 3600.0
        r = np.where(np.isnan(r_meas), 1.0, r_meas)
        H_c = Hb * (h0 * r ** 2 - (h0 - 1.0) * (Q / Qb) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = np.where(H_c > 0.2 * Hb, H_meas / H_c - 1.0, np.nan)
            # sensitivity of the derived head ratio to an error in each instrument it is computed from
            f["dhr_pd"] = 1000.0 / (rho * G * H_c)
            f["dhr_ps"] = -1000.0 / (rho * G * H_c)
            f["dhr_q"] = (H_meas / H_c ** 2) * 2.0 * Hb * (h0 - 1.0) * Q / Qb ** 2 / 3600.0
    f["hr"] = hr
    return f


# Profile-level system constants (public fleet configuration / TECHNICAL_MODEL.md): suction loss at design flow [m],
# strainer share of it, control-valve share of the design resistance and its design opening, loop pressurisation [kPa],
# suction static head [m] where no level is measured, salinity [PSU]. Unknown profiles fall back to generic values.
PROFILE = {
    "SHIP_SW_COOLING": dict(sl=0.9, ss=0.65, vs=0.15, vd=0.90, p_loop=0.0, zs=4.8, sal=6.0),
    "SHIP_LT_FRESHWATER": dict(sl=0.6, ss=0.50, vs=0.20, vd=0.85, p_loop=120.0, zs=4.8, sal=0.0),
    "INDUSTRIAL_CW": dict(sl=0.7, ss=0.60, vs=0.30, vd=0.75, p_loop=0.0, zs=2.5, sal=0.0),
    "PROCESS_TRANSFER": dict(sl=0.5, ss=0.70, vs=0.35, vd=0.70, p_loop=0.0, zs=2.2, sal=0.0),
}
GENERIC_PROFILE = dict(sl=0.7, ss=0.6, vs=0.25, vd=0.8, p_loop=0.0, zs=3.0, sal=0.0)


def steady_state(a: pd.Series, dem, tsup, zs, clog=0.05, R=1.0, iw=0.0):
    """Quasi-steady healthy (or perturbed) operating point from assets.csv + documented control laws.

    Uses only the public physics relations (retrofit_sim.physics); returns expected channel values. Absolute values carry
    the uncertainty of the profile assumptions; the *differences* between perturbed and healthy states (fault
    signatures) are robust and are what the diagnosis uses.
    """
    from . import physics as ph
    pr = PROFILE.get(a.profile, GENERIC_PROFILE)
    dem = np.asarray(dem, dtype=float)
    tsup = np.asarray(tsup, dtype=float)
    Qb = a.pump_rated_flow_m3_h / 3600.0
    Hb = a.pump_rated_head_m
    h0 = a.pump_shutoff_head_m / Hb
    Hst = a.static_head_m
    Qd = a.design_flow_m3_h / 3600.0
    Hd = a.design_head_m
    Kd = (Hd - Hst) / Qd ** 2
    K_vfull = Kd * pr["vs"] * (50.0 ** (pr["vd"] - 1.0)) ** 2
    K_s0 = pr["sl"] / Qd ** 2
    K_s = K_s0 * (1.0 + pr["ss"] * ((1.0 + clog) ** 2 - 1.0))
    K_pipe = max(Kd * (1.0 - pr["vs"]) - K_s0, 0.1 * Kd) * R
    hf = 1.0 - iw
    rmin = a.min_speed_ratio if not pd.isna(a.min_speed_ratio) else 0.5
    ctrl = a.control
    n = len(dem)
    vp = np.full(n, pr["vd"])
    if ctrl == "VFD_TEMPERATURE":
        need = dem * (36.0 - 25.0) / np.maximum(36.0 - tsup, 3.0)
        q_req = np.clip(need, 0.40, 1.15) * Qd
        K_tot = K_pipe + ph.valve_K(vp, K_vfull) + K_s
        r = np.clip(ph.speed_for_flow(q_req, Hb, Qb, h0, Hst, K_tot, hf), rmin, 1.0)
    elif ctrl == "VFD_PRESSURE":
        vp = np.clip(0.35 + 0.65 * dem, 0.2, 1.0)
        dp = 0.85 * Hd
        K_tot = K_pipe + ph.valve_K(vp, K_vfull) + K_s
        Q2 = np.maximum(dp - Hst, 0.1) / K_tot
        r = np.clip(np.sqrt((dp + Hb * hf * (h0 - 1) / Qb ** 2 * Q2) / (Hb * h0 * hf)), rmin, 1.0)
    elif ctrl == "FLOW_VALVE":
        q_sp = np.clip(dem, 0.2, 1.1) * Qd
        r = np.ones(n)
        K_needed = (Hb * h0 * hf - Hst) / np.maximum(q_sp, 1e-4) ** 2 - Hb * (h0 - 1) * hf / Qb ** 2 - K_pipe - K_s
        rel = np.sqrt(K_vfull / np.maximum(K_needed, 1e-6))
        vp = np.clip(1.0 + np.log(np.clip(rel, 1e-3, 1.0)) / np.log(50.0), 0.05, 1.0)
    else:
        r = np.ones(n)
    K_tot = K_pipe + ph.valve_K(vp, K_vfull) + K_s
    Q = ph.solve_operating_point(r, Hb, Qb, h0, Hst, K_tot, hf)
    H = ph.pump_head(Q, r, Hb, Qb, h0, hf)
    rho = ph.water_density(tsup, pr["sal"])
    A_s = Qb / 2.5
    ps = (pr["p_loop"] * 1000.0 + rho * G * (zs - K_s * Q ** 2) - 0.5 * rho * (Q / A_s) ** 2) / 1000.0
    pdis = ps + rho * G * H / 1000.0
    eta = getattr(a, "pump_bep_efficiency", 0.8)
    P_sh = ph.shaft_power(Q, H, r, Qb, Hb, eta, h0, rho, hf)
    pf = 0.86
    eff = float(np.clip(a.motor_rated_kw * 1000.0 / (np.sqrt(3) * a.supply_voltage_v * a.motor_rated_current_a * pf), 0.85, 0.97))
    P_in, _, _, I, _ = ph.motor_electrical(P_sh, a.motor_rated_kw * 1000.0, eff, pf, a.supply_voltage_v,
                                           0.35, 0.97 if ctrl.startswith("VFD") else None)
    return {"r": r, "n": r * a.motor_rated_rpm, "q": Q * 3600.0, "ps": ps, "pd": pdis, "pw": P_in / 1000.0, "i": I,
            "vp": vp * 100.0, "hr": np.full(n, hf - 1.0)}


SIG_FAULTS = {"SUCTION_STRAINER_CLOGGING": dict(clog=1.0), "DISCHARGE_RESTRICTION": dict(R=1.3),
              "IMPELLER_WEAR": dict(iw=0.05),
              # large magnitudes: control valves / drives saturate, so the signature changes direction
              "SUCTION_STRAINER_CLOGGING@L": dict(clog=3.5), "DISCHARGE_RESTRICTION@L": dict(R=2.2),
              "IMPELLER_WEAR@L": dict(iw=0.12)}


def physics_signatures(f: pd.DataFrame, a: pd.Series) -> None:
    """Per-minute expected change of every hydraulic channel for a small clog / restriction / impeller wear."""
    pr = PROFILE.get(a.profile, GENERIC_PROFILE)
    dem = np.clip(np.nan_to_num(f["ctx_dem"].values, nan=70.0) / 100.0, 0.05, 1.3)
    if a.control in ("FIXED_SPEED",):
        dem = np.full(len(f), 0.8)
    tsup = np.nan_to_num(pd.Series(f["ctx_tsup"].values).ffill().values, nan=15.0)
    lvl = f["ctx_lvl"].values
    zs = np.where(np.isfinite(lvl), lvl, pr["zs"])
    base = steady_state(a, dem, tsup, zs)
    for k in ("n", "q", "ps", "pd", "pw", "i", "vp"):
        f["ph_" + k] = base[k]
    r_exp = np.where(np.isfinite(f["r_exp"].values), f["r_exp"].values, base["r"])
    for name, kw in SIG_FAULTS.items():
        st = steady_state(a, dem, tsup, zs, **kw)
        for k in ("n", "q", "ps", "pd", "pw", "i", "vp", "hr"):
            f[f"sig_{name}_{k}"] = st[k] - base[k] if k != "hr" else st[k]
        f[f"sig_{name}_rx"] = (st["r"] - base["r"]) / np.maximum(r_exp, 0.3)


# ----------------------------------------------------------------------------------------------- context residuals
def _design(f: pd.DataFrame, k: str, ctrl: str) -> np.ndarray:
    u = f["u"].values
    ones = np.ones(len(f))
    if k in ("bt", "wt"):
        span = 50 if k == "bt" else 80          # ~2x the thermal time constant in minutes
        uu = ewm(np.where(f["run"].values, u, 0.0), span)
        amb = ewm(f["ctx_amb"].values, span)
        tf = ewm(f["ctx_tsup"].values, span)
        cols = [ones, uu, uu ** 2, amb, tf] if ctrl != "FIXED_SPEED" else [ones, amb, tf]
        return np.column_stack(cols)
    lvl = f["ctx_lvl"].values
    has_lvl = a_profile(f) in ("INDUSTRIAL_CW", "PROCESS_TRANSFER")      # profiles with a measured suction level
    if k in ("ps", "pd") and has_lvl:
        base = [ones] if ctrl == "FIXED_SPEED" else [ones, u, u ** 2, u ** 3]
        pr = PROFILE.get(a_profile(f), GENERIC_PROFILE)
        return np.column_stack(base + [np.nan_to_num(pd.Series(lvl).ffill().values, nan=pr["zs"])])
    if k == "tf":
        return np.column_stack([ones, ewm(f["ctx_tsup"].values, 3), u])
    if k in VIB:
        r = np.nan_to_num(f["r_meas"].values, nan=1.0)
        return np.column_stack([ones, r, r ** 2]) if ctrl != "FIXED_SPEED" and ctrl != "FLOW_VALVE" else \
            np.column_stack([ones, u] if ctrl == "FLOW_VALVE" else [ones])
    if ctrl == "FIXED_SPEED":
        return np.column_stack([ones])
    return np.column_stack([ones, u, u ** 2, u ** 3])


def a_profile(f: pd.DataFrame) -> str:
    return f.attrs.get("profile", "")


def _ridge(X, y, lam=1e-3):
    XtX = X.T @ X
    n = XtX.shape[0]
    reg = lam * np.trace(XtX) / max(n, 1) * np.eye(n)
    reg[0, 0] = 0.0
    return np.linalg.solve(XtX + reg, X.T @ y)


def residual_z(f: pd.DataFrame, a: pd.Series, P: dict) -> tuple[pd.DataFrame, list]:
    """Causal context-model residual z-scores for every available channel plus the physics channels rx and hr."""
    n = len(f)
    steady = f["steady"].values
    ctrl = a.control
    t_idx = np.arange(n)
    cal_end = int(P["cal_hours"] * 60)
    checkpoints = list(range(P["min_cal_min"], cal_end + 1, P["refit_every_min"]))
    if not checkpoints or checkpoints[-1] != cal_end:
        checkpoints.append(cal_end)
    st_avail, st_sparse = static_channels(a)
    avail = [k for k in CH if k in st_avail and CH[k] + "__mean" in f]
    channels = [k for k in avail if k not in ("n",) or ctrl != "VFD_TEMPERATURE"] + \
               (["rx"] if ctrl == "VFD_TEMPERATURE" and "n" in avail else []) + \
               (["hr"] if all(k in avail for k in ("ps", "pd", "q")) else [])
    if "n" in avail and ctrl == "VFD_TEMPERATURE":
        channels.append("n")
    for k in channels:
        if k in ("rx", "hr"):
            x = f[k].values.astype(float)
            sparse = False
        else:
            x = f[CH[k] + "__mean"].values.astype(float)
            sparse = k in st_sparse
            if sparse:
                x = ffill_limit(x, 6)
        X = _design(f, k, ctrl) if k not in ("rx", "hr") else np.ones((n, 1))
        pred = np.full(n, np.nan)
        scale = np.full(n, np.nan)
        lo_u = np.full(n, np.nan)
        hi_u = np.full(n, np.nan)
        ok = steady & ~np.isnan(x) & np.all(np.isfinite(X), axis=1)
        prev_end = 0
        coef, sc, ulo, uhi, ncal = None, None, None, None, 1
        ncal_arr = np.full(n, np.nan)
        for cp in checkpoints + [n]:
            # rows [prev_end, cp) use the fit made from data strictly before prev_end
            if coef is not None and prev_end < n:
                seg = slice(prev_end, min(cp, n))
                pred[seg] = X[seg] @ coef
                scale[seg] = sc
                ncal_arr[seg] = ncal
                lo_u[seg], hi_u[seg] = ulo, uhi
            if cp >= n:
                break
            sel = ok & (t_idx < cp)
            if sel.sum() >= P["min_cal_min"] * (0.2 if sparse else 1.0):
                Xs, ys = X[sel], x[sel]
                if k in ("rx", "hr"):
                    env = P["rx_env"] if k == "rx" else P["hr_env"]
                    b = float(np.clip(np.median(ys), env[0], env[1]))
                    coef = np.array([b])
                    res = ys - b
                else:
                    coef = _ridge(Xs, ys)
                    res = ys - Xs @ coef
                    keep = np.abs(res - np.median(res)) < 4 * 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
                    if keep.sum() > 0.5 * len(ys):
                        coef = _ridge(Xs[keep], ys[keep])
                        res = ys - Xs @ coef
                mad = 1.4826 * np.median(np.abs(res - np.median(res)))
                lvl = abs(np.median(ys))
                floor = {"rx": 0.004, "hr": 0.008}.get(k, max(ABS_FLOOR.get(k, 0.1), REL_FLOOR.get(k, 0.0) * lvl))
                sc = max(mad, floor)
                uu = f["u"].values[sel]
                ulo, uhi = np.nanmin(uu), np.nanmax(uu)
                ncal = int(sel.sum()) * (5 if sparse else 1)
            prev_end = cp
        # extrapolation guard: inflate the scale when the operating point is outside the calibration range
        u = f["u"].values
        width = np.maximum(hi_u - lo_u, 0.05)
        out = np.maximum(lo_u - u, 0) + np.maximum(u - hi_u, 0)
        scale_eff = scale * (1.0 + 4.0 * out / width) * np.sqrt(1.0 + P["cal_inflate_min"] / np.maximum(ncal_arr, 1.0))
        z = (x - pred) / scale_eff
        z[~f["run"].values] = np.nan
        if k in P["settle_min"]:
            z[f["since_start"].values < P["settle_min"][k]] = np.nan
        f["z_" + k] = z
        f["e_" + k] = x - pred          # residual in engineering units (for evidence text)
        f["s_" + k] = scale_eff
        f["x_" + k] = x
    return f, channels


# ----------------------------------------------------------------------------------------------------- detectors
def _cfg_alert(a, detector, cls, ftype, conf, evidence, action):
    return {"asset_id": a.asset_id, "alert_time_utc": None, "event_class": cls, "fault_type": ftype,
            "confidence": float(np.clip(conf, 0.05, 0.99)), "evidence": evidence, "recommended_action": action,
            "detector": detector}


ACTIONS = {
    "BEARING_WEAR": "Plan a pump bearing inspection: check lubrication, listen at the bearing housing and schedule a bearing "
                    "replacement at the next stop; trend vibration daily until then.",
    "SUCTION_STRAINER_CLOGGING": "Inspect and clean (backflush) the suction strainer at the next opportunity; check the "
                                 "suction-side valves are fully open.",
    "CAVITATION": "Reduce the risk of cavitation: check strainer and suction level / submergence, reduce flow if the process "
                  "allows, and inspect for crackling noise at the pump inlet.",
    "DISCHARGE_RESTRICTION": "Check discharge-side valves (position after maintenance) and heat-exchanger fouling / "
                             "pressure drop; plan cleaning.",
    "MISALIGNMENT": "Check shaft alignment and coupling condition (laser alignment), especially after recent maintenance.",
    "IMBALANCE": "Inspect the impeller for fouling or damage and plan cleaning / balancing.",
    "IMPELLER_WEAR": "Plan a performance test against the pump curve and inspect the impeller and wear rings.",
    "AIR_ENTRAINMENT": "Check suction level, sea-chest venting / vortexing and air ingress on the suction side.",
    "SENSOR_DRIFT": "Verify the transmitter against a portable reference gauge and recalibrate; do not act on the machine "
                    "based on this channel until verified.",
    "SENSOR_CALIBRATION_SHIFT": "Check the transmitter zero / span against a reference (recent work on the instrument?); "
                                "recalibrate before acting on this reading.",
    "SENSOR_STUCK": "Check the sensor, cabling and transmitter: the reading is frozen while the process varies.",
    "SENSOR_NOISE_SPIKES": "Check cable screening / grounding and the transmitter: the reading is noisy or spiky while "
                           "coupled channels are steady.",
    "COMMUNICATION_LOSS": "Check the gateway / network link and wireless sensor battery; data are missing.",
    "UNKNOWN": "Inspect the pump; the pattern does not match a known fault clearly.",
}
CLASS = {k: ("SENSOR" if k.startswith("SENSOR") or k == "COMMUNICATION_LOSS" else "MACHINE") for k in ACTIONS}


class Track:
    """Episode tracker for one detector.

    While the detector is active it votes a fault type every minute. A type is alerted once it has been the vote for
    `stable_min` consecutive minutes (or, if votes stay split, the majority type after `max_wait_min`). Each type has
    its own sub-episode: it is alerted at most once until it has been absent for `close_after` minutes (shorter for
    episodic faults such as cavitation), so a clog that escalates to cavitation, or repeated cavitation episodes,
    each get their own alert, but a stable problem never re-alerts.
    """

    def __init__(self, name, P, close_after=None, escalate_only=True, realert_min=0):
        self.realert_min = realert_min
        self.last_alert_of = {}
        self.name = name
        self.P = P
        self.close_after = close_after if close_after is not None else P["close_after_min"]
        self.escalate_only = escalate_only
        self.quiet = 0
        self.open = False
        self.last_active = -10 ** 9
        self.start = None
        self.votes = []
        self.type_last = {}        # type -> last minute voted
        self.type_alerted = {}     # type -> True while its sub-episode is alerted and open
        self.any_alert = False

    def reset(self):
        """Close the episode (e.g. the pump was locked out for work: a repair resolves the problem)."""
        self.open = False
        self.type_alerted = {}
        self.quiet = 0

    def _close_after(self, ft):
        return self.P["close_after_type"].get(ft, self.P["close_after_min"])

    def step(self, j, active, ftype, observed=True):
        """observed=False (pump stopped, start-up transient, data gap) neither extends nor ends an episode."""
        P = self.P
        emit = None
        for ft in list(self.type_alerted):
            if ft in P["close_after_type"] and j - self.type_last.get(ft, -10 ** 9) > self._close_after(ft):
                del self.type_alerted[ft]
        if not active and not observed:
            return None
        if active:
            self.quiet = 0
            if not self.open:
                self.open, self.start, self.votes, self.any_alert = True, j, [], False
            self.last_active = j
            self.votes.append(ftype)
            self.type_last[ftype] = j
            recent = self.votes[-P["stable_min"]:]
            stable = len(recent) >= P["stable_min"] and len(set(recent)) == 1
            cand = None
            if stable:
                cand = recent[-1]
            elif not self.any_alert and j - self.start >= P["max_wait_min"]:
                vals, cnt = np.unique([v for v in self.votes[-P["max_wait_min"]:] if v is not None] or ["UNKNOWN"],
                                      return_counts=True)
                cand = vals[np.argmax(cnt)]
            if cand is not None and self.any_alert and self.escalate_only and cand not in P["escalation_types"]:
                cand = None
            if cand is not None and self.realert_min and j - self.last_alert_of.get(cand, -10 ** 9) < self.realert_min:
                cand = None
            if cand is not None and cand not in self.type_alerted and not (cand == "UNKNOWN" and self.any_alert):
                emit = cand
                self.last_alert_of[cand] = j
                self.type_alerted[cand] = True
                self.type_last[cand] = j
                self.any_alert = True
        else:
            self.quiet += 1
            if self.open and self.quiet > self.close_after:
                self.open = False
                self.type_alerted = {}
        return emit


def _persist(flag: np.ndarray, n: int) -> np.ndarray:
    return _runs(flag) >= n


def detect_asset(f: pd.DataFrame, a: pd.Series, channels: list, P: dict, snaps=None) -> list:
    st_avail, st_sparse = static_channels(a)
    n = len(f)
    t = f["t"]
    steady = f["steady"].values
    run = f["run"].values
    alerts = []

    # ---------------- gateway / communication loss
    gap = (f["rows"].values == 0)
    gap_run = _runs(gap)
    tr = Track("gap", P)
    for j in range(n):
        if gap_run[j] == P["gap_min"]:
            alerts.append((j, _cfg_alert(a, "gap", "SENSOR", "COMMUNICATION_LOSS", 0.9,
                                         f"No telemetry rows for {P['gap_min']} min (gateway / network outage).",
                                         ACTIONS["COMMUNICATION_LOSS"])))

    Z = {k: f["z_" + k].values for k in channels}
    reset = ~run
    ZF = {k: ewm_seg(np.where(steady, Z[k], np.nan), 5, reset) for k in channels}
    ZS = {k: ewm_seg(np.where(steady, Z[k], np.nan), P["slow_span"], reset) for k in channels}
    ZF = {k: np.where(steady, np.nan_to_num(v), 0.0) for k, v in ZF.items()}
    ZS = {k: np.where(steady, v, np.nan) for k, v in ZS.items()}
    ES = {k: ewm_seg(np.where(steady, f["e_" + k].values, np.nan), 15, reset) for k in channels}
    act = {}
    for k in channels:
        fast = _persist(steady & (np.abs(ZF[k]) > P["k_fast"]), P["persist_fast"])
        slow = _persist(steady & (np.abs(np.nan_to_num(ZS[k])) > P["k_slow"]), P["persist_slow"])
        act[k] = fast | slow
    zmag = {k: np.where(np.abs(ZF[k]) > np.abs(np.nan_to_num(ZS[k])), ZF[k], np.nan_to_num(ZS[k])) for k in channels}

    # ---------------- per-channel instrument faults: stuck and noise
    sensor_open = {k: np.zeros(n, dtype=bool) for k in channels}
    for k in SENSOR_CH:
        col = CH[k]
        if k not in st_avail or col + "__mean" not in f:
            continue
        cnt = f[col + "__count"].fillna(0).values
        sparse = k in st_sparse
        mean = f[col + "__mean"].values
        # stuck: identical consecutive values while running
        if sparse:
            stuck_flag = np.zeros(n, dtype=bool)
            last, same, last_j, nat = None, 0, -1, 0
            cal_end = int(P["cal_hours"] * 60)
            for j in range(n):
                v = mean[j]
                if not np.isnan(v):
                    same = same + 1 if (last is not None and v == last) else 1
                    last, last_j = v, j
                    if j < cal_end and run[j]:
                        nat = max(nat, same)       # longest natural constant run seen during calibration
                thr = max(P["stuck_sparse_samples"], int(np.ceil(P["stuck_nat_mult"] * nat)))
                stuck_flag[j] = run[j] and same >= thr and j - last_j <= 10 and not np.isnan(mean[last_j] if last_j >= 0 else np.nan)
        else:
            sd = f[col + "__std"].values
            mx, mn = f[col + "__max"].values, f[col + "__min"].values
            flat = (sd == 0) | (mx == mn)
            same_as_prev = np.r_[False, mean[1:] == mean[:-1]]
            stuck_flag = _persist(run & flat & (same_as_prev | flat) & (cnt >= 30), P["stuck_min"])
        trk = Track("stuck_" + k, P, P["close_sensor_min"], realert_min=P["sensor_realert_min"])
        for j in range(n):
            ft = trk.step(j, bool(stuck_flag[j]), "SENSOR_STUCK", bool(run[j]))
            sensor_open[k][j] |= trk.open
            if ft:
                alerts.append((j, _cfg_alert(a, "stuck", "SENSOR", "SENSOR_STUCK", 0.85,
                                             f"{NAME[k]} ({col}) frozen at {_last_valid(mean, j):.2f} while the pump runs and coupled "
                                             f"channels vary.", ACTIONS["SENSOR_STUCK"])))
        # noise / spikes (dense channels only): 1-min std far above its calibration level, coupled channels calm
        if not sparse and col + "__std" in f:
            sd = f[col + "__std"].values.astype(float)
            ref = _causal_ref(sd, steady, P)
            ratio = sd / np.maximum(ref, 1e-6)
            cpl = [c for c in COUPLED.get(k, []) if c in st_avail and c not in st_sparse and CH[c] + "__std" in f]
            cr = np.zeros(n)
            for c in cpl:
                sdc = f[CH[c] + "__std"].values.astype(float)
                rc = sdc / np.maximum(_causal_ref(sdc, steady, P), 1e-6)
                cr = np.maximum(cr, np.nan_to_num(rc))
            rr = ewm(np.nan_to_num(ratio, nan=1.0), 5)
            heavy = np.nan_to_num(f["ctx_hs"].values) > 2.0
            nflag = _persist(run & (rr > P["noise_ratio"]) & (cr < 0.5 * rr) & ~(heavy & (k in ("ps", "pd"))),
                             P["noise_persist"])
            trk = Track("noise_" + k, P, P["close_sensor_min"], realert_min=P["sensor_realert_min"])
            for j in range(n):
                ft = trk.step(j, bool(nflag[j]), "SENSOR_NOISE_SPIKES", bool(run[j]))
                sensor_open[k][j] |= trk.open
                if ft:
                    alerts.append((j, _cfg_alert(a, "noise", "SENSOR", "SENSOR_NOISE_SPIKES", 0.8,
                                                 f"{NAME[k]} ({col}) 1-min scatter {rr[j]:.1f}x its normal level; coupled "
                                                 f"channels steady.", ACTIONS["SENSOR_NOISE_SPIKES"])))

    # ---------------- main loop: hydraulic signature diagnosis, single-channel instrument faults, machine tracks
    ZC = {k: ewm_seg(np.where(steady, Z[k], np.nan), 15, reset) for k in channels}
    strong = {k: _persist(steady & (np.abs(ZF[k]) > P["k_strong"]), 3) for k in channels}
    for k in channels:
        act[k] = act[k] | strong[k]
    hyd_ch = [c for c in HYD if c in channels and not (c == "n" and "rx" in channels)]
    sig_ch = [c for c in hyd_ch if f"sig_SUCTION_STRAINER_CLOGGING_{c}" in f]
    other_sensor = [c for c in SENSOR_CH if c in channels and c not in hyd_ch and c not in P["no_drift_ch"]]
    single_trk = {k: Track("single_" + k, P, P["close_sensor_min"], realert_min=P["sensor_realert_min"]) for k in hyd_ch + other_sensor if k in SENSOR_CH or k == "rx"}
    sig = {name: {k: f[f"sig_{name}_{k}"].values for k in sig_ch} for name in SIG_FAULTS}
    scl = {k: f["s_" + k].values for k in channels}
    crest = f[CH["c"] + "__mean"].values if CH["c"] + "__mean" in f else np.full(n, np.nan)
    crest_s = ewm_seg(np.where(steady, crest, np.nan), 30, reset) if not np.all(np.isnan(crest)) else np.full(n, np.nan)
    since_lock = f["since_lockout"].values
    cm = P["close_machine_min"]
    trk_b = Track("bearing", P, cm)
    trk_v = Track("velocity", P, cm)
    trk_h = Track("hydraulic", P, cm)
    zc_hist = {k: ZC[k] for k in channels}
    # absolute physics residuals (fleet prior): catch hydraulic faults already present at window start
    vfd = a.control.startswith("VFD")
    abs_sc = P["abs_scale_vfd"] if vfd else P["abs_scale_fixed"]
    abs_ch = [k for k in ("i", "pw", "pd", "q") if k in channels and "ph_" + k in f]
    ZA = {}
    for k in abs_ch:
        ph_k = np.abs(f["ph_" + k].values)
        rel = (f["x_" + k].values - f["ph_" + k].values) / np.maximum(ph_k, 1e-6)
        ZA[k] = ewm_seg(np.where(steady, rel / abs_sc[k], np.nan), 30, reset)
    sigA = {name: {k: f[f"sig_{name}_{k}"].values / (np.maximum(np.abs(f["ph_" + k].values), 1e-6) * abs_sc[k])
                   for k in abs_ch} for name in SIG_FAULTS}
    trk_a = Track("absolute", P, P["close_machine_min"])
    # vibration snapshots (every ~4 h): latest snapshot taken before the end of minute j
    sf = snapshot_features(snaps, a)
    snap_idx = np.full(n, -1)
    if len(sf):
        ts = pd.to_datetime(sf["t"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
        tend = (f["t"] + pd.Timedelta(minutes=1)).values.astype("datetime64[ns]").astype(np.int64)
        snap_idx = np.searchsorted(ts, tend, side="left") - 1
        age = np.where(snap_idx >= 0, (tend - ts[np.maximum(snap_idx, 0)]) / 6e10, np.inf)
        snap_idx = np.where(age <= P["snap_valid_min"], snap_idx, -1)
    SV1, SV2, SENV = (sf["v1"].values, sf["v2"].values, sf["env_bpfo"].values) if len(sf) else (None, None, None)
    ZL = {k: ewm_seg(np.where(steady, Z[k], np.nan), P["ref_span"], reset) for k in channels}
    labels = np.array([None] * n, dtype=object)

    def zget(k, j, src):
        if k not in src or sensor_open.get(k, np.zeros(n, bool))[j]:
            return 0.0
        v = src[k][j]
        return 0.0 if np.isnan(v) else float(v)

    lock = f["operating_mode"].values == "LOCKED_OUT"
    lock_run = _runs(lock)
    for j in range(n):
        if j > 0 and lock[j - 1] and not lock[j] and lock_run[j - 1] >= P["lockout_reset_min"]:
            for tk in (trk_b, trk_v, trk_h):
                tk.reset()
        if not steady[j]:
            for tk in list(single_trk.values()) + [trk_b, trk_v, trk_h, trk_a]:
                tk.step(j, False, None, False)
            continue
        zz = {k: zget(k, j, zmag) for k in channels}
        zc = {k: zget(k, j, ZC) for k in channels}
        active = {k for k in channels if act[k][j] and not sensor_open.get(k, np.zeros(n, bool))[j]}
        hyd_active = [k for k in hyd_ch if k in active]
        # --- hydraulic diagnosis by physics signatures (or a single instrument)
        lab = None
        if hyd_active:
            if trk_h.open and trk_h.any_alert:
                # an alerted hydraulic problem is ongoing: diagnose only the change on top of it (escalation)
                dzc = {k: zc[k] - _lag(ZL[k], j, 30) for k in zc}
                dzz = {k: zz[k] - _lag(ZL[k], j, 30) for k in zz}
                if zz.get("ps", 0.0) < -P["cav_ps_z"] and (zz.get("a", 0.0) > 3 or zz.get("v", 0.0) > 3) and \
                        (zz.get("pd", 0.0) < -3 or zz.get("hr", 0.0) < -3):
                    lab = "CAVITATION"
                elif zz.get("ps", 0.0) < -3.0:
                    lab = None          # low suction: never call air entrainment on top of a clog
                else:
                    lab = _classify_hyd(dzc, dzz, sig_ch, hyd_ch, sig, scl, j, f, a.control, P, escalation_only=True)
            else:
                lab = _classify_hyd(zc, zz, sig_ch, hyd_ch, sig, scl, j, f, a.control, P)
        labels[j] = lab
        # --- single-channel instrument faults (hydraulic channels via the classifier, others via coupling)
        for k, tk in single_trk.items():
            if k in hyd_ch:
                single = bool(act[k][j]) and lab == "SENSOR:" + k
            else:
                cpl = [c for c in COUPLED.get(k, []) if c in channels]
                cmax = max([abs(zz.get(c, 0.0)) for c in cpl] + [0.0])
                dz = abs(zc[k] - _lag(zc_hist[k], j, 240))
                dc = max([abs(zc[c] - _lag(zc_hist[c], j, 240)) for c in cpl] + [0.0])
                single = bool(act[k][j]) and ((cmax < P["single_ratio"] * abs(zz[k]) and cmax < 2.5) or
                                              (dz > P["k_slow"] and dc < P["single_ratio"] * dz and dc < 2.0))
            ftype = _step_or_ramp(ZF[k], j, P, steady) if single else None
            ft = tk.step(j, single, ftype, True)
            if k in sensor_open:
                sensor_open[k][j] |= tk.open and tk.any_alert
            if ft:
                kk = k if k != "rx" else "n"
                alerts.append((j, _cfg_alert(a, "single", "SENSOR", ft, 0.7,
                                             f"{NAME[k]} deviates alone: {ES[k][j]:+.2f} from the context-expected value "
                                             f"(z={zz[k]:+.1f}); the physically coupled channels do not show the change a real "
                                             f"machine fault would cause. Pattern: {'step change' if ft == 'SENSOR_CALIBRATION_SHIFT' else 'gradual ramp'}.",
                                             ACTIONS[ft].replace("the transmitter", f"the {NAME[kk]} transmitter"))))
        # --- snapshot spectra (fleet-level thresholds): BPFO envelope, 1x / 2x running-speed velocity lines
        snap_brg = snap_mis = snap_imb = False
        si = snap_idx[j]
        if si >= 0:
            v1, v2, envo = SV1[si], SV2[si], SENV[si]
            snap_brg = envo >= P["snap_env"]
            snap_mis = v2 >= P["snap_v2"] and v2 >= P["snap_r21"] * v1
            snap_imb = (not snap_mis) and v1 >= P["snap_v1"] and v2 < 0.7 * v1
        # --- bearing: acceleration / crest factor (fleet crest prior catches damage present at window start)
        crest_hi = (not np.isnan(crest_s[j])) and crest_s[j] > P["crest_abs"]
        brg = (("a" in active and zz.get("a", 0) > 0) or ("c" in active and zz.get("c", 0) > 0) or crest_hi or snap_brg)
        cav_now = lab in ("CAVITATION", "AIR_ENTRAINMENT") or zz.get("ps", 0.0) < -P["cav_ps_z"]
        b_type = "BEARING_WEAR" if brg and not cav_now and (snap_brg or crest_hi or not (zz.get("v", 0) > 2 * max(zz.get("a", 0), 0.1))) else None
        e = trk_b.step(j, b_type is not None, b_type)
        if e:
            sn = f" Vibration snapshot: bearing outer-race (BPFO) envelope peak {SENV[si]:.1f}x the noise floor (healthy < 4)." \
                if si >= 0 and snap_brg else ""
            alerts.append((j, _cfg_alert(a, "bearing", "MACHINE", e, 0.85 if snap_brg else 0.7,
                                         _evidence(f, j, ["a", "c", "bt", "v"], zz, crest_s, ES) + sn, ACTIONS[e])))
        # --- velocity family: imbalance / misalignment; a bearing in late stage raises acceleration much more than velocity
        v_up = "v" in active and zz.get("v", 0) > 0
        v_type = None
        if snap_mis and not cav_now:
            v_type = "MISALIGNMENT"
        elif snap_imb and not cav_now:
            v_type = "IMBALANCE"
        elif v_up and not cav_now and not (lab is not None and lab.startswith(("SUCTION", "DISCHARGE"))):
            brg_explains = (zz.get("a", 0) > max(0.5 * zz.get("v", 0), P["k_slow"])) or snap_brg or crest_hi
            if not brg_explains:
                mis = (zz.get("bt", 0) > 1.5 or zz.get("wt", 0) > 1.5 or since_lock[j] < P["post_maint_min"]
                       or _step_or_ramp(ZF["v"], j, P, steady) == "SENSOR_CALIBRATION_SHIFT")
                v_type = "MISALIGNMENT" if mis else "IMBALANCE"
        e = trk_v.step(j, v_type is not None, v_type)
        if e:
            sn = (f" Vibration snapshot: 1x {SV1[si]:.2f} mm/s, 2x {SV2[si]:.2f} mm/s"
                  f" ({'2x dominant: typical of misalignment' if snap_mis else '1x dominant: typical of imbalance' if snap_imb else 'no clear harmonic pattern'}).") if si >= 0 else ""
            alerts.append((j, _cfg_alert(a, "velocity", "MACHINE", e, 0.8 if (snap_mis or snap_imb) else 0.6,
                                         _evidence(f, j, ["v", "bt", "wt", "a"], zz, crest_s, ES) + sn, ACTIONS[e])))
        # --- absolute physics track (pre-existing hydraulic faults)
        a_type = None
        if abs_ch and not any(sensor_open.get(k, np.zeros(n, bool))[j] for k in abs_ch):
            za = np.array([0.0 if np.isnan(ZA[k][j]) else ZA[k][j] for k in abs_ch])
            if np.sum(np.abs(za) > 2.0) >= 2 and np.max(np.abs(za)) > P["k_abs"]:
                nz = float(za @ za)
                bestA, resA = None, nz
                for name, sg in sigA.items():
                    sv = np.array([sg[k][j] for k in abs_ch])
                    ss_ = float(sv @ sv)
                    if ss_ < 1e-9:
                        continue
                    al = float(za @ sv) / ss_
                    res = nz - al * float(za @ sv) if al > 0 else nz
                    if res < resA:
                        bestA, resA = name.split("@")[0], res
                if bestA is not None and resA < P["abs_unexplained"] * nz:
                    a_type = bestA
        e = trk_a.step(j, a_type is not None, a_type)
        if e and not (trk_h.open and e in trk_h.type_alerted):
            ev_txt = "; ".join(f"{NAME[k]} {100 * ZA[k][j] * abs_sc[k]:+.1f}% vs pump/system-curve physics" for k in abs_ch
                               if not np.isnan(ZA[k][j]))
            alerts.append((j, _cfg_alert(a, "absolute", "MACHINE", e, 0.6, ev_txt + f" (expected from assets.csv ratings at "
                                         f"the current duty; deviation present since the start of monitoring).", ACTIONS[e])))
        # --- hydraulic machine track
        h_type = lab if (lab is not None and not lab.startswith("SENSOR:")) else None
        e = trk_h.step(j, h_type is not None, h_type)
        if e and trk_a.open and e in trk_a.type_alerted:
            e = None
        if e:
            alerts.append((j, _cfg_alert(a, "hydraulic", "MACHINE", e, 0.7, _evidence(f, j, hyd_ch + ["a", "v"], zz, crest_s, ES),
                                         ACTIONS[e])))
    return _finalize(alerts, f, a)


def _last_valid(x: np.ndarray, j: int) -> float:
    v = x[:j + 1][~np.isnan(x[:j + 1])]
    return float(v[-1]) if len(v) else float("nan")


def _lag(x: np.ndarray, j: int, lag: int) -> float:
    v = x[j - lag] if j - lag >= 0 else np.nan
    return 0.0 if np.isnan(v) else float(v)


def _classify_hyd(zc, zz, sig_ch, hyd_ch, sig, scl, j, f, ctrl, P, escalation_only=False):
    """Best explanation of the hydraulic residual vector: a physics fault signature, or one instrument alone."""
    chans = [k for k in hyd_ch if not np.isnan(scl[k][j])]
    if not chans:
        return None
    z = np.array([zc[k] for k in chans])
    nz = float(z @ z)
    if nz < 1e-6:
        return None
    g = lambda k: zc.get(k, 0.0)
    q = lambda k: zz.get(k, 0.0)
    k3 = 3.0
    # episodic hydraulic faults first (fast residuals: episodes can be short)
    vib = q("a") > k3 or q("v") > k3
    head_low = q("hr") < -k3 or q("pd") < -k3
    if q("ps") < -P["cav_ps_z"] and vib and head_low:
        return "CAVITATION"
    if escalation_only and nz < P["k_slow"] ** 2:
        return None
    best, best_res = None, nz
    fits = {}
    for name, sg in sig.items():
        s = np.array([sg[k][j] / scl[k][j] if k in sg else 0.0 for k in chans])
        ss = float(s @ s)
        if ss < 1e-6:
            continue
        alpha = float(z @ s) / ss
        res = nz - alpha * float(z @ s) if alpha > 0 else nz
        name = name.split("@")[0]
        if res < fits.get(name, np.inf):
            fits[name] = res
        if res < best_res:
            best, best_res = name, res
    # one instrument alone (either sign); an error in pd / ps / q also moves the derived head ratio hr
    sens_k, sens_res = None, nz
    for k in chans:
        if k == "hr":
            continue
        sv = np.zeros(len(chans))
        sv[chans.index(k)] = 1.0
        if "hr" in chans and k in ("pd", "ps", "q") and "dhr_" + k in f:
            c = f["dhr_" + k].values[j]
            if np.isfinite(c):
                sv[chans.index("hr")] = c * scl[k][j] / scl["hr"][j]
        zs_ = float(z @ sv)
        res = nz - zs_ ** 2 / float(sv @ sv)
        if res < sens_res:
            sens_k, sens_res = k, res
    if sens_k is None:
        sens_k = max(chans, key=lambda k: abs(g(k)))
    if (sens_k in SENSOR_CH or sens_k == "rx") and not escalation_only:
        if (best is None or sens_res + P["sensor_margin"] < best_res) and sens_res < 0.35 * nz:
            return "SENSOR:" + sens_k
    if escalation_only:
        if best == "IMPELLER_WEAR" and best_res < P["unexplained"] * nz and q("ps") > -k3:
            hs, lvl = f["ctx_hs"].values[j], f["ctx_lvl"].values[j]
            if ((not np.isnan(hs)) and hs > P["air_hs"]) or ((not np.isnan(lvl)) and lvl < P["air_lvl"]) or vib:
                return "AIR_ENTRAINMENT"
        return None
    if best is None or best_res > P["unexplained"] * nz:
        return "UNKNOWN"
    if best == "IMPELLER_WEAR":
        hs, lvl = f["ctx_hs"].values[j], f["ctx_lvl"].values[j]
        weather = (not np.isnan(hs)) and hs > P["air_hs"]
        low_level = (not np.isnan(lvl)) and lvl < P["air_lvl"]
        if (weather or low_level or vib) and q("ps") > -k3:
            return "AIR_ENTRAINMENT"
    return best


def _causal_ref(sd: np.ndarray, steady: np.ndarray, P: dict) -> np.ndarray:
    """Calibration-period median of a 1-min std, expanding then frozen (causal)."""
    n = len(sd)
    cal_end = int(P["cal_hours"] * 60)
    out = np.full(n, np.nan)
    vals = []
    cur = np.nan
    for j in range(n):
        out[j] = cur
        if j < cal_end and steady[j] and not np.isnan(sd[j]):
            vals.append(sd[j])
            if len(vals) >= 30 and (len(vals) % 15 == 0):
                cur = float(np.median(vals))
    return out


def _step_or_ramp(zf: np.ndarray, j: int, P: dict, steady: np.ndarray | None = None) -> str:
    """Calibration step (abrupt) vs drift (gradual): running time (steady minutes only; stops are skipped) the 5-min
    smoothed residual needed to climb from normal (|z| < 1.5) to 70 % of its current level."""
    if j < 5:
        return "SENSOR_CALIBRATION_SHIFT"
    now = zf[j]
    if abs(now) < 1e-6:
        return "SENSOR_DRIFT"
    lo = max(0, j - 36 * 60)
    i = j
    ok = (lambda x: True) if steady is None else (lambda x: bool(steady[x]))
    while i > lo and (not ok(i) or abs(zf[i]) > 0.7 * abs(now)):
        i -= 1
    rise = 0
    while i > lo and (not ok(i) or abs(zf[i]) > 1.5):
        if ok(i):
            rise += 1
        i -= 1
    return "SENSOR_CALIBRATION_SHIFT" if rise <= P["step_rise_min"] else "SENSOR_DRIFT"


def _hyd_diagnose(z: dict, ctrl: str, f: pd.DataFrame, j: int) -> str:
    g = lambda k: z.get(k, 0.0)
    k3 = 3.0
    ps_low = g("ps") < -k3
    vib = g("a") > k3 or g("v") > k3
    hs = f["ctx_hs"].values[j]
    lvl = f["ctx_lvl"].values[j]
    weather = (not np.isnan(hs)) and hs > 2.0
    low_level = (not np.isnan(lvl)) and lvl < 1.6
    head_low = g("hr") < -k3 or g("pd") < -k3
    if ps_low and vib and head_low:
        return "CAVITATION"
    if (weather or low_level) and head_low and not ps_low:
        return "AIR_ENTRAINMENT"
    if vib and head_low and not ps_low and g("a") > k3:
        return "AIR_ENTRAINMENT"
    if ps_low:
        return "SUCTION_STRAINER_CLOGGING"
    if ctrl == "VFD_TEMPERATURE":
        if g("rx") > k3 or g("n") > k3:
            if g("pd") > k3:
                return "DISCHARGE_RESTRICTION"
            if g("hr") < -k3:
                return "IMPELLER_WEAR"
            return "SUCTION_STRAINER_CLOGGING"
        if g("q") < -k3 or g("pw") < -k3:
            return "DISCHARGE_RESTRICTION"
    elif ctrl == "VFD_PRESSURE":
        if g("q") < -k3 or g("n") < -k3 or g("pw") < -k3:
            return "DISCHARGE_RESTRICTION"
        if g("n") > k3 or g("pw") > k3:
            return "IMPELLER_WEAR" if g("hr") < -k3 else "SUCTION_STRAINER_CLOGGING"
    elif ctrl == "FLOW_VALVE":
        if g("vp") > k3 and g("pd") > k3:
            return "DISCHARGE_RESTRICTION"
        if g("vp") > k3 or g("q") < -k3:
            return "SUCTION_STRAINER_CLOGGING"
        if g("pd") > k3:
            return "DISCHARGE_RESTRICTION"
        if g("pd") < -k3:
            return "IMPELLER_WEAR"
    else:  # FIXED_SPEED
        if g("pd") > k3 and (g("i") < 0 or g("q") < 0 or g("pw") < 0):
            return "DISCHARGE_RESTRICTION"
        if g("q") < -k3 and g("pd") >= 0:
            return "DISCHARGE_RESTRICTION"
        if g("pd") < -k3 and (g("i") < 0 or g("pw") < 0):
            return "IMPELLER_WEAR"
    if g("hr") < -k3:
        return "IMPELLER_WEAR"
    return "UNKNOWN"


def _evidence(f, j, keys, zz, crest_s, ES=None) -> str:
    parts = []
    for k in sorted([k for k in keys if k in zz], key=lambda k: -abs(zz[k])):
        if abs(zz[k]) >= 2.0:
            e = ES[k][j] if ES is not None and k in ES else (f["e_" + k].values[j] if "e_" + k in f else np.nan)
            unit = "" if k in ("rx", "hr") else ""
            if k in ("rx", "hr"):
                parts.append(f"{NAME[k]} {100 * e:+.1f}% (z={zz[k]:+.1f})")
            else:
                parts.append(f"{NAME[k]} {e:+.2f} vs expected (z={zz[k]:+.1f})")
    if not np.isnan(crest_s[j]) and crest_s[j] > 3.5:
        parts.append(f"crest factor {crest_s[j]:.2f} (healthy ~3.2)")
    ctx = f"demand {f['ctx_dem'].values[j]:.0f} %, supply water {f['ctx_tsup'].values[j]:.1f} degC"
    return ("; ".join(parts[:5]) or "sustained deviation from the context model") + f". Context: {ctx}."


def _finalize(alerts, f, a):
    """Sort, enforce >= 6 min spacing on the asset (no two types in one scoring burst), format times."""
    alerts.sort(key=lambda x: x[0])
    out, last_j = [], -10 ** 9
    for j, r in alerts:
        jj = max(j, last_j + 6)
        if jj >= len(f):
            continue
        r["alert_time_utc"] = (f["t"].iloc[jj] + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append(r)
        last_j = jj
    return out


# --------------------------------------------------------------------------------------------------------- driver
def prepare_asset(inputs: dict, a: pd.Series, P: dict) -> tuple[pd.DataFrame, list]:
    feat = inputs["feat"][inputs["feat"].asset_id == a.asset_id]
    ctx = inputs["ctx"][inputs["ctx"].asset_id == a.asset_id]
    f = build_frame(feat, ctx, a)
    f.attrs["profile"] = a.profile
    mode = f["operating_mode"].values
    f["run"] = (mode == "RUNNING") & (f["running_fraction"].values >= 0.99) & (f["rows"].values >= 30)
    f["since_start"] = _runs(f["run"].values)
    f["steady"] = f["since_start"].values >= P["steady_min"]
    f["since_lockout"] = _since(mode == "LOCKED_OUT")
    f = physics_columns(f, a)
    physics_signatures(f, a)
    f, channels = residual_z(f, a, P)
    return f, channels


def run_b4(inputs: dict, P: dict | None = None, return_frames: bool = False):
    P = {**PARAMS, **(P or {})}
    rows, frames = [], {}
    for _, a in inputs["assets"].iterrows():
        f, channels = prepare_asset(inputs, a, P)
        rows += detect_asset(f, a, channels, P, inputs.get("snaps"))
        if return_frames:
            frames[a.asset_id] = (f, channels)
    cols = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action",
            "detector"]
    pred = pd.DataFrame(rows, columns=cols)
    return (pred, frames) if return_frames else pred


# --------------------------------------------------------------------------------------------- vibration snapshots
def snapshot_features(snaps: pd.DataFrame, a: pd.Series) -> pd.DataFrame:
    """Per snapshot: 1x / 2x / 3x running-speed velocity lines [mm/s RMS] and the bearing-defect envelope peak
    (BPFO / BPFI x running speed in the 1.8-4.2 kHz envelope spectrum, relative to the envelope noise floor)."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    cols = ["t", "rpm", "v1", "v2", "v3", "env_bpfo", "env_bpfi", "hf"]
    if snaps is None or not len(snaps):
        return pd.DataFrame(columns=cols)
    s = snaps[snaps.asset_id == a.asset_id].sort_values("timestamp_utc")
    out = []
    for _, r in s.iterrows():
        x = np.asarray(r.samples, dtype=float)
        fs, rpm = float(r.sample_rate_hz), float(r.rpm_at_capture)
        if rpm < 120 or len(x) < 1024:
            continue
        n = len(x)
        fr = rpm / 60.0
        w = np.hanning(n)
        fq = np.fft.rfftfreq(n, 1.0 / fs)
        amp = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()
        vel = amp * G / (2 * np.pi * np.maximum(fq, 1.0)) * 1000.0 / np.sqrt(2.0)

        def pk(f0, arr):
            m = (fq > 0.97 * f0) & (fq < 1.03 * f0)
            return float(arr[m].max()) if m.any() else np.nan
        hi = min(4200.0, 0.45 * fs)
        sos = butter(4, [1800.0, hi], btype="band", fs=fs, output="sos")
        xb = sosfiltfilt(sos, x)
        env = np.abs(hilbert(xb))
        env = env - env.mean()
        E = np.abs(np.fft.rfft(env * w)) * 2.0 / w.sum()
        floor = float(np.median(E[(fq > 5) & (fq < 500)])) + 1e-12
        bo, bi = a.bearing_bpfo_order * fr, a.bearing_bpfi_order * fr
        out.append({"t": pd.Timestamp(r.timestamp_utc), "rpm": rpm, "v1": pk(fr, vel), "v2": pk(2 * fr, vel),
                    "v3": pk(3 * fr, vel), "env_bpfo": max(pk(bo, E), pk(2 * bo, E)) / floor,
                    "env_bpfi": pk(bi, E) / floor, "hf": float(np.sqrt(np.mean(xb ** 2)))})
    return pd.DataFrame(out, columns=cols)
