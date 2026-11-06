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
    "stuck_min": 20, "stuck_sparse_samples": 9,
    "noise_ratio": 3.0, "noise_persist": 8,
    "crest_abs": 3.75,          # fleet prior: healthy crest factor ~3.2 (bearing damage / cavitation raise it)
    "rx_env": (0.0, 0.025),     # plausible calibration bias of measured / physics speed ratio - 1
    "hr_env": (-0.03, 0.03),
    "single_ratio": 0.25,
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
    sp = f.get("motor_speed_rpm__mean")
    r_meas = (sp.values / a.motor_rated_rpm) if sp is not None and sp.notna().any() else np.where(f["run"].values, 1.0, 0.0)
    f["r_meas"] = r_meas
    f["rx"] = r_meas / r_exp - 1.0 if ctrl == "VFD_TEMPERATURE" else np.nan
    # pump-curve head ratio where suction, discharge and flow are measured (speed from the drive or rated)
    hr = np.full(len(f), np.nan)
    if all(f.get(CH[k] + "__mean") is not None and f[CH[k] + "__mean"].notna().any() for k in ("ps", "pd", "q")):
        rho = 1000.0
        H_meas = (f[CH["pd"] + "__mean"].values - f[CH["ps"] + "__mean"].values) * 1000.0 / (rho * G)
        Q = f[CH["q"] + "__mean"].values / 3600.0
        r = np.where(np.isnan(r_meas), 1.0, r_meas)
        H_c = Hb * (h0 * r ** 2 - (h0 - 1.0) * (Q / Qb) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = np.where(H_c > 0.2 * Hb, H_meas / H_c - 1.0, np.nan)
    f["hr"] = hr
    return f


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
    has_lvl = np.isfinite(lvl).mean() > 0.9
    if k in ("ps", "pd") and has_lvl:
        base = [ones] if ctrl == "FIXED_SPEED" else [ones, u, u ** 2, u ** 3]
        return np.column_stack(base + [np.nan_to_num(lvl, nan=np.nanmedian(lvl))])
    if k == "tf":
        return np.column_stack([ones, ewm(f["ctx_tsup"].values, 3), u])
    if k in VIB:
        r = np.nan_to_num(f["r_meas"].values, nan=1.0)
        return np.column_stack([ones, r, r ** 2]) if ctrl != "FIXED_SPEED" and ctrl != "FLOW_VALVE" else \
            np.column_stack([ones, u] if ctrl == "FLOW_VALVE" else [ones])
    if ctrl == "FIXED_SPEED":
        return np.column_stack([ones])
    return np.column_stack([ones, u, u ** 2, u ** 3])


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
    avail = []
    for k, col in CH.items():
        m = col + "__mean"
        if m in f and f[m].notna().sum() > 30:
            avail.append(k)
    channels = [k for k in avail if k not in ("n",) or ctrl != "VFD_TEMPERATURE"] + \
               (["rx"] if ctrl == "VFD_TEMPERATURE" else []) + (["hr"] if f["hr"].notna().sum() > 30 else [])
    if "n" in avail and ctrl == "VFD_TEMPERATURE":
        channels.append("n")
    for k in channels:
        if k in ("rx", "hr"):
            x = f[k].values.astype(float)
            sparse = False
        else:
            x = f[CH[k] + "__mean"].values.astype(float)
            sparse = f[CH[k] + "__count"].fillna(0).median() < 30 if CH[k] + "__count" in f else False
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
        f["x_" + k] = x
    return f, channels


# ----------------------------------------------------------------------------------------------------- detectors
def _cfg_alert(a, t, cls, ftype, conf, evidence, action):
    return {"asset_id": a.asset_id, "alert_time_utc": t, "event_class": cls, "fault_type": ftype,
            "confidence": float(np.clip(conf, 0.05, 0.99)), "evidence": evidence, "recommended_action": action}


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

    def __init__(self, name, P, close_after=None, escalate_only=True):
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
            if cand is not None and cand not in self.type_alerted and not (cand == "UNKNOWN" and self.any_alert):
                emit = cand
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
            alerts.append((j, _cfg_alert(a, None, "SENSOR", "COMMUNICATION_LOSS", 0.9,
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
        if col + "__mean" not in f or f[col + "__mean"].notna().sum() < 30:
            continue
        cnt = f[col + "__count"].fillna(0).values
        sparse = np.median(cnt[run]) < 30 if run.any() else False
        mean = f[col + "__mean"].values
        # stuck: identical consecutive values while running
        if sparse:
            stuck_flag = np.zeros(n, dtype=bool)
            last, same, last_j = None, 0, -1
            for j in range(n):
                v = mean[j]
                if not np.isnan(v):
                    same = same + 1 if (last is not None and v == last) else 1
                    last, last_j = v, j
                stuck_flag[j] = run[j] and same >= P["stuck_sparse_samples"] and j - last_j <= 10
        else:
            sd = f[col + "__std"].values
            mx, mn = f[col + "__max"].values, f[col + "__min"].values
            flat = (sd == 0) | (mx == mn)
            same_as_prev = np.r_[False, mean[1:] == mean[:-1]]
            stuck_flag = _persist(run & flat & (same_as_prev | flat) & (cnt >= 30), P["stuck_min"])
        trk = Track("stuck_" + k, P, P["close_sensor_min"])
        for j in range(n):
            ft = trk.step(j, bool(stuck_flag[j]), "SENSOR_STUCK", bool(run[j]))
            sensor_open[k][j] |= trk.open
            if ft:
                alerts.append((j, _cfg_alert(a, None, "SENSOR", "SENSOR_STUCK", 0.85,
                                             f"{NAME[k]} ({col}) frozen at {mean[j]:.2f} while the pump runs and coupled "
                                             f"channels vary.", ACTIONS["SENSOR_STUCK"])))
        # noise / spikes (dense channels only): 1-min std far above its calibration level, coupled channels calm
        if not sparse and col + "__std" in f:
            sd = f[col + "__std"].values.astype(float)
            ref = _causal_ref(sd, steady, P)
            ratio = sd / np.maximum(ref, 1e-6)
            cpl = [c for c in COUPLED.get(k, []) if c in CH and CH[c] + "__std" in f and f[CH[c] + "__std"].notna().sum() > 30]
            cr = np.zeros(n)
            for c in cpl:
                sdc = f[CH[c] + "__std"].values.astype(float)
                cnt_c = f[CH[c] + "__count"].fillna(0).values
                if np.median(cnt_c[run]) < 30 if run.any() else True:
                    continue
                rc = sdc / np.maximum(_causal_ref(sdc, steady, P), 1e-6)
                cr = np.maximum(cr, np.nan_to_num(rc))
            rr = ewm(np.nan_to_num(ratio, nan=1.0), 5)
            heavy = np.nan_to_num(f["ctx_hs"].values) > 2.0
            nflag = _persist(run & (rr > P["noise_ratio"]) & (cr < 0.5 * rr) & ~(heavy & (k in ("ps", "pd"))),
                             P["noise_persist"])
            trk = Track("noise_" + k, P, P["close_sensor_min"])
            for j in range(n):
                ft = trk.step(j, bool(nflag[j]), "SENSOR_NOISE_SPIKES", bool(run[j]))
                sensor_open[k][j] |= trk.open
                if ft:
                    alerts.append((j, _cfg_alert(a, None, "SENSOR", "SENSOR_NOISE_SPIKES", 0.8,
                                                 f"{NAME[k]} ({col}) 1-min scatter {rr[j]:.1f}x its normal level; coupled "
                                                 f"channels steady.", ACTIONS["SENSOR_NOISE_SPIKES"])))

    # ---------------- single-channel drift / calibration shift
    for k in [c for c in SENSOR_CH if c in channels]:
        cpl = [c for c in COUPLED.get(k, []) if c in channels]
        if not cpl:
            continue
        cmax = np.max(np.vstack([np.abs(np.nan_to_num(zmag[c])) for c in cpl]), axis=0)
        single = act[k] & (cmax < P["single_ratio"] * np.abs(zmag[k])) & (cmax < 2.5)
        trk = Track("single_" + k, P, P["close_sensor_min"])
        for j in range(n):
            ftype = _step_or_ramp(Z[k], j) if single[j] else None
            ft = trk.step(j, bool(single[j]), ftype, bool(steady[j]))
            sensor_open[k][j] |= trk.open
            if ft:
                alerts.append((j, _cfg_alert(a, None, "SENSOR", ft, 0.7,
                                             f"{NAME[k]} ({CH[k]}) deviates alone: {f['e_' + k].values[j]:+.2f} from the "
                                             f"context-expected value (z={zmag[k][j]:+.1f}); physically coupled channels "
                                             f"({', '.join(cpl[:4])}) are normal.", ACTIONS[ft])))

    # ---------------- machine detectors
    def zget(k, j, src=None):
        src = src or zmag
        if k not in channels:
            return 0.0
        if sensor_open.get(k) is not None and sensor_open[k][j]:
            return 0.0
        v = src[k][j]
        return 0.0 if np.isnan(v) else float(v)

    crest = f[CH["c"] + "__mean"].values if CH["c"] + "__mean" in f else np.full(n, np.nan)
    crest_s = ewm_seg(np.where(steady, crest, np.nan), 30, reset) if not np.all(np.isnan(crest)) else np.full(n, np.nan)
    since_lock = f["since_lockout"].values
    hyd_ch = [c for c in HYD if c in channels]
    vib_ch = [c for c in VIB if c in channels]

    cm = P["close_machine_min"]
    trk_b = Track("bearing", P, cm)
    trk_v = Track("velocity", P, cm)
    trk_h = Track("hydraulic", P, cm)
    for j in range(n):
        if not steady[j]:
            trk_b.step(j, False, None, False)
            trk_v.step(j, False, None, False)
            trk_h.step(j, False, None, False)
            continue
        zz = {k: zget(k, j) for k in channels}
        active = {k for k in channels if act[k][j] and not sensor_open[k][j] if k in sensor_open} | \
                 {k for k in channels if act[k][j] and k not in sensor_open}
        hyd_active = [k for k in hyd_ch if k in active]
        # bearing: acceleration / crest factor (fleet crest prior catches damage present at window start)
        crest_hi = (not np.isnan(crest_s[j])) and crest_s[j] > P["crest_abs"]
        brg = (("a" in active and zz.get("a", 0) > 0) or ("c" in active and zz.get("c", 0) > 0) or crest_hi)
        cav_like = (zz.get("ps", 0) < -P["k_slow"] or len(hyd_active) >= 2) and zz.get("a", 0) > 0
        b_type = None
        if brg and not cav_like:
            b_type = "BEARING_WEAR"
        e = trk_b.step(j, b_type is not None, b_type)
        if e:
            alerts.append((j, _cfg_alert(a, None, "MACHINE", e, 0.75, _evidence(f, j, ["a", "c", "bt", "v"], zz, crest_s, ES),
                                         ACTIONS[e])))
        # velocity family: imbalance / misalignment (bearing late stage is attributed to the bearing track)
        v_up = "v" in active and zz.get("v", 0) > 0
        v_type = None
        if v_up and not cav_like and not trk_b.open:
            mis = (zz.get("bt", 0) > 1.5 or zz.get("wt", 0) > 1.5 or since_lock[j] < 12 * 60
                   or _step_or_ramp(Z["v"], j) == "SENSOR_CALIBRATION_SHIFT")
            v_type = "MISALIGNMENT" if mis else "IMBALANCE"
        e = trk_v.step(j, v_type is not None, v_type)
        if e:
            alerts.append((j, _cfg_alert(a, None, "MACHINE", e, 0.7, _evidence(f, j, ["v", "bt", "wt", "a"], zz, crest_s, ES),
                                         ACTIONS[e])))
        # hydraulic family
        h_type = None
        if hyd_active:
            h_type = _hyd_diagnose(zz, a.control, f, j)
        e = trk_h.step(j, h_type is not None, h_type)
        if e:
            alerts.append((j, _cfg_alert(a, None, "MACHINE", e, 0.7, _evidence(f, j, hyd_ch + ["a", "v"], zz, crest_s, ES),
                                         ACTIONS[e])))
    return _finalize(alerts, f, a)


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


def _step_or_ramp(z: np.ndarray, j: int) -> str:
    """Calibration step (abrupt) vs drift (gradual) from the residual history up to j."""
    if j < 30 or np.isnan(z[j]):
        return "SENSOR_DRIFT"
    now = np.nanmean(z[max(0, j - 4):j + 1])
    past = z[max(0, j - 240):max(1, j - 20)]
    if np.all(np.isnan(past)):
        return "SENSOR_CALIBRATION_SHIFT"
    # fraction of the current deviation already present 20-30 min ago
    recent = np.nanmean(z[max(0, j - 30):max(1, j - 20)]) if np.any(~np.isnan(z[max(0, j - 30):max(1, j - 20)])) else 0.0
    older = np.nanmean(z[max(0, j - 240):max(1, j - 180)]) if np.any(~np.isnan(z[max(0, j - 240):max(1, j - 180)])) else 0.0
    if abs(now) < 1e-6:
        return "SENSOR_DRIFT"
    if abs(older / now) < 0.3 and abs(recent / now) > 0.7:
        return "SENSOR_CALIBRATION_SHIFT"      # was normal a few hours ago and already at today's level 20 min ago
    if abs(recent / now) < 0.4:
        return "SENSOR_CALIBRATION_SHIFT"      # jumped within the last 20 min
    return "SENSOR_DRIFT"


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
    mode = f["operating_mode"].values
    f["run"] = (mode == "RUNNING") & (f["running_fraction"].values >= 0.99) & (f["rows"].values >= 30)
    f["since_start"] = _runs(f["run"].values)
    f["steady"] = f["since_start"].values >= P["steady_min"]
    f["since_lockout"] = _since(mode == "LOCKED_OUT")
    f = physics_columns(f, a)
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
    cols = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"]
    pred = pd.DataFrame(rows, columns=cols)
    return (pred, frames) if return_frames else pred
