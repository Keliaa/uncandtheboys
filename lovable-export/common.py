"""Shared rules for the Lovable cockpit export: paths, the B4 run, deviation, open-alert levels, JSON hygiene.

Everything here is causal: a value for time t is computed from minutes completed by t, B4 alerts published by t,
operating modes up to t and plant alarms raised by t. validate.py recomputes sampled snapshots from truncated inputs
with these same functions.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = REPO / "data" / "full"
B4_DIR = REPO / "physics-context-detector"
DEMO_DIR = REPO / "operator-demo"
STRESS_ROOT = B4_DIR / "build"
FROZEN_PARAMS = B4_DIR / "results" / "b4_frozen_params.json"
FROZEN_TRAINVAL = B4_DIR / "results" / "b4_trainval_predictions.csv"
FROZEN_TEST = B4_DIR / "results" / "b4_test_predictions.csv"
B1_TRAINVAL = REPO / "baseline-reproduction" / "results" / "b1_trainval_predictions.csv"
TIER_ABLATION = B4_DIR / "results" / "tier_ablation.csv"

sys.path.insert(0, str(DEMO_DIR))
import prepare_demo as pdm  # noqa: E402  (operator-demo bundle builder, reused unchanged)

SNAPSHOT_MIN = 30
WEEK_MIN = 7 * 24 * 60
OPEN_HOLD_MIN = 12 * 60          # presentation rule: an alert group stays open 12 h after its latest alert
MIN_CAL_POINTS = 10              # same constants as the cockpit (src/data/deviation.ts)
SIGMA_FLOOR = 1e-3
MAX_STALE_MIN = 30
LEVEL_RANK = {"WARNING": 3, "CAUTION": 2, "ADVISORY": 1}
STATUSES = ("active", "calibrating", "not_running", "settling", "unavailable")
ALERT_KEYS = ("fault_type", "event_class", "evidence", "recommended_action", "alert_time_utc")
EM_DASH = "—"


# ------------------------------------------------------------------------------------------------- helpers
def utc(x) -> pd.Timestamp:
    return pdm.utc(x)


def iso(t) -> str:
    return pdm.iso(t)


def git_head(repo: Path = REPO) -> tuple[str, str]:
    """(short sha, commit time in UTC ISO). Used instead of wall-clock time so two runs are byte-identical."""
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    ts = subprocess.run(["git", "-C", str(repo), "show", "-s", "--format=%cI", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
    return sha, iso(ts)


def clean(o, nd: int = 3):
    """Round floats to nd decimals, NaN/Inf -> None, numpy scalars -> Python, recursively."""
    if isinstance(o, dict):
        return {k: clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, nd) for v in o]
    if isinstance(o, str):
        return no_em_dash(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if not math.isfinite(x):
            return None
        r = round(x, nd)
        return 0.0 if r == 0 else r
    return o


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def no_em_dash(s):
    return s.replace(EM_DASH, "-") if isinstance(s, str) else s


# ------------------------------------------------------------------------------------------------- B4 run
_B4 = None


def b4_module():
    global _B4
    if _B4 is None:
        _B4 = pdm.import_b4()
    return _B4


def frozen_params() -> dict:
    return json.loads(FROZEN_PARAMS.read_text())


def run_b4(inp: dict, params: dict):
    """Run the unchanged B4 with frozen parameters on already loaded inputs of exactly one pump."""
    b4 = b4_module()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        pred, frames = b4.run_b4(inp, params, return_frames=True)
    (aid,) = list(frames)
    f, _ = frames[aid]
    return pred, f


def truncate_inputs(inp: dict, cut: pd.Timestamp) -> dict:
    """Inputs as known at `cut` (same rule as physics-context-detector/tests/test_b4_causality.py)."""
    out = dict(inp)
    f = inp["feat"]
    out["feat"] = f[pd.to_datetime(f["window_start_utc"], utc=True) + pd.Timedelta(minutes=1) <= cut]
    c = inp["ctx"]
    out["ctx"] = c[pd.to_datetime(c["timestamp_utc"], utc=True) <= cut]
    if inp.get("snaps") is not None:
        s = inp["snaps"]
        out["snaps"] = s[pd.to_datetime(s["timestamp_utc"], utc=True) <= cut]
    return out


def truncate_alarms(alarms: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    a = alarms[pd.to_datetime(alarms.raised_utc, utc=True) <= cut].copy()
    cl = pd.to_datetime(a.cleared_utc, utc=True)
    a.loc[(cl > cut).values, "cleared_utc"] = None
    return a


# ------------------------------------------------------------------------------------- per-pump state frame
class PumpState:
    """Minute-level state of one pump needed for snapshots: offsets, monitoring status, residuals, modes."""

    def __init__(self, a: pd.Series, f: pd.DataFrame, pred: pd.DataFrame, alarms: pd.DataFrame, params: dict):
        b4 = b4_module()
        P = {**b4.PARAMS, **params}
        self.asset = a
        self.ws = utc(a.window_start_utc)
        self.cal_min = int(P["cal_hours"] * 60)
        mon = pdm.monitoring_frame(b4, f, a, P)
        self.status = mon.status.values.astype(str)
        self.keys = list(mon.attrs["keys"])
        t = pd.to_datetime(f["t"], utc=True)
        self.off = ((t - self.ws) / pd.Timedelta(minutes=1)).values.astype(float)   # minute start offset
        self.mode = f["operating_mode"].astype(str).values
        _, sparse = b4.static_channels(a)
        self.resid = {}
        for k in self.keys:
            meas, exp = pdm.display_series(b4, f, k, sparse)
            self.resid[k] = (meas - exp).astype(float)
        self.sigma = calibration_sigmas(self.resid, self.off, self.cal_min)
        self.alerts = alert_table(pred, self.ws)
        self.lock_off = self.off[self.mode == "LOCKED_OUT"] + 1     # a minute's mode is known when it completes
        self.alarms = alarms


def calibration_sigmas(resid: dict, off: np.ndarray, cal_min: int) -> dict:
    """Residual std (ddof 1) per channel over the calibration window [0, cal_min) minutes; None if < 10 points.
    Identical to calibrationSigmas() in the cockpit (src/data/deviation.ts)."""
    cal = off < cal_min
    out = {}
    for k, r in resid.items():
        rc = r[cal & np.isfinite(r)]
        out[k] = max(float(np.std(rc, ddof=1)), SIGMA_FLOOR) if len(rc) >= MIN_CAL_POINTS else None
    return out


def deviation_at(resid: dict, sigma: dict, idx: int) -> tuple[float | None, str | None]:
    """Cockpit deviationAt(): max over channels of |measured - expected| / sigma using the latest finite residual
    within MAX_STALE_MIN minutes of minute idx; channels without a calibration sigma fall back to the std of all
    residuals up to idx. Returns (None, None) when nothing is usable."""
    if idx < 0:
        return None, None
    best, best_k = None, None
    for k, r in resid.items():
        i = idx
        while i >= 0 and idx - i <= MAX_STALE_MIN and not np.isfinite(r[i]):
            i -= 1
        if i < 0 or idx - i > MAX_STALE_MIN:
            continue
        s = sigma.get(k)
        if s is None:
            past = r[: idx + 1]
            past = past[np.isfinite(past)]
            if len(past) < 2:
                continue
            s = max(float(np.std(past, ddof=1)), SIGMA_FLOOR)
        z = abs(float(r[i])) / s
        if best is None or z > best:
            best, best_k = z, k
    return best, best_k


def alert_table(pred: pd.DataFrame, ws: pd.Timestamp) -> pd.DataFrame:
    p = pred.reset_index(drop=True).copy()
    p["t_off"] = (pd.to_datetime(p.alert_time_utc, utc=True) - ws) / pd.Timedelta(minutes=1)
    p["group"] = p.asset_id.astype(str) + "|" + p.recommended_action.astype(str)
    return p.sort_values(["t_off"], kind="stable").reset_index(drop=True)


def open_alerts(alerts: pd.DataFrame, lock_off: np.ndarray, t_off: float) -> pd.DataFrame:
    """Presentation rule: a group (pump + recommended_action) is open from an alert's time until 12 h after the
    group's latest alert, or until the pump is LOCKED_OUT, whichever comes first. Returns the latest alert of every
    open group at t_off, using only alerts published by t_off and modes of minutes started by t_off."""
    known = alerts[alerts.t_off <= t_off]
    if known.empty:
        return known
    latest = known.groupby("group", sort=False).tail(1)
    keep = []
    for _, r in latest.iterrows():
        if t_off >= r.t_off + OPEN_HOLD_MIN:
            continue
        if ((lock_off > r.t_off) & (lock_off <= t_off)).any():
            continue
        keep.append(r)
    return pd.DataFrame(keep, columns=alerts.columns)


def alarm_active(alarms: pd.DataFrame, t: pd.Timestamp) -> bool:
    if alarms is None or alarms.empty:
        return False
    r = pd.to_datetime(alarms.raised_utc, utc=True)
    c = pd.to_datetime(alarms.cleared_utc, utc=True)
    return bool(((r <= t) & (c.isna() | (c > t))).any())


def most_severe(open_df: pd.DataFrame):
    if open_df is None or open_df.empty:
        return None
    o = open_df.assign(rank=np.where(open_df.event_class == "MACHINE", 2, 1))
    r = o.sort_values(["rank", "t_off"], ascending=[False, False], kind="stable").iloc[0]
    return {k: no_em_dash(str(r[k])) for k in ALERT_KEYS}


def snapshot(st: PumpState, o: int) -> dict:
    """Fleet snapshot at offset o minutes after monitoring start (causal: minutes complete by o)."""
    idx = int(np.searchsorted(st.off + 1, o, side="right")) - 1
    t = st.ws + pd.Timedelta(minutes=o)
    status = st.status[idx] if idx >= 0 else "unavailable"
    if status not in STATUSES:
        status = "unavailable"
    dev, ch = (None, None) if status == "calibrating" else deviation_at(st.resid, st.sigma, idx)
    op = open_alerts(st.alerts, st.lock_off, o)
    alert = most_severe(op)
    if alarm_active(st.alarms, t):
        level = "WARNING"
    elif alert is not None:
        level = "CAUTION" if alert["event_class"] == "MACHINE" else "ADVISORY"
    else:
        level = None
    return {"offset_min": int(o), "deviation": None if dev is None else round(dev, 3), "deviation_channel": ch,
            "level": level, "status": status, "alert": alert}


def open_intervals(st: PumpState, end_min: float) -> list[tuple[float, float]]:
    """Union of open-alert intervals (minutes since monitoring start) under the same presentation rule."""
    iv = []
    for _, r in st.alerts.iterrows():
        e = min(r.t_off + OPEN_HOLD_MIN, end_min)
        lk = st.lock_off[st.lock_off > r.t_off]
        if len(lk):
            e = min(e, float(lk.min()))
        if e > r.t_off:
            iv.append((float(r.t_off), float(e)))
    iv.sort()
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged
