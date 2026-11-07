"""D06: benign periods. Normal events look like faults: start-ups, load changes, heavy weather.

Marks pump-minutes where the detector should hold its alert. Uses only operating_context.csv (schedule state,
duty demand, sea state), which is plant information available at the time, never fault labels or benign_windows files.
Rules: a transient schedule state (manoeuvring, start-up, shutdown, set-point change, shift change, planned stop,
changeover), the end of a maintenance lockout, sea state Hs above a limit, or a fast duty change. Each stays marked for
`tail_min` more minutes after it ends (the plant needs time to settle).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..splits import DATA_FULL

TRANSIENT_STATES = {"arrival manoeuvring", "departure manoeuvring", "start-up", "shutdown", "set-point change",
                    "shift start", "shift end", "planned stop", "duty/standby changeover"}
RESTART_FROM = "maintenance (locked out)"


def _tail(mask: np.ndarray, tail: int) -> np.ndarray:
    if tail <= 0:
        return mask
    c = np.cumsum(mask.astype(int))
    prev = np.concatenate([np.zeros(tail, int), c[:-tail]]) if tail < len(c) else np.zeros(len(c), int)
    return (c - prev) > 0


def benign_minutes(asset_ids, data: Path | str = DATA_FULL, tail_min: int = 30, hs_limit: float = 2.2,
                   duty_step: float = 15.0, duty_lag: int = 10) -> pd.DataFrame:
    """Columns asset_id, window_start_utc, benign (bool), reason."""
    ctx = pd.read_csv(Path(data) / "operating_context.csv", usecols=["asset_id", "timestamp_utc", "schedule_state",
                                                                      "duty_demand_pct", "sea_state_hs_m"])
    ctx = ctx[ctx.asset_id.isin(list(asset_ids))]
    ctx["window_start_utc"] = pd.to_datetime(ctx.timestamp_utc, utc=True)
    out = []
    for aid, g in ctx.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc").reset_index(drop=True)
        st = g.schedule_state
        transient = st.isin(TRANSIENT_STATES).to_numpy()
        restart = ((st.shift(1) == RESTART_FROM) & (st != RESTART_FROM)).to_numpy()
        weather = (g.sea_state_hs_m > hs_limit).fillna(False).to_numpy()
        duty = g.duty_demand_pct.rolling(5, min_periods=1).mean()
        fast = ((duty - duty.shift(duty_lag)).abs() > duty_step).fillna(False).to_numpy()
        reasons = {"transient state": transient, "maintenance restart": restart, "heavy weather": weather,
                   "fast duty change": fast}
        b = np.zeros(len(g), bool)
        why = np.full(len(g), "", dtype=object)
        for name, m in reasons.items():
            m = _tail(m, tail_min)
            why[m & ~b] = name
            b |= m
        out.append(pd.DataFrame({"asset_id": aid, "window_start_utc": g.window_start_utc, "benign": b, "reason": why}))
    return pd.concat(out, ignore_index=True)


def benign_lookup(benign: pd.DataFrame | None, asset_id: str, times: pd.Series) -> np.ndarray:
    """Boolean array aligned to `times` for one pump (False where unknown)."""
    if benign is None:
        return np.zeros(len(times), bool)
    b = benign[benign.asset_id == asset_id].set_index("window_start_utc")["benign"]
    return b.reindex(times).fillna(False).to_numpy(bool)
