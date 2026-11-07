"""T01: guarded baseline update.

D01 scales every channel against a fixed commissioning baseline (first 24 h of steady running, by speed regime).
A machine that ages slowly drifts away from it and raises false alerts. Three variants of the expected value:
  static   : never updated (D01)
  always   : EWMA update on every steady minute (also learns the fault)
  guarded  : EWMA update only while no deviation is active and every channel has trust >= 0.5. The update stops when a
             deviation starts (|z| above `stop_z` on any channel) and restarts `hold_min` minutes after it ends.
The spread (scale) stays fixed. The update uses only the minute itself after its z-score is computed (causal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import BASELINE_H, CH  # noqa: E402

FLOOR = {"v": 0.05, "a": 0.01, "c": 0.05, "bt": 0.3, "ps": 1.0, "pd": 2.0, "i": 0.3, "pw": 0.2, "q": 2.0,
         "n": 5.0, "wt": 0.5, "tf": 0.3, "vp": 0.5}
TAU_MIN = 1440.0       # EWMA time constant: one day of steady minutes
STOP_Z = 4.0           # a deviation is "active" above this |z|: half the alert limit (8)
HOLD_MIN = 120         # the update restarts this long after the deviation ended
TRUST_MIN = 0.5


def adaptive_z(zf: pd.DataFrame, mode: str = "guarded", tau_min: float = TAU_MIN, stop_z: float = STOP_Z,
               hold_min: int = HOLD_MIN, trust: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recompute the z_<key> columns of a z-frame with an online baseline. mode: 'static' | 'always' | 'guarded'.

    trust: output of ingest.quality.trust_frame(zf) (needed for 'guarded'; computed if omitted).
    """
    if mode == "static":
        return zf
    if mode not in ("always", "guarded"):
        raise ValueError("mode must be static, always or guarded")
    if mode == "guarded" and trust is None:
        from ..ingest.quality import trust_frame
        trust = trust_frame(zf)
    alpha = 1.0 - np.exp(-1.0 / tau_min)
    out = zf.copy()
    keys = [k for k in CH if f"{CH[k]}__mean" in zf and f"z_{k}" in zf]
    for k in keys:
        out[f"z_{k}"] = out[f"z_{k}"].astype("float64")
    for aid, g in zf.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc")
        idx = g.index.to_numpy()
        t0 = g["window_start_utc"].min()
        base = g[(g["window_start_utc"] < t0 + pd.Timedelta(hours=BASELINE_H)) & g["steady"]]
        regimes = g["regime"].to_numpy()
        steady = g["steady"].to_numpy(bool)
        X = np.stack([g[f"{CH[k]}__mean"].to_numpy(float) for k in keys], axis=1)
        T = np.stack([trust.loc[idx, f"trust_{k}"].to_numpy(float) for k in keys], axis=1) if mode == "guarded" else None
        mu0, sd, gmu = [], [], []
        for k in keys:
            m = f"{CH[k]}__mean"
            med = base.groupby("regime")[m].median()
            mad = base.groupby("regime")[m].apply(lambda s: (s - s.median()).abs().median())
            gm, gmad = base[m].median(), (base[m] - base[m].median()).abs().median()
            mu0.append(med.to_dict()); gmu.append(gm)
            sd.append({"mad": mad.to_dict(), "gmad": gmad, "floor": FLOOR[k]})
        state = [dict(d) for d in mu0]          # regime -> current expected value
        Z = np.full(X.shape, np.nan)
        since_dev = hold_min + 1
        for i in range(len(g)):
            r = regimes[i]
            dev_now = False
            for j, k in enumerate(keys):
                x = X[i, j]
                if not np.isfinite(x):
                    continue
                mu = state[j].get(r, gmu[j])
                if not np.isfinite(mu):
                    continue
                s = max(1.4826 * sd[j]["mad"].get(r, sd[j]["gmad"]), sd[j]["floor"])
                z = (x - mu) / s
                Z[i, j] = z
                if abs(z) > stop_z:
                    dev_now = True
            since_dev = 0 if dev_now else since_dev + 1
            if not steady[i]:
                continue
            if mode == "guarded":
                if dev_now or since_dev <= hold_min or (T[i] < TRUST_MIN).any():
                    continue
            for j in range(len(keys)):
                x = X[i, j]
                if np.isfinite(x) and np.isfinite(Z[i, j]):
                    mu = state[j].get(r, gmu[j])
                    state[j][r] = mu + alpha * (x - mu)
        for j, k in enumerate(keys):
            out.loc[idx, f"z_{k}"] = Z[:, j]
    return out
