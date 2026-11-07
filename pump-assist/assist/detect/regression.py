"""D02: regression expected value and EWMA.

For each channel, a ridge regression predicts the healthy value from speed, duty demand, fluid temperature and
sea/supply temperature (so the expected value follows the operating conditions, not only speed bands as in D01).
It is fitted on the pump's own commissioning baseline (first 24 h of steady running, like D01), so it also works for
pumps that have no healthy training data. The residual is scaled by its robust spread in the baseline, then smoothed
with an EWMA, which finds slow changes earlier than a per-minute |z| limit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import BASELINE_H, CH  # noqa: E402

from ..splits import DATA_FULL

# Regressors (condition variables). A target channel is never regressed on itself.
CONDITIONS = {"speed_ratio": None, "duty_demand_pct": None, "fluid_temp_c__mean": "tf",
              "sea_temp_c": None, "ambient_temp_c__mean": None}
FLOOR = {"v": 0.05, "a": 0.01, "c": 0.05, "bt": 0.3, "ps": 1.0, "pd": 2.0, "i": 0.3, "pw": 0.2, "q": 2.0,
         "n": 5.0, "wt": 0.5, "tf": 0.3, "vp": 0.5}
SKIP_TARGETS = {"n", "tf"}      # conditions themselves
EWMA_SPAN = 30
RIDGE = 1.0


def load_context(asset_ids, data: Path | str = DATA_FULL) -> pd.DataFrame:
    c = pd.read_csv(Path(data) / "operating_context.csv", usecols=["asset_id", "timestamp_utc", "duty_demand_pct", "sea_temp_c"])
    c = c[c.asset_id.isin(list(asset_ids))].copy()
    c["window_start_utc"] = pd.to_datetime(c.timestamp_utc, utc=True)
    return c.drop(columns="timestamp_utc")


def ewma_scores(zf: pd.DataFrame, context: pd.DataFrame, span: int = EWMA_SPAN, ridge: float = RIDGE) -> pd.DataFrame:
    """Per pump-minute: score = max over channels of |EWMA of scaled residual|. NaN outside steady running."""
    out = []
    ctx = context.set_index(["asset_id", "window_start_utc"])
    for aid, g in zf.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc").reset_index(drop=True)
        c = ctx.loc[aid].reindex(g["window_start_utc"])
        X = pd.DataFrame({
            "speed_ratio": g["speed_ratio"].to_numpy(),
            "duty": c["duty_demand_pct"].to_numpy() / 100.0,
            "tf": g.get("fluid_temp_c__mean", pd.Series(np.nan, index=g.index)).to_numpy(),
            "sea": c["sea_temp_c"].to_numpy(),
            "amb": g.get("ambient_temp_c__mean", pd.Series(np.nan, index=g.index)).to_numpy(),
        })
        X = X.loc[:, X.notna().sum() > 0]
        steady = g["steady"].to_numpy(bool)
        t0 = g["window_start_utc"].min()
        base = steady & (g["window_start_utc"] < t0 + pd.Timedelta(hours=BASELINE_H)).to_numpy()
        best = np.zeros(len(g))
        for key, col in CH.items():
            m = f"{col}__mean"
            if key in SKIP_TARGETS or m not in g or g[m].notna().sum() == 0:
                continue
            Xk = X.drop(columns=[x for x in ("tf",) if key == "tf"], errors="ignore")
            Xv = Xk.to_numpy(dtype=float)
            y = g[m].to_numpy(dtype=float)
            ok_b = base & np.isfinite(y) & np.isfinite(Xv).all(axis=1)
            if ok_b.sum() < 30:
                continue
            mu, sd = Xv[ok_b].mean(axis=0), Xv[ok_b].std(axis=0)
            sd[sd < 1e-9] = 1.0
            Z = (np.nan_to_num(Xv, nan=0.0) - mu) / sd
            A = np.c_[np.ones(len(Z)), Z]
            Ab = A[ok_b]
            reg = ridge * np.eye(A.shape[1]); reg[0, 0] = 0.0
            beta = np.linalg.solve(Ab.T @ Ab + reg, Ab.T @ y[ok_b])
            res = y - A @ beta
            spread = max(1.4826 * np.median(np.abs(res[ok_b] - np.median(res[ok_b]))), FLOOR[key])
            r = pd.Series(np.where(steady & np.isfinite(y) & np.isfinite(Xv).all(axis=1), res / spread, np.nan))
            ew = r.dropna().ewm(span=span, adjust=False).mean().reindex(r.index)
            best = np.fmax(best, np.abs(ew.to_numpy()))   # fmax ignores NaN
        out.append(pd.DataFrame({"asset_id": aid, "window_start_utc": g["window_start_utc"],
                                 "d02_score": np.where(steady, best, np.nan)}))
    return pd.concat(out, ignore_index=True)
