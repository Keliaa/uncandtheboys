"""D05: operating modes by clustering, instead of speed bands.

D01 scales each channel against the pump's baseline inside an operating mode. Speed bands do not exist on fixed-speed pumps (one mode
for everything). Clusters of duty demand, fluid temperature, sea/supply temperature, valve position and speed (where they exist) work
on every pump. Per pump: standardise the condition variables on the first 24 h of steady running (the same commissioning window as D01),
fit K-means or a Gaussian mixture with 3 to 6 clusters (K-means: best silhouette; GMM: lowest BIC, no labels used), then give every minute
the cluster of its conditions. Only the baseline window is used for fitting, so nothing from the future enters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import BASELINE_H  # noqa: E402

K_RANGE = (3, 4, 5, 6)
MIN_BASELINE = 120
SEED = 0


def _condition_matrix(g: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    c = ctx.reindex(g["window_start_utc"])
    cols = {
        "duty": c["duty_demand_pct"].to_numpy(float),
        "fluid_temp": g["fluid_temp_c__mean"].to_numpy(float) if "fluid_temp_c__mean" in g else np.nan,
        "sea_temp": c["sea_temp_c"].to_numpy(float),
        "valve": g["valve_position_pct__mean"].to_numpy(float) if "valve_position_pct__mean" in g else np.nan,
        "speed": g["speed_ratio"].to_numpy(float) if "speed_ratio" in g else np.nan,
    }
    X = pd.DataFrame(cols)
    return X.loc[:, X.notna().sum() > 0]


def cluster_regimes(f: pd.DataFrame, context: pd.DataFrame, method: str = "kmeans", k_range=K_RANGE, seed: int = SEED) -> pd.Series:
    """f: output of baselines.regime_frame (has asset_id, window_start_utc, steady, speed_ratio, ...). Returns an int Series (index of f).
    method: 'kmeans' or 'gmm'. Minutes whose conditions are unknown get -1."""
    if method not in ("kmeans", "gmm"):
        raise ValueError("method must be kmeans or gmm")
    ctx = context.set_index(["asset_id", "window_start_utc"])
    out = pd.Series(-1, index=f.index, dtype=int)
    for aid, g in f.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc")
        X = _condition_matrix(g, ctx.loc[aid])
        base = (g["steady"] & (g["window_start_utc"] < g["window_start_utc"].min() + pd.Timedelta(hours=BASELINE_H))).to_numpy()
        Xb = X[base].dropna()
        if len(Xb) < MIN_BASELINE or X.shape[1] == 0:
            continue
        mu, sd = Xb.mean(), Xb.std().replace(0, 1.0)
        Z = ((X - mu) / sd)
        Zb = Z[base].dropna().to_numpy()
        rng = np.random.default_rng(seed)
        sub = Zb[rng.choice(len(Zb), min(len(Zb), 3000), replace=False)]
        best, best_score = None, None
        for k in k_range:
            if len(Zb) < 10 * k:
                continue
            if method == "kmeans":
                m = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Zb)
                s = silhouette_score(sub, m.predict(sub))          # higher is better
            else:
                m = GaussianMixture(n_components=k, covariance_type="diag", n_init=2, random_state=seed).fit(Zb)
                s = -m.bic(Zb)                                      # higher is better
            if best is None or s > best_score:
                best, best_score = m, s
        if best is None:
            continue
        ok = Z.notna().all(axis=1).to_numpy()
        lab = np.full(len(g), -1)
        lab[ok] = best.predict(Z[ok].to_numpy())
        out.loc[g.index] = lab
    return out
