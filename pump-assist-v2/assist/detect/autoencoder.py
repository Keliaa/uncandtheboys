"""D07: autoencoder comparison with PCA (D03).

An MLP (scikit-learn MLPRegressor) is trained to reproduce its input through a small middle layer, on the same healthy z-scores as
D03 (per instrumentation tier, same standardisation). The reconstruction error is the score; the limit is a quantile of the error on
healthy minutes that were NOT used for training (a 20 % hold-out), because training errors are too small. Everything else matches D03
so the two can be compared at the same limit quantile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import CH  # noqa: E402
from .pca import Z_CLIP, tier_channels

HOLDOUT = 0.2
SEED = 0


@dataclass
class AutoencoderModel:
    tier: str
    channels: list[str]
    mean: np.ndarray
    scale: np.ndarray
    net: MLPRegressor
    limit: float

    def _x(self, Z: np.ndarray) -> np.ndarray:
        return np.nan_to_num((np.clip(Z, -Z_CLIP, Z_CLIP) - self.mean) / self.scale, nan=0.0)   # a missing channel counts as no deviation

    def errors(self, Z: np.ndarray):
        """Returns (reconstruction error per row, squared error per channel)."""
        X = self._x(Z)
        e2 = (X - self.net.predict(X)) ** 2
        return e2.sum(axis=1), e2


def fit_tier_autoencoder(tier: str, zf: pd.DataFrame, healthy: pd.Series, asset_ids, quantile: float = 0.99, seed: int = SEED) -> AutoencoderModel:
    keys = tier_channels(zf, asset_ids)
    Z = np.clip(zf.loc[healthy & zf.asset_id.isin(asset_ids), [f"z_{k}" for k in keys]].to_numpy(float), -Z_CLIP, Z_CLIP)
    Z = Z[np.isfinite(Z).any(axis=1)]
    mean = np.nanmean(Z, axis=0)
    scale = np.maximum(np.nanstd(Z, axis=0), 1.0)
    X = np.nan_to_num((Z - mean) / scale, nan=0.0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_hold = int(len(X) * HOLDOUT)
    hold, train = X[idx[:n_hold]], X[idx[n_hold:]]
    d = X.shape[1]
    net = MLPRegressor(hidden_layer_sizes=(max(4, d), max(2, d // 3), max(4, d)), activation="tanh", max_iter=300, random_state=seed,
                       early_stopping=False, alpha=1e-3).fit(train, train)
    m = AutoencoderModel(tier, keys, mean, scale, net, 0.0)
    err = ((hold - net.predict(hold)) ** 2).sum(axis=1)
    m.limit = float(np.quantile(err, quantile))
    return m


def autoencoder_flags(zf: pd.DataFrame, assets: pd.DataFrame, models: dict[str, AutoencoderModel]) -> pd.DataFrame:
    """Frame asset_id, window_start_utc, flag (error above the limit on a steady minute), err, top (3 channels with the largest error)."""
    from retrofit_sim.baselines import CH as CHN
    tier_of = dict(zip(assets.asset_id, assets.instrumentation_tier))
    out = []
    for aid, g in zf.groupby("asset_id", sort=False):
        m = models.get(tier_of[aid])
        if m is None:
            continue
        Z = g.reindex(columns=[f"z_{k}" for k in m.channels]).to_numpy(float)
        err, e2 = m.errors(Z)
        flag = g["steady"].to_numpy(bool) & (err > m.limit)
        top = [[CHN[m.channels[j]] for j in np.argsort(-e2[i])[:3]] if flag[i] else [] for i in range(len(g))]
        out.append(pd.DataFrame({"asset_id": aid, "window_start_utc": g["window_start_utc"].to_numpy(), "flag": flag, "err": err, "top": top}))
    return pd.concat(out, ignore_index=True)
