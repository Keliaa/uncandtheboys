"""D03: PCA and Hotelling T-squared on the D01 z-scores.

Finds unusual combinations of channels when each channel alone is in range, and gives the share of each channel in
the alarm. One model per instrumentation tier (pumps of one tier share the same channels). Fitted on healthy TRAIN
minutes only: steady running, outside labelled event windows. Limits are the 99.9 % value of the healthy minutes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import CH  # noqa: E402

Z_CLIP = 20.0
VAR_KEPT = 0.95


@dataclass
class PcaModel:
    tier: str
    channels: list[str]          # z-score column keys (CH keys)
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray       # (k, d)
    eigvals: np.ndarray          # (k,)
    t2_limit: float
    spe_limit: float

    def _x(self, Z: np.ndarray) -> np.ndarray:
        return (np.clip(Z, -Z_CLIP, Z_CLIP) - self.mean) / self.scale

    def scores(self, Z: np.ndarray):
        """Z: (n, d) z-scores for self.channels. Returns T2, SPE, per-channel T2 contribution shares (n, d)."""
        X = np.nan_to_num(self._x(Z), nan=0.0)
        T = X @ self.components.T                      # (n, k)
        t2 = (T ** 2 / self.eigvals).sum(axis=1)
        R = X - T @ self.components
        spe = (R ** 2).sum(axis=1)
        contrib = np.clip(X * ((T / self.eigvals) @ self.components), 0.0, None)   # channel part of T2 (negative parts dropped)
        share = contrib / np.maximum(contrib.sum(axis=1, keepdims=True), 1e-12)
        return t2, spe, share


def tier_channels(zf: pd.DataFrame, asset_ids) -> list[str]:
    """CH keys that carry z-scores in every pump of the group."""
    keys = []
    for k in CH:
        col = f"z_{k}"
        if col in zf and all(zf.loc[zf.asset_id == a, col].notna().any() for a in asset_ids):
            keys.append(k)
    return keys


def fit_tier_model(tier: str, zf: pd.DataFrame, healthy: pd.Series, asset_ids, quantile: float = 0.999) -> PcaModel:
    keys = tier_channels(zf, asset_ids)
    Z = np.clip(zf.loc[healthy & zf.asset_id.isin(asset_ids), [f"z_{k}" for k in keys]].to_numpy(float), -Z_CLIP, Z_CLIP)
    Z = Z[np.isfinite(Z).any(axis=1)]
    # Slow channels (wireless, every 5 min) are missing in most minutes: a missing value counts as "no deviation" (0).
    mean = np.nanmean(Z, axis=0)
    scale = np.maximum(np.nanstd(Z, axis=0), 1.0)    # a channel that is very quiet when healthy must not dominate T2
    X = np.nan_to_num((Z - mean) / scale, nan=0.0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    k = int(np.searchsorted(np.cumsum(var) / var.sum(), VAR_KEPT) + 1)
    m = PcaModel(tier, keys, mean, scale, Vt[:k], var[:k], 0.0, 0.0)
    t2, spe, _ = m.scores(Z)
    m.t2_limit, m.spe_limit = float(np.quantile(t2, quantile)), float(np.quantile(spe, quantile))
    return m


def healthy_mask(zf: pd.DataFrame, events: pd.DataFrame | None, margin_h: float = 1.0, z_max: float = 4.0) -> pd.Series:
    """Steady minutes that look healthy: no gap, every |z| below z_max, and (if events are given) outside every labelled
    event window with a margin before onset. Pre-existing faults can cover a whole window, so `events=None` skips that test."""
    ok = zf["steady"].to_numpy(bool) & ~zf["gap"].to_numpy(bool)
    zc = [f"z_{k}" for k in CH if f"z_{k}" in zf]
    ok &= (zf[zc].abs().max(axis=1).fillna(0.0) < z_max).to_numpy()
    if events is None:
        return pd.Series(ok, index=zf.index)
    ev = events.copy()
    for c in ("onset_utc", "end_utc"):
        ev[c] = pd.to_datetime(ev[c], utc=True)
    for _, e in ev.iterrows():
        m = (zf.asset_id == e.asset_id).to_numpy()
        end = e.end_utc + pd.Timedelta(minutes=30) if pd.notna(e.end_utc) else pd.Timestamp("2100-01-01", tz="UTC")
        win = (zf.window_start_utc >= e.onset_utc - pd.Timedelta(hours=margin_h)) & (zf.window_start_utc <= end)
        ok &= ~(m & win.to_numpy())
    return pd.Series(ok, index=zf.index)


def pca_flags(zf: pd.DataFrame, assets: pd.DataFrame, models: dict[str, PcaModel], use_spe: bool = True):
    """Frame asset_id, window_start_utc, flag (T2 or SPE above its limit on a steady minute), t2, spe, top (top-3 channels)."""
    tier_of = dict(zip(assets.asset_id, assets.instrumentation_tier))
    out = []
    for aid, g in zf.groupby("asset_id", sort=False):
        m = models.get(tier_of[aid])
        if m is None:
            continue
        Z = g.reindex(columns=[f"z_{k}" for k in m.channels]).to_numpy(float)
        t2, spe, share = m.scores(Z)
        flag = g["steady"].to_numpy(bool) & ((t2 > m.t2_limit) | (use_spe & (spe > m.spe_limit)))
        top = [[CH[m.channels[j]] for j in np.argsort(-share[i])[:3]] if flag[i] else [] for i in range(len(g))]
        out.append(pd.DataFrame({"asset_id": aid, "window_start_utc": g["window_start_utc"].to_numpy(), "flag": flag,
                                 "t2": t2, "spe": spe, "top": top}))
    return pd.concat(out, ignore_index=True)
