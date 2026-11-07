"""Segment-reset smoothing (B4 idea #4, data-driven version).

Each channel's z-score is smoothed with an EWMA inside a running segment: the smoothing restarts after every stop or start-up (any
non-steady minute), so a score is never carried over a repair or a restart. A persistent but small deviation that raw z-scores cross
only now and then becomes a clear signal; a one-minute spike does not. The score is the largest |smoothed z| over channels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import CH  # noqa: E402


def smoothed_scores(zf: pd.DataFrame, span: int = 30) -> pd.DataFrame:
    """asset_id, window_start_utc, ewma_score (NaN on non-steady minutes)."""
    keys = [k for k in CH if f"z_{k}" in zf]
    out = []
    for aid, g in zf.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc")
        steady = g["steady"].to_numpy(bool)
        seg = pd.Series((~steady).cumsum(), index=g.index)
        Z = g[[f"z_{k}" for k in keys]].astype(float).where(pd.Series(steady, index=g.index), axis=0)
        S = Z.groupby(seg).transform(lambda x: x.ewm(span=span, adjust=False, ignore_na=True).mean())
        score = np.nan_to_num(S.abs().to_numpy(), nan=0.0).max(axis=1)
        out.append(pd.DataFrame({"asset_id": aid, "window_start_utc": g["window_start_utc"].to_numpy(), "ewma_score": np.where(steady, score, np.nan)}))
    return pd.concat(out, ignore_index=True)
