"""I01: read the 1-minute feature file and give rows to the detector in time order.

A row with window_start 12:00 covers 12:00:00-12:00:59 and is available at 12:01:00
(`available_utc`). Nothing here reads fault labels. `splits` restricts which pumps
may be read, so fitting code can ask for train only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ..splits import DATA_FULL, SPLITS, load_split_map

MINUTE = pd.Timedelta(minutes=1)


def load_assets(data: Path | str = DATA_FULL) -> pd.DataFrame:
    return pd.read_csv(Path(data) / "assets.csv")


def _allowed_ids(data, splits) -> set[str]:
    bad = [s for s in splits if s not in SPLITS]
    if bad:
        raise ValueError(f"unknown split(s) {bad}; use {SPLITS}")
    m = load_split_map(Path(data) / "assets.csv")
    return {a for a, s in m.items() if s in splits}


def load_features(asset_ids=None, data: Path | str = DATA_FULL,
                  splits: tuple[str, ...] = SPLITS) -> pd.DataFrame:
    """Rows of the requested pumps, sorted by asset and time, with `available_utc`.

    Raises if a requested pump is outside `splits` (e.g. fitting code passes ("train",)).
    """
    allowed = _allowed_ids(data, splits)
    if asset_ids is None:
        asset_ids = sorted(allowed)
    elif isinstance(asset_ids, str):
        asset_ids = [asset_ids]
    outside = sorted(set(asset_ids) - allowed)
    if outside:
        raise ValueError(f"pumps {outside} are not in the allowed splits {splits}")
    f = pd.read_parquet(Path(data) / "features_1min.parquet", filters=[("asset_id", "in", list(asset_ids))])
    f["window_start_utc"] = pd.to_datetime(f["window_start_utc"], utc=True)
    f = f.sort_values(["asset_id", "window_start_utc"], kind="stable").reset_index(drop=True)
    f["available_utc"] = f["window_start_utc"] + MINUTE
    return f


def stream_rows(asset_id: str, data: Path | str = DATA_FULL,
                splits: tuple[str, ...] = SPLITS) -> Iterator[pd.Series]:
    """Give the rows of one pump one at a time, oldest first (a row only after its minute ends)."""
    f = load_features(asset_id, data, splits)
    for _, row in f.iterrows():
        yield row
