"""V01: statistics of the vibration snapshots (RMS, crest factor, kurtosis) and their change against the pump's own start.

Snapshots are 1 s recordings at 10 240 Hz every 4 h while running (T1 and T2 pumps). Each value is compared with the median
of the pump's first `BASELINE_N` snapshots (about the first day). A growing RMS together with a rising crest factor and
kurtosis points to bearing wear (small repeated impacts). An alert may only use snapshots taken at or before its time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from ..alert import AlertRecord, Evidence
from ..splits import DATA_FULL, SPLITS, load_split_map

BASELINE_N = 6
RMS_RATIO_LIMIT = 1.5       # evidence is given when RMS is at least this multiple of baseline ...
KURT_DELTA_LIMIT = 1.0      # ... or kurtosis rose by this much, or crest factor by CREST_DELTA_LIMIT
CREST_DELTA_LIMIT = 1.5      # healthy pumps (01, 09) vary by up to about 1 in crest factor


def snapshot_stats(asset_ids=None, data: Path | str = DATA_FULL, splits: tuple[str, ...] = SPLITS) -> pd.DataFrame:
    """One row per snapshot: rms, crest, kurtosis and their change against the pump's baseline snapshots."""
    m = load_split_map(Path(data) / "assets.csv")
    allowed = {a for a, s in m.items() if s in splits}
    ids = sorted(allowed) if asset_ids is None else ([asset_ids] if isinstance(asset_ids, str) else list(asset_ids))
    outside = sorted(set(ids) - allowed)
    if outside:
        raise ValueError(f"pumps {outside} are not in the allowed splits {splits}")
    if not ids:      # pumps without a vibration sensor (or none requested)
        return pd.DataFrame(columns=['asset_id', 'snapshot_id', 'timestamp_utc', 'rms', 'crest', 'kurtosis', 'rms_ratio', 'crest_delta', 'kurt_delta', 'has_baseline', 'changed'])
    v = pd.read_parquet(Path(data) / "vibration_snapshots.parquet", filters=[("asset_id", "in", ids)])
    rows = []
    for _, r in v.iterrows():
        x = np.asarray(r["samples"], dtype=float)
        x = x - x.mean()
        rms = float(np.sqrt(np.mean(x ** 2)))
        rows.append({"asset_id": r.asset_id, "snapshot_id": r.snapshot_id, "timestamp_utc": pd.Timestamp(r.timestamp_utc),
                     "rms": rms, "crest": float(np.max(np.abs(x)) / max(rms, 1e-12)), "kurtosis": float(kurtosis(x))})
    d = pd.DataFrame(rows).sort_values(["asset_id", "timestamp_utc"]).reset_index(drop=True)
    parts = []
    for aid, g in d.groupby("asset_id", sort=False):
        g = g.copy()
        b = g.iloc[:BASELINE_N]
        g["rms_ratio"] = g["rms"] / b["rms"].median()
        g["crest_delta"] = g["crest"] - b["crest"].median()
        g["kurt_delta"] = g["kurtosis"] - b["kurtosis"].median()
        g["has_baseline"] = np.arange(len(g)) >= BASELINE_N      # the first snapshots define the baseline
        g["changed"] = g.has_baseline & ((g.rms_ratio >= RMS_RATIO_LIMIT) | (g.kurt_delta >= KURT_DELTA_LIMIT) | (g.crest_delta >= CREST_DELTA_LIMIT))
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_snapshot_evidence(alerts: list[AlertRecord], stats: pd.DataFrame) -> list[AlertRecord]:
    """For BEARING_WEAR alerts, add the change of the latest earlier snapshot as evidence (z field holds the RMS ratio)."""
    for a in alerts:
        if a.fault_type != "BEARING_WEAR":
            continue
        t = pd.Timestamp(a.alert_time_utc)
        s = stats[(stats.asset_id == a.asset_id) & (stats.timestamp_utc <= t) & stats.has_baseline]
        if s.empty:
            continue
        r = s.iloc[-1]
        what = ("shows a change" if r.changed else "shows no clear change (this argues against bearing wear)")
        a.evidence.append(Evidence("vibration_snapshot", float(r.rms_ratio),
                                   f"Vibration snapshot at {r.timestamp_utc:%Y-%m-%d %H:%M} {what}: RMS {r.rms_ratio:.2f} x start value, "
                                   f"crest factor {r.crest_delta:+.2f}, kurtosis {r.kurt_delta:+.2f}"))
    return alerts
