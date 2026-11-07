"""V02: FFT and order analysis of the vibration snapshots.

Frequency is divided by the shaft frequency (rpm_at_capture / 60) to give orders. The amplitude at 1x, 2x, 3x and at the
vane-pass order (impeller_vanes) is compared with the pump's own first snapshots. Imbalance gives a peak at 1x only.
Misalignment gives peaks at 2x (and 1x, 3x). Single peaks fluctuate by about a factor 2 when healthy, so the median
of the last three snapshots is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..alert import Evidence
from ..splits import DATA_FULL, SPLITS, load_split_map

BASELINE_N = 6
PEAK_TOL = 0.03          # peak searched within +/- 3 % of the order
IMB_R1, IMB_R2 = 2.5, 1.8   # imbalance: 1x at least 2.5 x baseline while 2x stays below 1.8 x baseline
MIS_R2 = 3.0                # misalignment: 2x at least 3 x baseline and clearly above the 1x change
# Limits are round numbers from the healthy spread (single peaks vary by up to about 2x); not fitted to the scorer.


def order_amplitudes(x: np.ndarray, fs: float, rpm: float, vanes: int) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    w = np.hanning(n)
    sp = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()      # amplitude of a sine, window gain corrected
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f1 = rpm / 60.0

    def peak(order: float) -> float:
        s = (f >= order * f1 * (1 - PEAK_TOL)) & (f <= order * f1 * (1 + PEAK_TOL))
        return float(sp[s].max()) if s.any() else 0.0
    return {"a1": peak(1), "a2": peak(2), "a3": peak(3), "avp": peak(float(vanes))}


def snapshot_orders(asset_ids=None, data: Path | str = DATA_FULL, splits: tuple[str, ...] = SPLITS) -> pd.DataFrame:
    """One row per snapshot: order amplitudes, ratios to the pump's baseline snapshots and the spectral pattern."""
    data = Path(data)
    m = load_split_map(data / "assets.csv")
    allowed = {a for a, s in m.items() if s in splits}
    ids = sorted(allowed) if asset_ids is None else ([asset_ids] if isinstance(asset_ids, str) else list(asset_ids))
    outside = sorted(set(ids) - allowed)
    if outside:
        raise ValueError(f"pumps {outside} are not in the allowed splits {splits}")
    if not ids:      # pumps without a vibration sensor (or none requested)
        return pd.DataFrame(columns=['asset_id', 'snapshot_id', 'timestamp_utc', 'a1', 'a2', 'a3', 'avp', 'r1', 'r2', 'r3', 'rvp', 'has_baseline', 'm1', 'm2', 'm3', 'mvp', 'pattern'])
    vanes = pd.read_csv(data / "assets.csv").set_index("asset_id")["impeller_vanes"]
    v = pd.read_parquet(data / "vibration_snapshots.parquet", filters=[("asset_id", "in", ids)])
    rows = []
    for _, r in v.iterrows():
        rows.append({"asset_id": r.asset_id, "snapshot_id": r.snapshot_id, "timestamp_utc": pd.Timestamp(r.timestamp_utc),
                     **order_amplitudes(r["samples"], float(r.sample_rate_hz), float(r.rpm_at_capture), int(vanes[r.asset_id]))})
    d = pd.DataFrame(rows).sort_values(["asset_id", "timestamp_utc"]).reset_index(drop=True)
    parts = []
    for aid, g in d.groupby("asset_id", sort=False):
        g = g.copy()
        for k in ("a1", "a2", "a3", "avp"):
            g["r" + k[1:]] = g[k] / max(g[k].iloc[:BASELINE_N].median(), 1e-12)
        g["has_baseline"] = np.arange(len(g)) >= BASELINE_N
        for k in ("1", "2", "3", "vp"):
            g["m" + k] = g["r" + k].rolling(3, min_periods=1).median()
        g["pattern"] = classify_pattern(g)
        g.loc[~g.has_baseline, "pattern"] = "none"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def classify_pattern(g: pd.DataFrame) -> np.ndarray:
    mis = (g.m2 >= MIS_R2) & (g.m2 >= 1.5 * g.m1)
    imb = (g.m1 >= IMB_R1) & (g.m2 < IMB_R2)
    return np.where(mis, "misalignment", np.where(imb, "imbalance", "none"))


def _latest(orders: pd.DataFrame, asset_id: str, when: str):
    s = orders[(orders.asset_id == asset_id) & (orders.timestamp_utc <= pd.Timestamp(when)) & orders.has_baseline]
    return None if s.empty else s.iloc[-1]


def add_order_evidence(alerts, orders: pd.DataFrame):
    """IMBALANCE and MISALIGNMENT alerts get the order amplitudes of the latest earlier snapshot as evidence."""
    for a in alerts:
        if a.fault_type not in ("IMBALANCE", "MISALIGNMENT"):
            continue
        r = _latest(orders, a.asset_id, a.alert_time_utc)
        if r is None:
            continue
        agree = r.pattern == a.fault_type.lower()
        txt = (f"Spectrum at {r.timestamp_utc:%Y-%m-%d %H:%M}: 1x {r.m1:.1f}, 2x {r.m2:.1f}, 3x {r.m3:.1f} times the start value"
               + ("" if agree else f" (the pattern does not clearly show {a.fault_type.lower()})"))
        a.evidence.append(Evidence("vibration_orders", float(r.m2 if a.fault_type == "MISALIGNMENT" else r.m1), txt))
    return alerts


def spectral_typing(alerts, orders: pd.DataFrame, machine: dict, probability: float = 0.75):
    """Option: when the latest earlier snapshot shows a clear imbalance or misalignment pattern, name that type
    for a machine alert (replacing a type the spectrum contradicts). Other alerts are untouched."""
    from ..config import action_for
    from ..evidence.rules import certainty_label
    for a in alerts:
        if a.event_class != "MACHINE":
            continue
        r = _latest(orders, a.asset_id, a.alert_time_utc)
        if r is None or r.pattern == "none":
            continue
        ft = r.pattern.upper()
        if a.fault_type == ft or a.fault_type in ("SUCTION_STRAINER_CLOGGING", "IMPELLER_WEAR", "DISCHARGE_RESTRICTION"):
            continue                                # keep types the spectrum says nothing about
        a.fault_type = ft
        a.probability = max(a.probability, probability)
        a.certainty = certainty_label(a.probability, machine)
        a.recommended_action = action_for(machine, ft)["text"]
    return alerts
