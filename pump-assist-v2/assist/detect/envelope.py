"""V03: envelope analysis for bearings.

A damaged bearing gives small repeated impacts that excite a housing resonance (2-4 kHz here). Band-pass the snapshot there,
take the envelope (Hilbert transform), and read its spectrum in shaft orders. A peak at the bearing outer-race (BPFO) or
inner-race (BPFI) order, high above the noise floor, points to bearing damage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from ..splits import DATA_FULL, SPLITS, load_split_map

BAND_HZ = (2000.0, 4000.0)
ORDER_RANGE = (0.5, 12.0)       # orders used for the noise floor
PEAK_TOL = 0.04                 # peak searched within +/- 4 % of the defect order (speed is measured, not exact)
HARMONICS = 2                   # BPFO/BPFI and their 2nd harmonic are combined (max)


def envelope_spectrum(x: np.ndarray, fs: float, rpm: float, band=BAND_HZ):
    """Returns (orders, amplitude) of the envelope spectrum."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    env = env - env.mean()
    amp = np.abs(np.fft.rfft(env * np.hanning(len(env)))) * 2.0 / len(env)
    freq = np.fft.rfftfreq(len(env), 1.0 / fs)
    return freq / (rpm / 60.0), amp


def peak_snr(orders: np.ndarray, amp: np.ndarray, target: float, tol: float = PEAK_TOL) -> float:
    """Highest peak within target +/- tol (also its 2nd harmonic) divided by the median spectrum level (noise floor)."""
    noise = np.median(amp[(orders >= ORDER_RANGE[0]) & (orders <= ORDER_RANGE[1])])
    best = 0.0
    for h in range(1, HARMONICS + 1):
        sel = (orders >= h * target * (1 - tol)) & (orders <= h * target * (1 + tol))
        if sel.any():
            best = max(best, float(amp[sel].max()))
    return best / max(float(noise), 1e-12)


def bearing_envelope(asset_ids=None, data: Path | str = DATA_FULL, splits: tuple[str, ...] = SPLITS) -> pd.DataFrame:
    """One row per snapshot: snr_bpfo, snr_bpfi (peak above noise floor, as a ratio)."""
    data = Path(data)
    m = load_split_map(data / "assets.csv")
    allowed = {a for a, s in m.items() if s in splits}
    ids = sorted(allowed) if asset_ids is None else ([asset_ids] if isinstance(asset_ids, str) else list(asset_ids))
    outside = sorted(set(ids) - allowed)
    if outside:
        raise ValueError(f"pumps {outside} are not in the allowed splits {splits}")
    if not ids:      # pumps without a vibration sensor (or none requested)
        return pd.DataFrame(columns=['asset_id', 'snapshot_id', 'timestamp_utc', 'snr_bpfo', 'snr_bpfi'])
    assets = pd.read_csv(data / "assets.csv").set_index("asset_id")
    v = pd.read_parquet(data / "vibration_snapshots.parquet", filters=[("asset_id", "in", ids)])
    rows = []
    for _, r in v.iterrows():
        o, a = envelope_spectrum(np.asarray(r["samples"]), float(r.sample_rate_hz), float(r.rpm_at_capture))
        A = assets.loc[r.asset_id]
        rows.append({"asset_id": r.asset_id, "snapshot_id": r.snapshot_id, "timestamp_utc": pd.Timestamp(r.timestamp_utc),
                     "snr_bpfo": peak_snr(o, a, float(A.bearing_bpfo_order)), "snr_bpfi": peak_snr(o, a, float(A.bearing_bpfi_order))})
    return pd.DataFrame(rows).sort_values(["asset_id", "timestamp_utc"]).reset_index(drop=True)


SNR_LIMIT = 6.0     # healthy pumps (01, 06, 09, 07) stay below 3.8; worn bearings reach 10 to 60


def add_envelope_evidence(alerts, env: pd.DataFrame, limit: float = SNR_LIMIT):
    """BEARING_WEAR alerts get the latest earlier snapshot's BPFO/BPFI peak as evidence, also when it is low."""
    from ..alert import Evidence
    for a in alerts:
        if a.fault_type != "BEARING_WEAR":
            continue
        s = env[(env.asset_id == a.asset_id) & (env.timestamp_utc <= pd.Timestamp(a.alert_time_utc))]
        if s.empty:
            continue
        r = s.iloc[-1]
        hit = max(r.snr_bpfo, r.snr_bpfi) >= limit
        which = "BPFO (outer race)" if r.snr_bpfo >= r.snr_bpfi else "BPFI (inner race)"
        text = (f"Envelope spectrum at {r.timestamp_utc:%Y-%m-%d %H:%M}: peak at {which} is {max(r.snr_bpfo, r.snr_bpfi):.0f} x the noise floor"
                + ("" if hit else " (no bearing defect peak; this argues against bearing wear)"))
        a.evidence.append(Evidence("vibration_envelope", float(max(r.snr_bpfo, r.snr_bpfi)), text))
    return alerts


def bearing_alerts(env: pd.DataFrame, machine: dict, limit: float = SNR_LIMIT, persist: int = 2,
                   refractory: pd.Timedelta = pd.Timedelta(hours=96)):    # one alert per bearing episode: repeats beyond a reminder count as excess
    """BEARING_WEAR AlertRecords when the defect peak is above `limit` in `persist` consecutive snapshots."""
    from ..alert import AlertRecord, Evidence
    from ..config import action_for
    from ..evidence.rules import certainty_label
    alerts = []
    for aid, g in env.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp_utc").reset_index(drop=True)
        snr = np.maximum(g.snr_bpfo, g.snr_bpfi).to_numpy()
        run = 0
        last = None
        for i in range(len(g)):
            run = run + 1 if snr[i] >= limit else 0
            if run >= persist:
                t = g.timestamp_utc.iloc[i] + pd.Timedelta(minutes=1)
                if last is not None and t - last < refractory:
                    continue
                last = t
                which = "BPFO (outer race)" if g.snr_bpfo.iloc[i] >= g.snr_bpfi.iloc[i] else "BPFI (inner race)"
                p = float(min(0.95, 0.6 + snr[i] / 100.0))
                alerts.append(AlertRecord(
                    asset_id=aid, alert_time_utc=t.strftime("%Y-%m-%dT%H:%M:%SZ"), event_class="MACHINE", fault_type="BEARING_WEAR",
                    probability=p, certainty=certainty_label(p, machine),
                    evidence=[Evidence("vibration_envelope", float(snr[i]), f"Envelope spectrum: peak at {which} is {snr[i]:.0f} x the noise floor")],
                    recommended_action=action_for(machine, "BEARING_WEAR")["text"]))
    return alerts
