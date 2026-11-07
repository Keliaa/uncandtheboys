"""Q01: data quality and channel trust.

Per channel and minute, trust is 0..1:
  0.0 frozen (std = 0 for `flat_min` steady minutes in a row) or no data (gap minute)
  0.2 quality flag "out of range" set in the minute
  0.3 std much higher than normal (zs > noise_k)
  1.0 otherwise
The detector ignores channels with trust < 0.5. Sensor problems found here become SENSOR alerts.
Works on the z-frame made by assist.detect.zscore.z_frame (it has flat_*, zs_*, gap, steady).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import CH  # noqa: E402

TRUST_MIN = 0.5
FLAT_MIN = 15
NOISE_K = 8.0
STUCK_CHANNELS = ("pd", "ps", "q", "bt", "i", "tf")   # channels where std = 0 for a long time cannot be real


def _runs(x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x), dtype=int)
    c = 0
    for i, v in enumerate(x):
        c = c + 1 if v else 0
        out[i] = c
    return out


def trust_frame(zf: pd.DataFrame, flat_min: int = FLAT_MIN, noise_k: float = NOISE_K) -> pd.DataFrame:
    """Same row order as zf. Columns trust_<key> (key as in baselines.CH), gap_run, stuck_run_<key>, noise_<key>."""
    out = []
    for a, g in zf.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc")
        d = pd.DataFrame(index=g.index)
        gap = g["gap"].to_numpy(bool)
        d["gap_run"] = _runs(gap)
        qa = g["quality_any"].fillna(0).astype(int).to_numpy() if "quality_any" in g else np.zeros(len(g), int)
        for key in CH:
            trust = np.ones(len(g))
            fl = g.get(f"flat_{key}")
            stuck = _runs((fl.fillna(False).to_numpy(bool) & g["steady"].to_numpy(bool)) if fl is not None
                          else np.zeros(len(g), bool)) if key in STUCK_CHANNELS else np.zeros(len(g), int)
            zs = g.get(f"zs_{key}")
            noisy = (zs.fillna(0).to_numpy() > noise_k) if zs is not None else np.zeros(len(g), bool)
            trust[noisy] = 0.3
            trust[(qa & 2) > 0] = np.minimum(trust[(qa & 2) > 0], 0.2)
            trust[stuck >= flat_min] = 0.0
            trust[gap] = 0.0
            d[f"stuck_run_{key}"] = stuck
            d[f"noise_{key}"] = noisy
            d[f"trust_{key}"] = trust
        out.append(d)
    return pd.concat(out).loc[zf.index]


def sensor_alerts(zf: pd.DataFrame, tf: pd.DataFrame, machine: dict, gap_persist: int = 10,
                  flat_min: int = FLAT_MIN, noise_persist: int = 10, refractory=pd.Timedelta(hours=12)):
    """SENSOR AlertRecords: COMMUNICATION_LOSS, SENSOR_STUCK, SENSOR_NOISE_SPIKES (first minute they hold long enough)."""
    from ..alert import AlertRecord, Evidence
    from ..config import action_for
    alerts = []
    for a, g in zf.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc")
        t = tf.loc[g.index]
        last: dict[str, pd.Timestamp] = {}

        def emit(when, ftype, text, ch):
            key = f"{ftype}:{ch}"
            when = when + pd.Timedelta(minutes=1)
            if key in last and when - last[key] < refractory:
                return
            last[key] = when
            alerts.append(AlertRecord(
                asset_id=a, alert_time_utc=when.strftime("%Y-%m-%dT%H:%M:%SZ"), event_class="SENSOR", fault_type=ftype,
                probability=0.6, certainty="Medium", evidence=[Evidence(ch, 0.0, text)],
                recommended_action=action_for(machine, ftype)["text"]))

        times = g["window_start_utc"].to_numpy()
        for i in np.flatnonzero(t["gap_run"].to_numpy() == gap_persist):
            emit(pd.Timestamp(times[i]), "COMMUNICATION_LOSS", "Rows are missing", "gateway")
        for key in STUCK_CHANNELS:
            for i in np.flatnonzero(t[f"stuck_run_{key}"].to_numpy() == flat_min):
                emit(pd.Timestamp(times[i]), "SENSOR_STUCK", f"{CH[key]} has the same value while the pump runs", CH[key])
        for key in CH:
            run = _runs(t[f"noise_{key}"].to_numpy(bool) & g["steady"].to_numpy(bool))
            for i in np.flatnonzero(run == noise_persist):
                emit(pd.Timestamp(times[i]), "SENSOR_NOISE_SPIKES", f"{CH[key]} varies much more than normal", CH[key])
    return alerts
