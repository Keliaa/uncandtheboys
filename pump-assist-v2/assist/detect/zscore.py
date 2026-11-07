"""D01: robust z-score per operating mode (the B1 method).

Uses regime_frame() and zscores() from the reference baselines. The baseline for each
pump is its own first 24 h of steady running, by speed regime. Because the baseline is a
commissioning baseline, z-scores inside those first 24 h use data up to 24 h ahead.
Alerts from this detector should not be raised in that window (use `baseline_until`).

Settings k and persist are chosen on TRAIN pumps only (scripts/tune_d01.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _ref  # noqa: F401
from retrofit_sim.baselines import BASELINE_H, CH, regime_frame, zscores  # noqa: E402

from .benign import benign_lookup

DEFAULT_K = 8.0       # chosen on train pumps only (results/d01_grid.csv)
DEFAULT_PERSIST = 10   # minutes (worklist D01)


def z_frame(features: pd.DataFrame, assets: pd.DataFrame, regime: str = "speed", data=None) -> pd.DataFrame:
    """Per-minute z-scores (columns z_<key>) for the pumps in `features`. Each pump is scaled on itself.

    regime: 'speed' (speed bands, D01) or 'kmeans' / 'gmm' (operating-mode clusters, D05; needs operating_context.csv in `data`)."""
    f = regime_frame(features, assets)
    if regime != "speed":
        from ..splits import DATA_FULL
        from .modes import cluster_regimes
        from .regression import load_context
        f["regime"] = cluster_regimes(f, load_context(sorted(f.asset_id.unique()), data or DATA_FULL), regime)
    return zscores(f)


def detect(features: pd.DataFrame, assets: pd.DataFrame, k: float = DEFAULT_K,
           persist: int = DEFAULT_PERSIST, trust_gate: bool = False, zf: pd.DataFrame | None = None,
           benign: pd.DataFrame | None = None, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per pump-minute: score, flag, persistent flag and deviating channels.

    flag      : steady running and some |z| > k on that minute
    sustained : flag held for `persist` minutes in a row (use this to raise an alert)
    channels  : list of {channel, z} for |z| > k, largest first (input for the evidence part)
    """
    if zf is None:
        zf = z_frame(features, assets)
    keys = list(CH)
    if trust_gate:      # Q01: ignore channels with trust < 0.5
        from ..ingest.quality import TRUST_MIN, trust_frame
        tf = trust_frame(zf)
        zf = zf.copy()
        for c in keys:
            zf.loc[tf[f"trust_{c}"] < TRUST_MIN, f"z_{c}"] = np.nan
    out = []
    for a, g in zf.groupby("asset_id", sort=False):
        g = g.sort_values("window_start_utc").reset_index(drop=True)
        Z = g[[f"z_{c}" for c in keys]].to_numpy(dtype=float)
        absz = np.nan_to_num(np.abs(Z), nan=0.0)
        score = absz.max(axis=1)
        flag = g["steady"].to_numpy(bool) & (score > k)
        if extra is not None:     # other detectors (D02): extra[asset_id, window_start_utc, flag]
            e = extra[extra.asset_id == a].set_index("window_start_utc")["flag"]
            flag |= g["steady"].to_numpy(bool) & e.reindex(g["window_start_utc"]).fillna(False).to_numpy(bool)
        flag &= ~benign_lookup(benign, a, g["window_start_utc"])   # D06: hold alerts in benign periods
        run = np.zeros(len(g), dtype=int)
        c = 0
        for i, x in enumerate(flag):
            c = c + 1 if x else 0
            run[i] = c
        chans, zall = [], []
        for i in range(len(g)):
            zall.append({CH[keys[j]]: float(Z[i, j]) for j in range(len(keys)) if np.isfinite(Z[i, j])} if flag[i] else {})
            if flag[i]:
                idx = [j for j in np.argsort(-absz[i]) if absz[i, j] > k]
                chans.append([{"channel": CH[keys[j]], "z": float(Z[i, j])} for j in idx])
            else:
                chans.append([])
        t0 = g["window_start_utc"].min()
        out.append(pd.DataFrame({
            "asset_id": a,
            "window_start_utc": g["window_start_utc"],
            "available_utc": g["window_start_utc"] + pd.Timedelta(minutes=1),
            "in_baseline": g["window_start_utc"] < t0 + pd.Timedelta(hours=BASELINE_H),
            "steady": g["steady"].to_numpy(bool), "score": score, "flag": flag, "run": run, "sustained": run >= persist, "channels": chans, "z_all": zall,
        }))
    return pd.concat(out, ignore_index=True)


REFRACTORY = pd.Timedelta(hours=12)


def to_alerts(det: pd.DataFrame, refractory: pd.Timedelta = REFRACTORY):
    """Turn sustained flags into AlertRecords (detection only: fault type is UNKNOWN until E01).

    One alert when the flag first holds for `persist` minutes, then at most one per `refractory`.
    No alert inside the commissioning-baseline window.
    """
    from ..alert import AlertRecord, Evidence
    alerts = []
    for a, g in det.groupby("asset_id", sort=False):
        last = None
        # first minute of each sustained stretch
        starts = g[g.sustained & ~g.sustained.shift(fill_value=False) & ~g.in_baseline]
        for _, r in starts.iterrows():
            if last is not None and r.available_utc - last < refractory:
                continue
            last = r.available_utc
            ev = [Evidence(c["channel"], c["z"], f"{c['channel']} z={c['z']:+.1f}") for c in r.channels[:3]]
            alerts.append(AlertRecord(asset_id=a, alert_time_utc=r.available_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      event_class="MACHINE", fault_type="UNKNOWN",
                                      probability=float(min(0.99, max(0.05, r.score / 20.0))), evidence=ev,
                                      recommended_action="Inspect the pump. Examine the channels in the evidence."))
    return alerts


def to_alerts_with_rules(det: pd.DataFrame, assets_cfg: dict, machine: dict, k: float = DEFAULT_K,
                         refractory: pd.Timedelta = REFRACTORY, classifier=None, gate: bool = False,
                         sensor_alone: float | None = None, stable_min: int = 1):
    """Like to_alerts, but E01 gives fault type, confidence, certainty and evidence text.

    assets_cfg: asset_id -> assist.config.Asset.
    stable_min: the first alert of a sustained stretch waits until the diagnosed (class, type) has been the same for this many
    minutes in a row (B4 idea #12); 1 = alert at the first sustained minute. A stretch that ends before it is stable gives no alert.
    """
    from ..alert import AlertRecord
    from ..evidence.rules import apply_classifier, classify
    alone = [] if sensor_alone is None else [sensor_alone]

    def diagnose(a, r):
        d = classify(r.z_all, assets_cfg[a], machine, k, *alone)
        if classifier is not None and d["event_class"] == "MACHINE":
            d = apply_classifier(d, classifier(a, r.z_all, r.available_utc), assets_cfg[a], machine, r.z_all, k, gate)
        return d

    alerts = []
    for a, g in det.groupby("asset_id", sort=False):
        last = {}
        live = (g.sustained & ~g.in_baseline).to_numpy()
        rows = g.reset_index(drop=True)
        i = 0
        while i < len(rows):
            if not live[i]:
                i += 1
                continue
            j = i
            while j < len(rows) and live[j]:
                j += 1
            chosen, prev, count = None, None, 0            # rows i..j-1 are one sustained stretch
            for q in range(i, j):
                r = rows.iloc[q]
                d = diagnose(a, r)
                key = (d["event_class"], d["fault_type"])
                count = count + 1 if key == prev else 1
                prev = key
                if count >= stable_min:
                    chosen = (r, d)
                    break
            i = j
            if chosen is None:
                continue
            r, d = chosen
            if d["fault_type"] in last and r.available_utc - last[d["fault_type"]] < refractory:
                continue
            last[d["fault_type"]] = r.available_utc
            alerts.append(AlertRecord(asset_id=a, alert_time_utc=r.available_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), z_at_alert=dict(r.z_all), source="d01", **d))
    return alerts
