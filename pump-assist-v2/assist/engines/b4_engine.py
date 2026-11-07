"""Engine 'b4': the B4 detector (physics-context-detector/b4.py, imported unchanged, read only), fed with the pipeline's shared inputs.

The shared load stage (split guard, feature rows, replayed rows, channel filter) builds B4's `inputs` dict, so B4 sees exactly the rows the v2
engine sees. B4's alerts come back as AlertRecords and go through the same shared stages (vibration evidence, episodes, recommend, what-if) as
v2's. B4 uses the simulator's pump curve, control laws and profile constants, so this engine is a reference, not part of the generator-free v2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..alert import AlertRecord, Evidence
from ..evidence.rules import certainty_label
from ..splits import REPO_ROOT

B4_DIR = REPO_ROOT / "physics-context-detector"


def _b4():
    for p in (str(REPO_ROOT / "original-challenge" / "src"), str(B4_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import b4          # noqa: WPS433  (physics-context-detector/b4.py)
    return b4


def b4_params(cfg_b4: dict) -> dict:
    """PARAMS overrides from the settings: params_file first, then params (JSON lists become tuples where B4 expects them)."""
    P: dict = {}
    if cfg_b4.get("params_file"):
        f = Path(cfg_b4["params_file"])
        P.update(json.loads((f if f.is_absolute() else REPO_ROOT / f).read_text()))
    P.update(cfg_b4.get("params") or {})
    base = _b4().PARAMS
    return {k: (tuple(v) if isinstance(base.get(k), tuple) and isinstance(v, list) else v) for k, v in P.items()}


def b4_inputs(sh) -> dict:
    """B4's `inputs` dict from the shared inputs (assets, 1-minute rows, operating context, vibration snapshots)."""
    ids = sorted(sh.feats.asset_id.unique())
    assets = sh.assets[sh.assets.asset_id.isin(ids)].reset_index(drop=True).copy()
    feat = sh.feats.drop(columns=["available_utc"], errors="ignore")
    for aid in ids:      # channels the pump does not have (channel filter) and rows that have not arrived (replay) are not B4's to guess
        i = assets.index[assets.asset_id == aid][0]
        assets.loc[i, "channels_available"] = "|".join(sh.cfg[aid].channels_available)
        last = feat.loc[feat.asset_id == aid, "window_start_utc"].max()
        end = pd.Timestamp(assets.loc[i, "window_end_utc"])
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if last + pd.Timedelta(minutes=1) < end:
            assets.loc[i, "window_end_utc"] = (last + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ctx = pd.read_csv(Path(sh.data) / "operating_context.csv")
    ctx = ctx[ctx.asset_id.isin(ids)].reset_index(drop=True)
    sp = Path(sh.data) / "vibration_snapshots.parquet"
    vib = [a for a in ids if a in set(sh.vib_ids)]
    snaps = pd.read_parquet(sp, filters=[("asset_id", "in", vib)]) if sp.exists() and vib else None
    return {"assets": assets, "feat": feat, "ctx": ctx, "snaps": snaps}


def run_b4_on_inputs(sh, cfg_b4: dict | None = None):
    """Returns (detection frame with a per-minute `score`, list of AlertRecord)."""
    b4 = _b4()
    P = b4_params(cfg_b4 or {})
    pred, frames = b4.run_b4(b4_inputs(sh), P, return_frames=True)
    from ..config import load_machine
    machine = load_machine()
    alerts = []
    for r in pred.itertuples(index=False):
        p = float(r.confidence)
        parts = [t.strip() for t in str(r.evidence).split(";") if t.strip()] or [str(r.evidence)]
        alerts.append(AlertRecord(asset_id=r.asset_id, alert_time_utc=r.alert_time_utc, event_class=r.event_class, fault_type=r.fault_type,
                                  probability=p, certainty=certainty_label(p, machine), top_types=[(r.fault_type, p)],
                                  evidence=[Evidence("b4", 0.0, t) for t in parts], recommended_action=str(r.recommended_action),
                                  source=f"b4:{r.detector}"))
    k_fast = float((P or {}).get("k_fast", b4.PARAMS["k_fast"]))
    parts_det = []
    for aid, (f, _channels) in frames.items():
        zc = [c for c in f.columns if c.startswith("z_")]
        score = np.nan_to_num(np.nanmax(np.abs(f[zc].to_numpy(float)), axis=1) if zc else np.zeros(len(f)), nan=0.0)
        steady = f["steady"].to_numpy(bool) if "steady" in f else np.ones(len(f), bool)
        t = pd.to_datetime(f["t"], utc=True)
        parts_det.append(pd.DataFrame({"asset_id": aid, "window_start_utc": t, "available_utc": t + pd.Timedelta(minutes=1), "score": score,
                                       "steady": steady, "flag": steady & (score > k_fast), "sustained": False, "in_baseline": False, "run": 0}))
    det = pd.concat(parts_det, ignore_index=True) if parts_det else pd.DataFrame(columns=["asset_id", "window_start_utc", "score", "steady", "flag"])
    det.attrs["low_trust_fraction"] = {}
    return det, alerts
