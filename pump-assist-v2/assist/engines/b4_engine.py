"""Engine 'b4': run the B4 detector (physics-context-detector/b4.py, imported unchanged, read only) and return its alerts as AlertRecords.

B4 uses the simulator's pump curve, control laws and profile constants, so this engine is a reference, not part of the generator-free v2.
It reads the stored files through B4's own loader, so replayed rows and channel filters are not supported. The split guard still applies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..alert import AlertRecord, Evidence
from ..config import load_machine
from ..evidence.rules import certainty_label
from ..splits import DATA_FULL, REPO_ROOT, load_split_map

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
        P.update(json.loads((Path(cfg_b4["params_file"]) if Path(cfg_b4["params_file"]).is_absolute() else REPO_ROOT / cfg_b4["params_file"]).read_text()))
    P.update(cfg_b4.get("params") or {})
    base = _b4().PARAMS
    return {k: (tuple(v) if isinstance(base.get(k), tuple) and isinstance(v, list) else v) for k, v in P.items()}


def run_b4_engine(asset_ids, splits, data=DATA_FULL, cfg_b4: dict | None = None):
    """Returns (detection frame with a per-minute `score`, list of AlertRecord) for the given pumps, like pipeline.run_pipeline."""
    b4 = _b4()
    ids = [asset_ids] if isinstance(asset_ids, str) else list(asset_ids)
    allowed = {a for a, s in load_split_map(Path(data) / "assets.csv").items() if s in splits}
    outside = sorted(set(ids) - allowed)
    if outside:
        raise ValueError(f"pumps {outside} are not in the allowed splits {splits}")
    inputs = b4.load_inputs(str(data), ids)
    pred, frames = b4.run_b4(inputs, b4_params(cfg_b4 or {}), return_frames=True)
    machine = load_machine()
    alerts = []
    for r in pred.itertuples(index=False):
        p = float(r.confidence)
        parts = [t.strip() for t in str(r.evidence).split(";") if t.strip()] or [str(r.evidence)]
        alerts.append(AlertRecord(asset_id=r.asset_id, alert_time_utc=r.alert_time_utc, event_class=r.event_class, fault_type=r.fault_type,
                                  probability=p, certainty=certainty_label(p, machine), top_types=[(r.fault_type, p)],
                                  evidence=[Evidence("b4", 0.0, t) for t in parts], recommended_action=str(r.recommended_action),
                                  source=f"b4:{r.detector}"))
    alerts.sort(key=lambda a: (a.asset_id, a.alert_time_utc))
    parts_det = []
    for aid, (f, _channels) in frames.items():
        zc = [c for c in f.columns if c.startswith("z_")]
        score = np.nanmax(np.abs(f[zc].to_numpy(float)), axis=1) if zc else np.zeros(len(f))
        t = pd.to_datetime(f["t"], utc=True)
        parts_det.append(pd.DataFrame({"asset_id": aid, "window_start_utc": t, "available_utc": t + pd.Timedelta(minutes=1),
                                       "score": np.nan_to_num(score, nan=0.0), "steady": True, "flag": False, "sustained": False,
                                       "in_baseline": False, "run": 0}))
    det = pd.concat(parts_det, ignore_index=True) if parts_det else pd.DataFrame(columns=["asset_id", "window_start_utc", "score"])
    det.attrs["low_trust_fraction"] = {}
    return det, alerts
