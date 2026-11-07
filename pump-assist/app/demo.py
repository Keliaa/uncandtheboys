"""C03: load the state written by scripts/c03_demo.py (replay of one pump, alerts with the time the system first showed them)."""
import json
from pathlib import Path

import pandas as pd

from assist.alert import AlertRecord, Evidence

DEMO_DIR = Path(__file__).resolve().parents[1] / "results" / "demo"


def available(demo_dir: Path = DEMO_DIR) -> bool:
    return (demo_dir / "alerts_live.json").exists() and (demo_dir / "replayed_minutes.parquet").exists()


def load_demo(demo_dir: Path = DEMO_DIR):
    """Returns (meta, [(detected_at_replay_utc, AlertRecord)], minute rows)."""
    state = json.loads((demo_dir / "alerts_live.json").read_text())
    out = []
    for d in state["alerts"]:
        d = dict(d)
        when = pd.Timestamp(d.pop("detected_at_replay_utc"))
        d["evidence"] = [Evidence(**e) for e in d["evidence"]]
        d["top_types"] = [tuple(t) for t in d["top_types"]]
        out.append((when, AlertRecord(**d)))
    rows = pd.read_parquet(demo_dir / "replayed_minutes.parquet")
    rows["window_start_utc"] = pd.to_datetime(rows["window_start_utc"], utc=True)
    meta = {k: state[k] for k in ("asset_id", "start", "end", "check_min")}
    return meta, out, rows
