"""Filesystem locations. Everything the demo writes stays inside operator-demo/."""
from __future__ import annotations

from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = DEMO_DIR.parent
BUNDLE_DIR = DEMO_DIR / "bundles"
STATE_DIR = DEMO_DIR / "state"
INTERACTION_LOG = STATE_DIR / "interactions.jsonl"

# read-only inputs used by the offline preparation step only
DATA_DIR = REPO_DIR / "data" / "full"
B4_DIR = REPO_DIR / "physics-context-detector"
B4_FROZEN_PARAMS = B4_DIR / "results" / "b4_frozen_params.json"
B4_FROZEN_PREDICTIONS = B4_DIR / "results" / "b4_trainval_predictions.csv"
CHALLENGE_SRC = REPO_DIR / "original-challenge" / "src"


def inside_demo(path: Path) -> bool:
    """True if `path` resolves inside operator-demo/ (outputs must never leave it)."""
    p, d = Path(path).resolve(), DEMO_DIR.resolve()
    return p == d or d in p.parents
