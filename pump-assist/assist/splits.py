"""Pump splits. Keep the supplied splits: never mix rows of different splits.

train: PUMP-01..07 (fit), validation: PUMP-08, 09 (choose), test: PUMP-10..12
(final warnings only; do not tune on them). A pump belongs to exactly one split.
`assets.csv` is the source of truth; the constants are a fallback and a check.
"""
from __future__ import annotations

import csv
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]      # pump-assist/: everything this experiment writes stays here
REPO_ROOT = EXP_ROOT.parent                          # shared, read-only: data/, original-challenge/, baseline-reproduction/
DATA_FULL = REPO_ROOT / "data" / "full"

SPLITS = ("train", "validation", "test", "stress")      # "stress": generated fleets (make_dataset.py --tier stress), labels in private/
EXPECTED = {
    "train": [f"PUMP-{i:02d}" for i in range(1, 8)],
    "validation": ["PUMP-08", "PUMP-09"],
    "test": ["PUMP-10", "PUMP-11", "PUMP-12"],
}


def load_split_map(assets_csv: Path | str | None = None) -> dict[str, str]:
    """asset_id -> split, from assets.csv (falls back to the documented split)."""
    path = Path(assets_csv) if assets_csv else DATA_FULL / "assets.csv"
    if not path.exists():
        return {a: s for s, ids in EXPECTED.items() for a in ids}
    with open(path, newline="", encoding="utf-8") as f:
        m = {r["asset_id"]: r["split"] for r in csv.DictReader(f)}
    bad = {a: s for a, s in m.items() if s not in SPLITS}
    if bad:
        raise ValueError(f"unknown split values in {path}: {bad}")
    return m


def pumps_in(split: str, split_map: dict[str, str] | None = None) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    m = split_map or load_split_map()
    return sorted(a for a, s in m.items() if s == split)


def require_split(asset_ids, allowed: tuple[str, ...], purpose: str,
                  split_map: dict[str, str] | None = None) -> None:
    """Raise if any pump is outside the splits allowed for `purpose`.

    Example: require_split(ids, ("train",), "fitting a baseline").
    """
    m = split_map or load_split_map()
    wrong = {a: m.get(a, "unknown") for a in asset_ids if m.get(a) not in allowed}
    if wrong:
        raise ValueError(f"{purpose}: only {allowed} pumps allowed, got {wrong}")
