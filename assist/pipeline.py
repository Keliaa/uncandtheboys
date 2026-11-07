"""The version 1 chain: ingest (I01) -> detect (D01) -> evidence (E01) -> recommend (R01).

Reads telemetry only. Nothing here writes to a pump. The split guard decides which pumps may be used.
"""
from __future__ import annotations

import pandas as pd

from .alert import AlertRecord, SUBMISSION_COLUMNS, to_submission_rows
from .config import _asset_from_dict, assets_from_csv, load_machine
from .detect.zscore import DEFAULT_K, DEFAULT_PERSIST, detect, to_alerts_with_rules
from .ingest.features import load_assets, load_features
from .recommend.actions import add_actions
from .splits import DATA_FULL


def run_pipeline(asset_ids, splits=("train", "validation", "test"), data=DATA_FULL,
                 k: float = DEFAULT_K, persist: int = DEFAULT_PERSIST):
    """Returns (detection frame with per-minute scores, list of AlertRecord)."""
    assets, machine = load_assets(data), load_machine()
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(f"{data}/assets.csv")}
    feats = load_features(asset_ids, data, splits)
    det = detect(feats, assets, k, persist)
    alerts = add_actions(to_alerts_with_rules(det, cfg, machine, k), machine)
    return det, alerts


def predictions_frame(alerts: list[AlertRecord]) -> pd.DataFrame:
    return pd.DataFrame(to_submission_rows(alerts), columns=SUBMISSION_COLUMNS)
