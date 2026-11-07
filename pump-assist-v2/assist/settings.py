"""Settings of pump-assist-v2: configs/pipeline.yaml over built-in defaults, with environment and runtime overrides.

Precedence: runtime overrides (function argument) > environment variable PUMP_ASSIST_ENGINE (engine only) > configs/pipeline.yaml > defaults here.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .splits import EXP_ROOT

CONFIG_PATH = EXP_ROOT / "configs" / "pipeline.yaml"
ENGINES = ("v2", "b4")

DEFAULTS: dict[str, Any] = {
    "engine": "v2",
    "b4": {"params_file": None, "params": {}},
    "v2": {
        "detect": {"k": 8.0, "persist": 10, "trust_gate": True, "regime": "speed", "baseline": "static",
                   "benign": {"enabled": True, "tail_min": 60, "hs_limit": 2.2}},
        "physics": {"enabled": True, "limit_pct": 2.0, "persist": 30},
        "quality": {"gap_persist": 30, "flat_min": 30},
        "maintenance": {"enabled": True, "step_z": 8.0},
        "evidence": {"e02": True, "gate": True, "sensor_alone": None, "shape_typing": True, "derived_sensor": True,
                     "episodes_quiet_min": 1440, "stable_min": 1},
        "recommend": {"what_if": False},
        "options": {"d02_limit": None, "ewma": None, "v03_alerts": False, "v02_typing": False},
    },
}


class Cfg(dict):
    """dict with attribute access: cfg.v2.detect.k"""
    def __getattr__(self, name: str):
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(v) if isinstance(v, dict) else v


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise KeyError(f"unknown setting {path + k!r}; known: {sorted(out)}")
        if isinstance(out[k], dict) and isinstance(v, dict) and out[k] and k != "params":
            out[k] = _merge(out[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


def load_settings(overrides: dict | None = None, path: Path | str | None = None, engine: str | None = None) -> Cfg:
    s = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        s = _merge(s, yaml.safe_load(p.read_text()) or {})
    env = os.environ.get("PUMP_ASSIST_ENGINE")
    if env:
        s["engine"] = env
    if overrides:
        s = _merge(s, overrides)
    if engine:
        s["engine"] = engine
    if s["engine"] not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {s['engine']!r}")
    return Cfg(s)
