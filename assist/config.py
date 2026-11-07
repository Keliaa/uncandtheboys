"""Load machine and asset configuration files (S02). No pump knowledge in code.

Machine file: configs/machines/<machine_type>.yaml (channels, symptoms, actions).
Asset file:   configs/assets/<asset_id>.yaml, one per pump, made from assets.csv.
A new pump (or machine) needs only a new asset file, not a code change.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .splits import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_MACHINE = "centrifugal_pump"
NUMERIC_ASSET_FIELDS = {
    "commissioned_year", "pump_rated_flow_m3_h", "pump_rated_head_m", "pump_shutoff_head_m",
    "pump_bep_efficiency", "npsh3_at_bep_m", "design_flow_m3_h", "design_head_m", "static_head_m",
    "motor_rated_kw", "motor_rated_current_a", "motor_rated_rpm", "supply_voltage_v",
    "supply_frequency_hz", "min_speed_ratio", "bearing_bpfo_order", "bearing_bpfi_order",
    "impeller_vanes", "vibration_alarm_mm_s", "vibration_trip_mm_s", "wireless_update_s",
}


def load_machine(machine_type: str = DEFAULT_MACHINE, config_dir: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_dir or CONFIG_DIR) / "machines" / f"{machine_type}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("channels", "symptoms", "actions"):
        if key not in cfg:
            raise ValueError(f"{path}: missing section {key!r}")
    return cfg


@dataclass
class Asset:
    asset_id: str
    machine_type: str
    split: str
    channels_available: list[str]
    nameplate: dict[str, Any]


def _asset_from_dict(d: dict[str, Any]) -> Asset:
    for k in ("asset_id", "machine_type", "split", "channels_available"):
        if k not in d:
            raise ValueError(f"asset file is missing {k!r}")
    return Asset(d["asset_id"], d["machine_type"], d["split"], list(d["channels_available"]),
                 dict(d.get("nameplate", {})))


def load_asset(asset_id: str, config_dir: Path | str | None = None) -> Asset:
    path = Path(config_dir or CONFIG_DIR) / "assets" / f"{asset_id}.yaml"
    with open(path, encoding="utf-8") as f:
        return _asset_from_dict(yaml.safe_load(f))


def _convert(field: str, value: str):
    if value == "":
        return None
    if field in NUMERIC_ASSET_FIELDS:
        x = float(value)
        return int(x) if x.is_integer() and field in {"commissioned_year", "impeller_vanes", "wireless_update_s"} else x
    return value


def assets_from_csv(assets_csv: Path | str, machine_type: str = DEFAULT_MACHINE) -> list[dict[str, Any]]:
    """Convert rows of assets.csv to asset-file dictionaries."""
    out = []
    with open(assets_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nameplate = {k: _convert(k, v) for k, v in row.items()
                         if k not in ("asset_id", "split", "channels_available")}
            out.append({"asset_id": row["asset_id"], "machine_type": machine_type, "split": row["split"],
                        "channels_available": sorted(row["channels_available"].split("|")) if row["channels_available"] else [],
                        "nameplate": nameplate})
    return out


def write_asset_files(assets_csv: Path | str, out_dir: Path | str | None = None,
                      machine_type: str = DEFAULT_MACHINE) -> list[Path]:
    out = Path(out_dir or CONFIG_DIR / "assets")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in assets_from_csv(assets_csv, machine_type):
        p = out / f"{d['asset_id']}.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False)
        paths.append(p)
    return paths


def usable_symptoms(machine: dict[str, Any], asset: Asset) -> dict[str, list[dict[str, Any]]]:
    """Fault types this pump can show, with only the evidence channels it has.

    A fault type is dropped when none of its `needs` alternatives is met.
    SENSOR types keep channel 'any'.
    """
    have = set(asset.channels_available)
    result = {}
    for ftype, spec in machine["symptoms"].items():
        needs = spec.get("needs", [])
        if needs and not any(all(c in have for c in alt) for alt in needs):
            continue
        ev = [e for e in spec["evidence"] if e["channel"] == "any" or e["channel"] in have]
        if ev:
            result[ftype] = ev
    return result


def action_for(machine: dict[str, Any], fault_type: str) -> dict[str, Any]:
    actions = machine["actions"]
    if fault_type in actions:
        return actions[fault_type]
    if fault_type.startswith("SENSOR_") and "SENSOR_*" in actions:
        return actions["SENSOR_*"]
    return actions["UNKNOWN"]
