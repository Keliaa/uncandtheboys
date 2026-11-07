"""Measurement names used by the operator interface.

`B4_NAME` / `B4_UNIT` mirror the strings B4 writes into its evidence text (physics-context-detector/b4.py NAME and
UNIT). They are duplicated here so the presentation never imports the detector; a test checks they stay identical.
"""
from __future__ import annotations

# key -> (display name, display unit)
CHANNELS = {
    "v": ("Vibration velocity", "mm/s"),
    "a": ("Vibration acceleration", "g"),
    "c": ("Vibration crest factor", ""),
    "bt": ("Bearing temperature", "°C"),
    "ps": ("Suction pressure", "kPa"),
    "pd": ("Discharge pressure", "kPa"),
    "i": ("Motor current", "A"),
    "pw": ("Motor power", "kW"),
    "q": ("Flow", "m³/h"),
    "n": ("Motor speed", "rpm"),
    "wt": ("Winding temperature", "°C"),
    "tf": ("Fluid temperature", "°C"),
    "vp": ("Valve position", "%"),
}

# telemetry column names (data dictionary) per key
COLUMNS = {"v": "vibration_rms_mm_s", "a": "vibration_accel_rms_g", "c": "vibration_crest_factor",
           "bt": "bearing_temp_c", "ps": "suction_pressure_kpa", "pd": "discharge_pressure_kpa",
           "i": "motor_current_a", "pw": "motor_power_kw", "q": "flow_m3_h", "n": "motor_speed_rpm",
           "wt": "motor_winding_temp_c", "tf": "fluid_temp_c", "vp": "valve_position_pct"}

B4_NAME = {"v": "vibration velocity", "a": "vibration acceleration", "c": "vibration crest factor",
           "bt": "bearing temperature", "ps": "suction pressure", "pd": "discharge pressure", "i": "motor current",
           "pw": "motor power", "q": "flow", "n": "motor speed", "wt": "winding temperature",
           "tf": "fluid temperature", "vp": "valve position", "rx": "speed vs physics-required speed",
           "hr": "head vs pump curve"}
B4_UNIT = {"v": "mm/s", "a": "g", "c": "", "bt": "degC", "ps": "kPa", "pd": "kPa", "i": "A", "pw": "kW", "q": "m3/h",
           "n": "rpm", "wt": "degC", "tf": "degC", "vp": "%"}

KEY_BY_B4_NAME = {v: k for k, v in B4_NAME.items()}
DISPLAY_UNIT = {"degC": "°C", "m3/h": "m³/h"}


def name(key: str) -> str:
    return CHANNELS[key][0] if key in CHANNELS else B4_NAME.get(key, key)


def unit(key: str) -> str:
    return CHANNELS[key][1] if key in CHANNELS else ""


def lower_name(key: str) -> str:
    """Name for use inside a sentence ('motor current')."""
    return name(key)[0].lower() + name(key)[1:]


def compound(key: str) -> str:
    """Hyphenated modifier for titles ('Verify the motor-current reading')."""
    return lower_name(key).replace(" ", "-")


def display_unit(b4_unit: str) -> str:
    return DISPLAY_UNIT.get(b4_unit, b4_unit)
