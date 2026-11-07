import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord, Evidence
from assist.config import Asset, load_machine
from assist.evidence.rules import classify
from assist.evidence.sensor import refine_sensor_types, rise_minutes

N = 2000


def zframe(kind):
    t = pd.date_range("2025-01-01", periods=N, freq="1min", tz="UTC")
    z = np.zeros(N)
    if kind == "step":
        z[1200:] = 20.0
    else:                                   # ramp over 10 h
        z[1000:1600] = np.linspace(0, 20, 600)
        z[1600:] = 20.0
    return pd.DataFrame({"asset_id": "P", "window_start_utc": t, "z_ps": z})


class E03Tests(unittest.TestCase):
    machine = load_machine()
    asset = Asset("P", "centrifugal_pump", "train", ["suction_pressure_kpa", "discharge_pressure_kpa", "motor_speed_rpm", "flow_m3_h"], {})

    def test_lone_channel_is_sensor_but_coupled_response_is_machine(self):
        lone = classify({"suction_pressure_kpa": -14.0, "discharge_pressure_kpa": 0.4, "motor_speed_rpm": 0.1}, self.asset, self.machine, 8.0)
        self.assertEqual(lone["event_class"], "SENSOR")
        coupled = classify({"suction_pressure_kpa": -14.0, "discharge_pressure_kpa": -5.0, "motor_speed_rpm": 6.0}, self.asset, self.machine, 8.0)
        self.assertEqual(coupled["event_class"], "MACHINE")

    def test_rise_time_step_vs_ramp(self):
        when = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=1900)
        self.assertLessEqual(rise_minutes(zframe("step"), "P", "suction_pressure_kpa", when), 10)
        self.assertGreater(rise_minutes(zframe("ramp"), "P", "suction_pressure_kpa", when), 120)      # 0 -> 20 over 10 h: about 3.5 h from z<1 to z>8

    def test_drift_and_shift_typing(self):
        when = (pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=1900)).strftime("%Y-%m-%dT%H:%M:%SZ")
        mk = lambda: AlertRecord("P", when, "SENSOR", "SENSOR_CALIBRATION_SHIFT", 0.6, "Medium",
                                 evidence=[Evidence("suction_pressure_kpa", 20.0, "x")], z_at_alert={"suction_pressure_kpa": 20.0})
        shift, drift = mk(), mk()
        refine_sensor_types([shift], zframe("step"), self.machine)
        refine_sensor_types([drift], zframe("ramp"), self.machine)
        self.assertEqual(shift.fault_type, "SENSOR_CALIBRATION_SHIFT")
        self.assertEqual(drift.fault_type, "SENSOR_DRIFT")
        self.assertIn("slowly", drift.evidence[-1].text)


if __name__ == "__main__":
    unittest.main()
