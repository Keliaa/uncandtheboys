import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.config import Asset, load_machine
from assist.evidence.rules import classify

T1 = ["motor_current_a", "discharge_pressure_kpa", "suction_pressure_kpa", "motor_speed_rpm", "vibration_rms_mm_s",
      "vibration_accel_rms_g", "vibration_crest_factor", "bearing_temp_c"]
T0 = ["motor_current_a", "discharge_pressure_kpa", "fluid_temp_c"]


class E01Tests(unittest.TestCase):
    machine = load_machine()

    def test_strainer(self):
        z = {"suction_pressure_kpa": -12.0, "motor_speed_rpm": 9.0, "discharge_pressure_kpa": 0.5, "motor_current_a": 0.2}
        d = classify(z, Asset("P", "centrifugal_pump", "train", T1, {}), self.machine, 8.0)
        self.assertEqual(d["fault_type"], "SUCTION_STRAINER_CLOGGING")
        self.assertIn(d["certainty"], ("Medium", "High"))
        self.assertIn("strainer", d["recommended_action"])
        self.assertTrue(d["evidence"])

    def test_lone_channel_is_sensor(self):
        z = {"bearing_temp_c": 14.0, "vibration_rms_mm_s": 0.1, "vibration_accel_rms_g": 0.0, "fluid_temp_c": 0.1}
        d = classify(z, Asset("P", "centrifugal_pump", "train", T1 + ["fluid_temp_c"], {}), self.machine, 8.0)
        self.assertEqual(d["event_class"], "SENSOR")

    def test_t0_never_names_vibration_faults(self):
        z = {"vibration_accel_rms_g": 30.0, "discharge_pressure_kpa": 0.0}
        d = classify(z, Asset("P", "centrifugal_pump", "train", T0, {}), self.machine, 8.0)
        self.assertNotIn(d["fault_type"], ("BEARING_WEAR", "IMBALANCE", "MISALIGNMENT"))

    def test_nothing_matches_is_unknown(self):
        d = classify({"discharge_pressure_kpa": 0.1}, Asset("P", "centrifugal_pump", "train", T0, {}), self.machine, 8.0)
        self.assertEqual(d["fault_type"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
