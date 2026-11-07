import csv, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assist import config as C

T0 = ["motor_current_a", "discharge_pressure_kpa", "fluid_temp_c", "ambient_temp_c"]
T1 = T0 + ["suction_pressure_kpa", "vibration_rms_mm_s", "vibration_accel_rms_g", "vibration_crest_factor", "bearing_temp_c"]
FIELDS = ["asset_id", "split", "profile", "instrumentation_tier", "pump_rated_flow_m3_h", "commissioned_year",
          "channels_available"]


def make_csv(path):
    rows = [dict(asset_id="PUMP-01", split="train", profile="SHIP_SW_COOLING", instrumentation_tier="T1_RETROFIT",
                 pump_rated_flow_m3_h="250.5", commissioned_year="2012", channels_available="|".join(T1)),
            dict(asset_id="PUMP-99", split="validation", profile="INDUSTRIAL_CW", instrumentation_tier="T0_LEGACY",
                 pump_rated_flow_m3_h="80", commissioned_year="2005", channels_available="|".join(T0))]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)


class ConfigTests(unittest.TestCase):
    def test_machine_file_complete(self):
        m = C.load_machine()
        for ft, spec in m["symptoms"].items():
            self.assertIn(spec["class"], ("MACHINE", "SENSOR"))
            self.assertTrue(spec["evidence"])
            for e in spec["evidence"]:
                self.assertTrue(e["channel"] == "any" or e["channel"] in m["channels"], (ft, e))
            self.assertEqual(C.action_for(m, ft)["text"] != "", True)
        self.assertIn("UNKNOWN", m["actions"])

    def test_actions_fallback(self):
        m = C.load_machine()
        self.assertIn("transmitter", C.action_for(m, "SENSOR_STUCK")["text"])
        self.assertIn("gateway", C.action_for(m, "COMMUNICATION_LOSS")["text"])
        self.assertIn("Inspect the pump", C.action_for(m, "WHATEVER")["text"])

    def test_new_asset_needs_only_a_file(self):
        with tempfile.TemporaryDirectory() as t:
            csv_path = Path(t) / "assets.csv"; make_csv(csv_path)
            paths = C.write_asset_files(csv_path, Path(t) / "assets")
            self.assertEqual(len(paths), 2)
            m = C.load_machine()
            t1, t0 = C.load_asset("PUMP-01", t), C.load_asset("PUMP-99", t)
            self.assertEqual(t1.nameplate["pump_rated_flow_m3_h"], 250.5)
            self.assertEqual(t1.nameplate["commissioned_year"], 2012)
            s0, s1 = C.usable_symptoms(m, t0), C.usable_symptoms(m, t1)
            self.assertNotIn("BEARING_WEAR", s0)           # T0: no vibration sensor
            self.assertNotIn("CAVITATION", s0)
            self.assertIn("DISCHARGE_RESTRICTION", s0)
            self.assertIn("BEARING_WEAR", s1)
            self.assertNotIn("flow_m3_h", [e["channel"] for e in s1["DISCHARGE_RESTRICTION"]])

    def test_load_asset_from_dir(self):
        with tempfile.TemporaryDirectory() as t:
            csv_path = Path(t) / "assets.csv"; make_csv(csv_path)
            C.write_asset_files(csv_path, Path(t) / "assets")
            a = C.load_asset("PUMP-99", t)
            self.assertEqual((a.split, a.machine_type), ("validation", "centrifugal_pump"))


if __name__ == "__main__":
    unittest.main()
