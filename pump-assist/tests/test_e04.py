import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.evidence.tree import FaultTree

CH = ["suction_pressure_kpa", "vibration_accel_rms_g", "motor_current_a"]


class E04Tests(unittest.TestCase):
    def data(self):
        rows = [{"suction_pressure_kpa": -12 + i % 3, "vibration_accel_rms_g": 0.0} for i in range(40)]
        rows += [{"suction_pressure_kpa": 0.0, "vibration_accel_rms_g": 14 + i % 3} for i in range(40)]
        return rows, ["SUCTION_STRAINER_CLOGGING"] * 40 + ["BEARING_WEAR"] * 40

    def test_depth_limit_and_prediction(self):
        with self.assertRaises(ValueError):
            FaultTree(CH, max_depth=4)
        rows, y = self.data()
        t = FaultTree(CH, 3).fit(rows, y)
        self.assertLessEqual(t.tree.get_depth(), 3)
        self.assertEqual(max(t.predict_proba({"suction_pressure_kpa": -13}).items(), key=lambda kv: kv[1])[0], "SUCTION_STRAINER_CLOGGING")
        self.assertEqual(max(t.predict_proba({"vibration_accel_rms_g": 15}).items(), key=lambda kv: kv[1])[0], "BEARING_WEAR")
        self.assertAlmostEqual(sum(t.predict_proba({}).values()), 1.0)          # missing channels count as no deviation

    def test_rules_and_leaves(self):
        rows, y = self.data()
        t = FaultTree(CH, 3).fit(rows, y)
        self.assertIn("class:", t.rules())
        leaves = t.leaves()
        self.assertEqual(sum(l["n"] for l in leaves), 80)
        self.assertTrue(all(l["channels"] <= set(CH) for l in leaves))


if __name__ == "__main__":
    unittest.main()
