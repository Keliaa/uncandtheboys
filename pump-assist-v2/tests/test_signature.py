import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.evidence.signature import LearnedSignatures

CH = ["suction_pressure_kpa", "vibration_accel_rms_g", "motor_current_a"]


def data():
    rows = [{"suction_pressure_kpa": -6 - i % 7, "motor_current_a": 1.0} for i in range(40)]
    rows += [{"vibration_accel_rms_g": 6 + i % 9, "suction_pressure_kpa": 0.5} for i in range(40)]
    return rows, ["SUCTION_STRAINER_CLOGGING"] * 40 + ["BEARING_WEAR"] * 40


class SignatureTests(unittest.TestCase):
    def test_all_methods_separate_two_signatures_and_ignore_magnitude(self):
        rows, y = data()
        for m in ("cosine", "knn", "logreg"):
            s = LearnedSignatures(CH, m).fit(rows, y)
            for z, want in (({"suction_pressure_kpa": -25.0}, "SUCTION_STRAINER_CLOGGING"), ({"suction_pressure_kpa": -5.0}, "SUCTION_STRAINER_CLOGGING"),
                            ({"vibration_accel_rms_g": 20.0}, "BEARING_WEAR")):
                p = s.predict_proba(z)
                self.assertEqual(max(p.items(), key=lambda kv: kv[1])[0], want, (m, z))
                self.assertAlmostEqual(sum(p.values()), 1.0)

    def test_empty_vector_and_bad_method(self):
        rows, y = data()
        self.assertEqual(LearnedSignatures(CH, "logreg").fit(rows, y).predict_proba({}), {})
        with self.assertRaises(ValueError):
            LearnedSignatures(CH, "svm")


if __name__ == "__main__":
    unittest.main()
