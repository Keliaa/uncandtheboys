import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.evidence.bayes import GaussianNB
from assist.evidence.rules import apply_classifier
from assist.config import Asset, load_machine

CH = ["suction_pressure_kpa", "motor_speed_rpm", "vibration_accel_rms_g"]


class E02Tests(unittest.TestCase):
    def model(self):
        rows = [{"suction_pressure_kpa": -10 + i % 3, "motor_speed_rpm": 8, "vibration_accel_rms_g": 0} for i in range(30)]
        rows += [{"suction_pressure_kpa": 0, "motor_speed_rpm": 0, "vibration_accel_rms_g": 12 + i % 3} for i in range(30)]
        return GaussianNB(CH).fit(rows, ["SUCTION_STRAINER_CLOGGING"] * 30 + ["BEARING_WEAR"] * 30)

    def test_probabilities(self):
        m = self.model()
        p = m.predict_proba({"suction_pressure_kpa": -11, "motor_speed_rpm": 9, "vibration_accel_rms_g": 0.5})
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertGreater(p["SUCTION_STRAINER_CLOGGING"], 0.9)

    def test_missing_channel_is_marginalised(self):
        m = self.model()
        p = m.predict_proba({"vibration_accel_rms_g": 13})            # other channels absent
        self.assertGreater(p["BEARING_WEAR"], 0.9)
        self.assertEqual(m.predict_proba({}), {})

    def test_temperature_flattens(self):
        m = self.model()
        z = {"suction_pressure_kpa": -11, "motor_speed_rpm": 9, "vibration_accel_rms_g": 0.5}
        self.assertLess(max(m.predict_proba_tempered(z, 50).values()), max(m.predict_proba(z).values()))

    def test_t0_pump_cannot_get_vibration_type(self):
        machine = load_machine()
        asset = Asset("P", "centrifugal_pump", "train", ["motor_current_a", "discharge_pressure_kpa"], {})
        d = apply_classifier({"event_class": "MACHINE"}, {"BEARING_WEAR": 0.9, "IMPELLER_WEAR": 0.1}, asset, machine)
        self.assertEqual([t for t, _ in d["top_types"]], ["IMPELLER_WEAR"])   # BEARING_WEAR needs a vibration sensor

    def test_gate_removes_type_whose_symptoms_do_not_match(self):
        machine = load_machine()
        asset = Asset("P", "centrifugal_pump", "train",
                      ["suction_pressure_kpa", "motor_speed_rpm", "vibration_accel_rms_g", "vibration_rms_mm_s"], {})
        z = {"suction_pressure_kpa": -14.0, "motor_speed_rpm": 0.5, "vibration_accel_rms_g": 0.2}
        probs = {"BEARING_WEAR": 0.86, "SUCTION_STRAINER_CLOGGING": 0.14}
        base = {"event_class": "MACHINE"}
        free = apply_classifier(base, probs, asset, machine)
        gated = apply_classifier(base, probs, asset, machine, z, 8.0, gate=True)
        self.assertEqual(free["top_types"][0][0], "BEARING_WEAR")      # without the gate: evidence contradicts the type
        self.assertEqual(gated["top_types"][0][0], "SUCTION_STRAINER_CLOGGING")
        none_match = apply_classifier({"event_class": "MACHINE", "fault_type": "UNKNOWN"}, {"BEARING_WEAR": 1.0}, asset, machine,
                                      {"suction_pressure_kpa": -14.0}, 8.0, gate=True)
        self.assertEqual(none_match["fault_type"], "UNKNOWN")          # nothing consistent: keep the E01 answer


if __name__ == "__main__":
    unittest.main()
