import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord
from assist.config import load_machine
from assist.detect.orders import add_order_evidence, classify_pattern, order_amplitudes, spectral_typing

FS, RPM = 10240.0, 1200.0


def tone(a1=0.1, a2=0.1, a3=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(10240) / FS
    f1 = RPM / 60
    return a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * 2 * f1 * t) + a3 * np.sin(2 * np.pi * 3 * f1 * t) + rng.normal(0, 0.01, len(t))


class V02Tests(unittest.TestCase):
    def test_order_amplitudes_read_the_right_orders(self):
        a = order_amplitudes(tone(0.4, 0.2, 0.1), FS, RPM, 6)
        self.assertAlmostEqual(a["a1"], 0.4, delta=0.05)
        self.assertAlmostEqual(a["a2"], 0.2, delta=0.03)
        self.assertAlmostEqual(a["a3"], 0.1, delta=0.02)
        self.assertLess(a["avp"], 0.05)               # nothing at the vane-pass order (6x)

    def test_pattern(self):
        g = pd.DataFrame({"m1": [1.0, 3.0, 2.0, 1.0], "m2": [1.0, 1.0, 7.0, 1.2]})
        self.assertEqual(classify_pattern(g).tolist(), ["none", "imbalance", "misalignment", "none"])

    def test_evidence_and_typing(self):
        t = pd.date_range("2025-01-01", periods=10, freq="4h", tz="UTC")
        o = pd.DataFrame({"asset_id": "P", "timestamp_utc": t, "has_baseline": [False] * 6 + [True] * 4,
                          "m1": 3.0, "m2": 1.0, "m3": 1.0, "pattern": ["none"] * 6 + ["imbalance"] * 4})
        when = (t[-1] + pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        imb = AlertRecord("P", when, "MACHINE", "IMBALANCE", 0.8, "High")
        add_order_evidence([imb], o)
        self.assertIn("1x 3.0", imb.evidence[-1].text)
        bearing = AlertRecord("P", when, "MACHINE", "BEARING_WEAR", 0.5, "Medium")
        strainer = AlertRecord("P", when, "MACHINE", "SUCTION_STRAINER_CLOGGING", 0.8, "High")
        sensor = AlertRecord("P", when, "SENSOR", "SENSOR_STUCK", 0.6, "Medium")
        spectral_typing([bearing, strainer, sensor], o, load_machine())
        self.assertEqual(bearing.fault_type, "IMBALANCE")             # spectrum contradicts bearing wear
        self.assertEqual(strainer.fault_type, "SUCTION_STRAINER_CLOGGING")   # not a vibration type: untouched
        self.assertEqual(sensor.fault_type, "SENSOR_STUCK")


if __name__ == "__main__":
    unittest.main()
