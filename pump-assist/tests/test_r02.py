import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord
from assist.config import load_machine
from assist.splits import DATA_FULL
from assist.recommend.whatif import _hyp

HAVE = (DATA_FULL / "operating_context.csv").exists()


class R02Tests(unittest.TestCase):
    def test_hyp(self):
        self.assertEqual(_hyp("clog=2.0 bearing_wear=0.6"), {"clog": 2.0, "bearing_wear": 0.6})
        self.assertEqual(_hyp("none"), {})

    @unittest.skipUnless(HAVE, "data/full missing")
    def test_machine_alert_gets_before_after_sensor_alert_none(self):
        from assist.recommend.whatif import what_if_for
        m = load_machine()
        a = AlertRecord("PUMP-08", "2025-09-10T06:12:00Z", "MACHINE", "SUCTION_STRAINER_CLOGGING", 0.8, "High")
        w = what_if_for(a, m)
        self.assertIsNotNone(w)
        self.assertIn("flow_change_pct", w)
        self.assertGreater(w["flow_change_pct"], 0)           # cleaning a clogged strainer raises flow
        self.assertLessEqual(w["risk_after"], w["risk_before"])
        self.assertTrue(w["caveats"])
        s = AlertRecord("PUMP-08", "2025-09-10T06:12:00Z", "SENSOR", "SENSOR_STUCK", 0.6, "Medium")
        self.assertIsNone(what_if_for(s, m))
        low = AlertRecord("PUMP-08", "2025-09-10T06:12:00Z", "MACHINE", "BEARING_WEAR", 0.2, "Low")
        self.assertIsNone(what_if_for(low, m))                 # Low certainty -> UNKNOWN action, no sandbox action


if __name__ == "__main__":
    unittest.main()
