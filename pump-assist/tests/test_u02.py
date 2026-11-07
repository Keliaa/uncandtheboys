import re, sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord, Evidence
from assist.config import Asset
from assist.report import pump_report


class ReportTests(unittest.TestCase):
    def test_self_contained_html(self):
        t = pd.date_range("2025-01-01", periods=200, freq="1min", tz="UTC")
        feats = pd.DataFrame({"asset_id": "P-1", "window_start_utc": t, "discharge_pressure_kpa__mean": 300.0})
        det = pd.DataFrame({"asset_id": "P-1", "window_start_utc": t, "score": 1.0})
        a = AlertRecord("P-1", "2025-01-01T01:00:00Z", "MACHINE", "SUCTION_STRAINER_CLOGGING", 0.8, "High",
                        top_types=[("SUCTION_STRAINER_CLOGGING", 0.8), ("CAVITATION", 0.2)],
                        evidence=[Evidence("suction_pressure_kpa", -9.0, "suction pressure <low>")],
                        recommended_action="Inspect and clean the suction strainer.")
        asset = Asset("P-1", "centrifugal_pump", "validation", ["discharge_pressure_kpa"], {"site": "S"})
        h = pump_report("P-1", feats, det, [a], asset, {"suction_pressure_kpa": 0.5}, "now")
        self.assertIn("data:image/png;base64,", h)
        self.assertIsNone(re.search(r"(src|href)=[\"']https?://", h))      # no internet needed
        self.assertNotIn("<button", h)                                      # no buttons
        self.assertIn("suction pressure &lt;low&gt;", h)                    # text is escaped
        self.assertIn("CAVITATION 20%", h)
        self.assertIn("no vibration sensor", h)
        self.assertEqual(h.count("SUCTION_STRAINER_CLOGGING 80%"), 0)       # top cause is not repeated as "other"


if __name__ == "__main__":
    unittest.main()
