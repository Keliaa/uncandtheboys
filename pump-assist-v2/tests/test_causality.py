"""Causality: an alert at time t may use only rows available at or before t (B4 idea #10).

For random cut times, run the system on the data truncated at the cut and on the full data; the alerts up to the cut must be identical
(asset, time, class, type, confidence). A leak (any use of later rows) changes some of them."""
import sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.splits import DATA_FULL

CUTS = ["2025-09-09T18:00:00Z", "2025-09-10T20:00:00Z", "2025-09-11T11:30:00Z"]       # PUMP-08, after its first-day baseline
KEY = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence"]


def key_rows(alerts, cut):
    rows = [(a.asset_id, a.alert_time_utc, a.event_class, a.fault_type, round(a.probability, 4)) for a in alerts if pd.Timestamp(a.alert_time_utc) <= cut]
    return sorted(rows)


@unittest.skipUnless((DATA_FULL / "features_1min.parquet").exists(), "data/full missing")
class CausalityTests(unittest.TestCase):
    def test_alerts_before_a_cut_do_not_depend_on_later_rows(self):
        from assist.ingest.features import load_features
        from assist.pipeline import run_pipeline
        feats = load_features("PUMP-08", splits=("validation",))
        _, full = run_pipeline(["PUMP-08"], ("validation",), features=feats)
        n_checked = 0
        for c in CUTS:
            cut = pd.Timestamp(c)
            part = feats[feats.window_start_utc < cut]
            _, early = run_pipeline(["PUMP-08"], ("validation",), features=part)
            a, b = key_rows(full, cut), key_rows(early, cut)
            self.assertEqual(a, b, f"alerts before {c} differ between full and truncated data")
            n_checked += len(a)
        self.assertGreater(n_checked, 0)          # the cuts must actually cover some alerts, or the test proves nothing


if __name__ == "__main__":
    unittest.main()
