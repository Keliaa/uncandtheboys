import sys, tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord
from assist.detect.vibration import add_snapshot_evidence, snapshot_stats


def make_data(d: Path, wear_from=12, n=20):
    rng = np.random.default_rng(0)
    pd.DataFrame({"asset_id": ["P-1"], "split": ["validation"]}).to_csv(d / "assets.csv", index=False)
    rows = []
    for i in range(n):
        x = rng.normal(0, 1.0, 10240)
        if i >= wear_from:                                   # growing impacts: higher RMS, crest factor and kurtosis
            x = x * (1 + 0.2 * (i - wear_from)) + (rng.random(10240) < 0.002) * rng.normal(0, 12, 10240)
        rows.append({"asset_id": "P-1", "snapshot_id": f"S{i}", "timestamp_utc": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=4 * i),
                     "sample_rate_hz": 10240, "n_samples": 10240, "axis": "radial", "unit": "g", "rpm_at_capture": 1000.0, "samples": x.tolist()})
    pd.DataFrame(rows).to_parquet(d / "vibration_snapshots.parquet")


class V01Tests(unittest.TestCase):
    def test_trend_and_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t); make_data(d)
            s = snapshot_stats("P-1", d, ("validation",))
            self.assertFalse(s.has_baseline.iloc[:6].any())
            self.assertFalse(s.changed.iloc[6:12].any())              # healthy part: no change flagged
            self.assertTrue(s.changed.iloc[-3:].all())                # wear part: flagged
            self.assertGreater(s.rms_ratio.iloc[-1], 2.0)
            self.assertGreater(s.kurt_delta.iloc[-1], s.kurt_delta.iloc[8] + 1)
            late = s.timestamp_utc.iloc[-1] + pd.Timedelta(minutes=5)
            a = AlertRecord("P-1", late.strftime("%Y-%m-%dT%H:%M:%SZ"), "MACHINE", "BEARING_WEAR", 0.8, "High")
            other = AlertRecord("P-1", late.strftime("%Y-%m-%dT%H:%M:%SZ"), "MACHINE", "IMBALANCE", 0.8, "High")
            add_snapshot_evidence([a, other], s)
            self.assertIn("RMS", a.evidence[-1].text)
            self.assertEqual(other.evidence, [])                      # only BEARING_WEAR gets this evidence
            early = AlertRecord("P-1", "2025-01-01T01:00:00Z", "MACHINE", "BEARING_WEAR", 0.8, "High")
            add_snapshot_evidence([early], s)
            self.assertEqual(early.evidence, [])                      # no snapshot from the future, no baseline yet

    def test_split_guard(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t); make_data(d)
            with self.assertRaises(ValueError):
                snapshot_stats("P-1", d, ("train",))


if __name__ == "__main__":
    unittest.main()
