import sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assist.splits import DATA_FULL
from assist.ingest.features import load_features, load_assets, stream_rows

HAVE_DATA = (DATA_FULL / "features_1min.parquet").exists()


@unittest.skipUnless(HAVE_DATA, "data/full missing")
class IngestTests(unittest.TestCase):
    def test_pump08_in_time_order(self):
        f = load_features("PUMP-08")
        self.assertEqual(set(f.asset_id), {"PUMP-08"})
        self.assertTrue(f.window_start_utc.is_monotonic_increasing)
        self.assertTrue(f.window_start_utc.is_unique)
        self.assertTrue(((f.available_utc - f.window_start_utc) == pd.Timedelta(minutes=1)).all())

    def test_split_guard(self):
        load_features("PUMP-01", splits=("train",))
        with self.assertRaises(ValueError):
            load_features("PUMP-08", splits=("train",))
        with self.assertRaises(ValueError):
            load_features(["PUMP-01", "PUMP-10"], splits=("train", "validation"))

    def test_stream_first_row(self):
        r = next(stream_rows("PUMP-09"))
        self.assertEqual(r.asset_id, "PUMP-09")
        self.assertEqual(r.available_utc, r.window_start_utc + pd.Timedelta(minutes=1))


@unittest.skipUnless(HAVE_DATA, "data/full missing")
class D01Tests(unittest.TestCase):
    def test_flags_for_validation_pumps(self):
        from assist.detect.zscore import detect
        assets = load_assets()
        f = load_features(["PUMP-08", "PUMP-09"], splits=("validation",))
        d = detect(f, assets)
        self.assertEqual(set(d.asset_id), {"PUMP-08", "PUMP-09"})
        self.assertTrue(d.flag.any() and d.sustained.any())
        self.assertTrue(d[d.sustained].channels.map(len).ge(1).all())
        self.assertTrue((d.sustained <= d.flag).all())


if __name__ == "__main__":
    unittest.main()
