import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.ingest.replay import MinuteBuffer, minute_rows
from assist.splits import DATA_FULL


def msg(ts, **kw):
    return {"topic": "t", "payload": {"timestamp_utc": ts, "operating_mode": kw.pop("mode", "RUNNING"), "quality_flags": kw.pop("flags", 0), **kw}}


class BufferTests(unittest.TestCase):
    def test_minute_only_complete_after_next_minute_starts(self):
        b = MinuteBuffer("P")
        self.assertIsNone(b.add(msg("2025-01-01T00:00:00+00:00", motor_current_a=10.0)))
        self.assertIsNone(b.add(msg("2025-01-01T00:00:30+00:00", motor_current_a=20.0, flags=2)))
        row = b.add(msg("2025-01-01T00:01:00+00:00", motor_current_a=99.0))
        self.assertEqual(row["window_start_utc"], pd.Timestamp("2025-01-01T00:00:00Z"))
        self.assertEqual(row["motor_current_a__mean"], 15.0)
        self.assertAlmostEqual(row["motor_current_a__std"], np.std([10, 20], ddof=1))
        self.assertEqual((row["motor_current_a__count"], row["rows"], row["quality_any"]), (2, 2, 2))
        self.assertEqual(b.flush()["motor_current_a__mean"], 99.0)

    def test_mode_and_running_fraction(self):
        rows = list(minute_rows([msg("2025-01-01T00:00:00+00:00", mode="STARTING"), msg("2025-01-01T00:00:01+00:00"),
                                 msg("2025-01-01T00:00:02+00:00")], "P"))
        self.assertEqual(rows[0]["operating_mode"], "RUNNING")
        self.assertAlmostEqual(rows[0]["running_fraction"], 2 / 3)


@unittest.skipUnless((DATA_FULL / "features_1min.parquet").exists(), "data/full missing")
class ReplayEqualsFileTests(unittest.TestCase):
    def test_rows_equal_i01_within_one_percent(self):
        from assist.ingest.features import load_features
        from assist.ingest.replay import replay_messages
        start, end = "2025-09-09T14:00:00Z", "2025-09-09T16:00:00Z"
        got = pd.DataFrame(list(minute_rows(replay_messages("PUMP-08", DATA_FULL, start, end), "PUMP-08")))
        ref = load_features("PUMP-08")
        ref = ref[(ref.window_start_utc >= pd.Timestamp(start)) & (ref.window_start_utc < pd.Timestamp(end))].reset_index(drop=True)
        got = got.iloc[:len(ref)].reset_index(drop=True)
        self.assertEqual(len(got), len(ref))
        self.assertTrue((got.window_start_utc.values == ref.window_start_utc.values).all())
        n_checked = 0
        for c in [c for c in got.columns if c.endswith(("__mean", "__std", "__min", "__max"))]:
            a, b = got[c].to_numpy(float), ref[c].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            tol = 0.01 * np.maximum(np.abs(b[ok]), 1e-3) + 1e-3
            self.assertTrue((np.abs(a[ok] - b[ok]) <= tol).all(), c)
            self.assertEqual(np.isnan(a).sum(), np.isnan(b).sum(), c)
            n_checked += 1
        self.assertGreater(n_checked, 20)
        for c in ("operating_mode", "rows", "quality_any"):
            self.assertTrue((got[c].astype(str).values == ref[c].astype(str).values).all(), c)


if __name__ == "__main__":
    unittest.main()
