import sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.splits import DATA_FULL

HAVE = (DATA_FULL / "features_1min.parquet").exists() and (Path(__file__).resolve().parents[2] / "physics-context-detector" / "b4.py").exists()


@unittest.skipUnless(HAVE, "data/full or physics-context-detector missing")
class EngineTests(unittest.TestCase):
    def test_b4_engine_is_b4_and_v2_engine_is_v2(self):
        from assist.pipeline import run_pipeline
        det, a4 = run_pipeline(["PUMP-08"], ("validation",), engine="b4")
        self.assertTrue(a4 and all(a.source.startswith("b4:") for a in a4))
        self.assertTrue({"asset_id", "window_start_utc", "score"} <= set(det.columns) and det.score.notna().all())
        _, a2 = run_pipeline(["PUMP-08"], ("validation",), engine="v2")
        self.assertTrue(a2 and not any(a.source.startswith("b4:") for a in a2))
        # the b4 engine returns exactly the alerts of B4 itself
        from assist.engines.b4_engine import _b4
        from assist.splits import DATA_FULL as D
        b4 = _b4()
        pred = b4.run_b4(b4.load_inputs(str(D), ["PUMP-08"]))
        self.assertEqual(sorted((a.alert_time_utc, a.fault_type) for a in a4), sorted(zip(pred.alert_time_utc, pred.fault_type)))

    def test_b4_engine_keeps_the_split_guard_and_rejects_replayed_rows(self):
        from assist.pipeline import run_pipeline
        with self.assertRaises(ValueError):
            run_pipeline(["PUMP-10"], ("validation",), engine="b4")                      # a test pump outside the allowed splits
        with self.assertRaises(ValueError):
            run_pipeline(["PUMP-08"], ("validation",), engine="b4", features=pd.DataFrame({"asset_id": ["PUMP-08"]}))

    def test_params_override_reaches_b4(self):
        from assist.engines.b4_engine import b4_params
        self.assertEqual(b4_params({"params": {"stable_min": 5}})["stable_min"], 5)
        self.assertEqual(b4_params({"params": {"rx_env": [0.0, 0.05]}})["rx_env"], (0.0, 0.05))     # lists become tuples where B4 expects them


if __name__ == "__main__":
    unittest.main()
