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

    def test_b4_engine_keeps_the_split_guard(self):
        from assist.pipeline import run_pipeline
        with self.assertRaises(ValueError):
            run_pipeline(["PUMP-10"], ("validation",), engine="b4")                      # a test pump outside the allowed splits (shared load stage)
        with self.assertRaises(ValueError):
            run_pipeline(["PUMP-10"], ("validation",), engine="v2")

    def test_b4_takes_replayed_rows_and_a_channel_filter_like_v2_does(self):
        """B4 reads the pipeline's shared inputs, so a truncated (replayed) feature table works and B4's alerts before the cut equal the full run."""
        from assist.ingest.features import load_features
        from assist.pipeline import run_pipeline
        feats = load_features("PUMP-08", splits=("validation",))
        _, full = run_pipeline(["PUMP-08"], ("validation",), engine="b4", features=feats)
        cut = pd.Timestamp("2025-09-10T20:00:00Z")
        _, part = run_pipeline(["PUMP-08"], ("validation",), engine="b4", features=feats[feats.window_start_utc < cut])
        key = lambda al: sorted((a.alert_time_utc, a.fault_type) for a in al if pd.Timestamp(a.alert_time_utc) <= cut)
        self.assertTrue(key(full))
        self.assertEqual(key(full), key(part))                    # no alert from rows that had not arrived, none lost
        t0 = {"ambient_temp_c", "discharge_pressure_kpa", "fluid_temp_c", "motor_current_a", "motor_speed_rpm", "motor_power_kw"}
        _, legacy = run_pipeline(["PUMP-09"], ("validation",), engine="b4", channel_filter=lambda a: t0)
        self.assertFalse([a for a in legacy if a.fault_type in ("BEARING_WEAR", "IMBALANCE", "MISALIGNMENT")])   # no vibration channel left

    def test_shared_stages_run_for_b4_too(self):
        from assist.pipeline import run_pipeline
        _, plain = run_pipeline(["PUMP-08"], ("validation",), engine="b4")
        _, rich = run_pipeline(["PUMP-08"], ("validation",), engine="b4", settings={"shared": {"vibration_evidence": True, "what_if": True}})
        self.assertEqual([(a.alert_time_utc, a.fault_type) for a in plain], [(a.alert_time_utc, a.fault_type) for a in rich])   # same alerts
        bearing = [a for a in rich if a.fault_type == "BEARING_WEAR"]
        self.assertTrue(bearing and any("Vibration snapshot" in e.text or "Envelope spectrum" in e.text for a in bearing for e in a.evidence))
        self.assertTrue(any(a.what_if for a in rich if a.event_class == "MACHINE"))                  # R02 on B4's alerts
        self.assertTrue(all(a.what_if is None for a in plain))

    def test_params_override_reaches_b4(self):
        from assist.engines.b4_engine import b4_params
        self.assertEqual(b4_params({"params": {"stable_min": 5}})["stable_min"], 5)
        self.assertEqual(b4_params({"params": {"rx_env": [0.0, 0.05]}})["rx_env"], (0.0, 0.05))     # lists become tuples where B4 expects them


if __name__ == "__main__":
    unittest.main()
