import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.splits import DATA_FULL

HAVE = (DATA_FULL / "features_1min.parquet").exists()


@unittest.skipUnless(HAVE, "data/full missing")
class D04Tests(unittest.TestCase):
    def test_healthy_pump_near_zero_and_worn_pump_negative(self):
        from assist.config import load_machine
        from assist.detect.physics import head_residual, detect_physics
        from assist.ingest.features import load_assets, load_features
        a, m = load_assets(), load_machine()
        f = load_features(["PUMP-08", "PUMP-05"], splits=("train", "validation"))
        r = head_residual(f, a, m)
        med = r.groupby("asset_id").residual_pct.median()
        self.assertLess(abs(med["PUMP-08"]), 2.0)      # healthy VFD pump sits on its curve
        self.assertLess(med["PUMP-05"], med["PUMP-08"])  # impeller wear: head below the curve
        d = detect_physics(f, a, m, limit_pct=2.0, persist=30)
        self.assertTrue(d[d.asset_id == "PUMP-05"].sustained.any())

    def test_physics_section_in_machine_file(self):
        from assist.config import load_machine
        self.assertIn("physics", load_machine())


if __name__ == "__main__":
    unittest.main()
