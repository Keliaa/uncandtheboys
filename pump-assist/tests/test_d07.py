import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.detect.autoencoder import autoencoder_flags, fit_tier_autoencoder
from assist.detect.pca import healthy_mask


def frame(n=4000, seed=0, break_corr=False):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = a + rng.normal(0, 0.15, n)
    c = rng.normal(0, 1, n)
    if break_corr:
        b = -b
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"asset_id": "P", "window_start_utc": t, "steady": True, "gap": False, "z_pd": a, "z_ps": b, "z_i": c})


class D07Tests(unittest.TestCase):
    def test_unusual_combination_flagged_healthy_quiet(self):
        z = frame()
        m = fit_tier_autoencoder("T", z, healthy_mask(z, None), ["P"], 0.99)
        assets = pd.DataFrame({"asset_id": ["P"], "instrumentation_tier": ["T"]})
        ok = autoencoder_flags(frame(seed=1), assets, {"T": m})
        bad = autoencoder_flags(frame(seed=1, break_corr=True), assets, {"T": m})
        self.assertLess(ok.flag.mean(), 0.05)
        self.assertGreater(bad.flag.mean(), 0.4)
        self.assertEqual(len(bad[bad.flag].top.iloc[0]), 3)
        self.assertTrue(m.limit > 0)

    def test_missing_channel_is_zero_not_nan(self):
        z = frame()
        z.loc[::3, "z_i"] = np.nan
        m = fit_tier_autoencoder("T", z, healthy_mask(z, None), ["P"])
        assets = pd.DataFrame({"asset_id": ["P"], "instrumentation_tier": ["T"]})
        self.assertTrue(np.isfinite(autoencoder_flags(z, assets, {"T": m}).err).all())


if __name__ == "__main__":
    unittest.main()
