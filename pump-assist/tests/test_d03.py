import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.detect.pca import fit_tier_model, pca_flags, healthy_mask


def frame(n=4000, seed=0, break_corr=False):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = a + rng.normal(0, 0.15, n)          # two channels that normally move together
    c = rng.normal(0, 1, n)
    if break_corr:
        b = -b                               # each channel alone stays in range, the combination is unusual
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"asset_id": "P", "window_start_utc": t, "steady": True, "gap": False,
                         "z_pd": a, "z_ps": b, "z_i": c})


class D03Tests(unittest.TestCase):
    def test_unusual_combination_and_channel_share(self):
        z = frame()
        h = healthy_mask(z, None)
        m = fit_tier_model("T", z, h, ["P"], 0.999)
        self.assertEqual(sorted(m.channels), ["i", "pd", "ps"])
        assets = pd.DataFrame({"asset_id": ["P"], "instrumentation_tier": ["T"]})
        ok = pca_flags(frame(seed=1), assets, {"T": m})
        bad = pca_flags(frame(seed=1, break_corr=True), assets, {"T": m})
        self.assertLess(ok.flag.mean(), 0.05)
        self.assertGreater(bad.flag.mean(), 0.5)
        top = bad[bad.flag].top.iloc[0]
        self.assertEqual(len(top), 3)

    def test_missing_channel_counts_as_no_deviation(self):
        z = frame()
        z.loc[::3, "z_i"] = np.nan
        m = fit_tier_model("T", z, healthy_mask(z, None), ["P"])
        assets = pd.DataFrame({"asset_id": ["P"], "instrumentation_tier": ["T"]})
        self.assertTrue(np.isfinite(pca_flags(z, assets, {"T": m}).t2).all())


if __name__ == "__main__":
    unittest.main()
