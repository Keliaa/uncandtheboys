import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.detect.modes import cluster_regimes

N = 3000


def frame():
    rng = np.random.default_rng(0)
    t = pd.date_range("2025-01-01", periods=N, freq="1min", tz="UTC")
    duty = np.where(np.arange(N) % 600 < 300, 90.0, 40.0) + rng.normal(0, 1, N)       # two duty levels, fixed-speed pump (no speed channel)
    f = pd.DataFrame({"asset_id": "P", "window_start_utc": t, "steady": True, "fluid_temp_c__mean": 25 + rng.normal(0, 0.1, N)})
    ctx = pd.DataFrame({"asset_id": "P", "window_start_utc": t, "duty_demand_pct": duty, "sea_temp_c": 12.0})
    return f, ctx, duty


class D05Tests(unittest.TestCase):
    def test_fixed_speed_pump_gets_modes_that_follow_the_duty(self):
        f, ctx, duty = frame()
        for method in ("kmeans", "gmm"):
            lab = cluster_regimes(f, ctx, method)
            self.assertGreaterEqual(lab[lab >= 0].nunique(), 2)
            hi, lo = set(lab[duty > 65]), set(lab[duty < 65])
            self.assertFalse(hi & lo, method)                    # high and low duty never share a mode

    def test_unknown_conditions_and_bad_method(self):
        f, ctx, _ = frame()
        ctx.loc[ctx.index[:50], "duty_demand_pct"] = np.nan      # unknown duty -> no mode (-1), others unaffected
        lab = cluster_regimes(f, ctx, "kmeans")
        self.assertTrue((lab.iloc[:50] == -1).all())
        with self.assertRaises(ValueError):
            cluster_regimes(f, ctx, "dbscan")


if __name__ == "__main__":
    unittest.main()
