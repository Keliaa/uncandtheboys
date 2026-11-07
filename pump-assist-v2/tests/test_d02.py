import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.detect.regression import ewma_scores


def frame(n=3000, step_at=None):
    rng = np.random.default_rng(0)
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    speed = 0.8 + 0.1 * np.sin(np.arange(n) / 200)
    duty = 60 + 20 * np.sin(np.arange(n) / 200)
    pdk = 300 * speed ** 2 + 0.2 * duty + rng.normal(0, 1, n)       # expected value follows speed and duty
    if step_at:
        pdk[step_at:] -= 15
    z = pd.DataFrame({"asset_id": "P", "window_start_utc": t, "speed_ratio": speed, "steady": True,
                      "discharge_pressure_kpa__mean": pdk})
    ctx = pd.DataFrame({"asset_id": "P", "window_start_utc": t, "duty_demand_pct": duty, "sea_temp_c": 10.0})
    return z, ctx


class D02Tests(unittest.TestCase):
    def test_healthy_stays_low_and_step_is_found(self):
        z, c = frame()
        healthy = ewma_scores(z, c).d02_score
        z2, c2 = frame(step_at=2000)
        faulty = ewma_scores(z2, c2).d02_score
        self.assertLess(healthy.iloc[1500:].max(), 4.0)
        self.assertGreater(faulty.iloc[2100:].min(), 5.0)
        self.assertLess(faulty.iloc[1500:1990].max(), 4.0)


if __name__ == "__main__":
    unittest.main()
