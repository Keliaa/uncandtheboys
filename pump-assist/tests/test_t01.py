import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.detect.baseline import adaptive_z

N = 6 * 1440


def frame(drift_kpa_day=0.0, ramp_from_day=None, ramp_kpa_day=0.0, seed=0, trust_ok=True):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=N, freq="1min", tz="UTC")
    d = np.arange(N) / 1440.0
    x = 300 + drift_kpa_day * d + rng.normal(0, 1.0, N)
    if ramp_from_day is not None:
        x += ramp_kpa_day * np.clip(d - ramp_from_day, 0, None)
    z = pd.DataFrame({"asset_id": "P", "window_start_utc": t, "steady": True, "regime": 10,
                      "discharge_pressure_kpa__mean": x, "z_pd": 0.0})
    tr = pd.DataFrame({"trust_pd": 1.0 if trust_ok else 0.0}, index=z.index)
    return z, tr


def static(z, tr):
    """The commissioning baseline without updates (what D01 uses), computed with the same code path."""
    return adaptive_z(z, "always", tau_min=1e15, trust=tr)


def zmax(z, lo_day, hi_day):
    s = z.z_pd.to_numpy()[int(lo_day * 1440):int(hi_day * 1440)]
    return float(np.nanmax(np.abs(s)))


class T01Tests(unittest.TestCase):
    def test_benign_aging_static_alerts_guarded_tracks(self):
        z, tr = frame(drift_kpa_day=4.0)
        st = static(z, tr)
        ga = adaptive_z(z, "guarded", trust=tr)
        al = adaptive_z(z, "always", trust=tr)
        self.assertGreater(zmax(st, 4, 6), 8.0)         # static baseline: false alert territory after a few days
        self.assertLess(zmax(ga, 1, 6), 4.0)            # guarded follows slow ageing ...
        self.assertLess(zmax(al, 1, 6), 4.0)            # ... and so does always-learning

    def test_fault_ramp_guarded_keeps_it_always_absorbs_it(self):
        z, tr = frame(ramp_from_day=2.0, ramp_kpa_day=30.0)
        ga = adaptive_z(z, "guarded", trust=tr)
        al = adaptive_z(z, "always", trust=tr)
        self.assertGreater(zmax(ga, 3, 6), 8.0)          # the fault stays visible
        self.assertLess(zmax(al, 3, 6), zmax(ga, 3, 6) / 2)   # always-learning learns the fault
        # the guard froze the baseline once the deviation started
        self.assertGreater(ga.z_pd.iloc[-1], 40.0)

    def test_untrusted_channel_blocks_update(self):
        z, tr = frame(drift_kpa_day=4.0, trust_ok=False)
        ga = adaptive_z(z, "guarded", trust=tr)
        st = static(z, tr)
        np.testing.assert_allclose(ga.z_pd.to_numpy(), st.z_pd.to_numpy(), atol=1e-9)    # never updated

    def test_bad_mode(self):
        z, tr = frame()
        with self.assertRaises(ValueError):
            adaptive_z(z, "sometimes", trust=tr)


if __name__ == "__main__":
    unittest.main()
