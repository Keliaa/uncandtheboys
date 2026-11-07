import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord, Evidence
from assist.evidence.changepoint import add_change_times, change_time

N = 1500


def frame(step_at=1000, ramp=False, pre_existing=False, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=N, freq="1min", tz="UTC")
    z = rng.normal(0, 1, N)
    if pre_existing:
        z += 15
    elif ramp:
        z[step_at:] += np.linspace(0, 20, N - step_at)
    else:
        z[step_at:] += 15
    return pd.DataFrame({"asset_id": "P", "window_start_utc": t, "z_pd": z}), t


class D08Tests(unittest.TestCase):
    def test_step_time_found(self):
        z, t = frame()
        c = change_time(z, "P", "discharge_pressure_kpa", t[1400])
        self.assertIsNotNone(c)
        self.assertLessEqual(abs((c - t[1000]).total_seconds()) / 60, 5)

    def test_no_change_time_without_alarm_or_for_preexisting(self):
        z, t = frame()
        self.assertIsNone(change_time(z, "P", "discharge_pressure_kpa", t[900]))         # before the step
        z2, t2 = frame(pre_existing=True)
        self.assertIsNone(change_time(z2, "P", "discharge_pressure_kpa", t2[1400]))
        self.assertIsNone(change_time(z, "P", "not_a_channel", t[1400]))

    def test_only_uses_data_up_to_alert_time(self):
        z, t = frame()
        z2 = z.copy(); z2.loc[1300:, "z_pd"] = -50.0                                        # future garbage must not matter
        self.assertEqual(change_time(z, "P", "discharge_pressure_kpa", t[1250]), change_time(z2, "P", "discharge_pressure_kpa", t[1250]))

    def test_ramp_is_described_as_building_not_changed(self):
        z, t = frame(ramp=True)
        a = AlertRecord("P", t[1450].strftime("%Y-%m-%dT%H:%M:%SZ"), "MACHINE", "IMPELLER_WEAR", 0.6, "Medium",
                        evidence=[Evidence("discharge_pressure_kpa", 15.0, "x")], z_at_alert={"discharge_pressure_kpa": 15.0})
        add_change_times([a], z)
        self.assertIn("building since", a.evidence[0].text)

    def test_evidence_text(self):
        z, t = frame()
        a = AlertRecord("P", t[1400].strftime("%Y-%m-%dT%H:%M:%SZ"), "SENSOR", "SENSOR_CALIBRATION_SHIFT", 0.6, "Medium",
                        evidence=[Evidence("discharge_pressure_kpa", 15.0, "discharge pressure is high (z = +15.0)")],
                        z_at_alert={"discharge_pressure_kpa": 15.0})
        add_change_times([a], z)
        self.assertIn("changed at 2025-01-01 16:", a.evidence[0].text)


if __name__ == "__main__":
    unittest.main()
