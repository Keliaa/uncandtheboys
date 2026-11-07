import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.config import Asset, load_machine
from assist.detect.maintenance import lockout_restarts, maintenance_alerts, restart_steps

N = 3 * 1440


def frame(step=0.0):
    t = pd.date_range("2025-01-01", periods=N, freq="1min", tz="UTC")
    mode = np.array(["RUNNING"] * N, dtype=object)
    lock = slice(2000, 2100)                         # a 100 min lockout on day 2
    mode[lock] = "LOCKED_OUT"
    z = pd.DataFrame({"asset_id": "P", "window_start_utc": t, "operating_mode": mode, "steady": mode == "RUNNING",
                      "z_pd": 0.0, "z_a": 0.0, "z_i": 0.0})
    z.loc[2100:, "z_a"] = step                       # vibration level after the restart
    z.loc[2100:, "z_i"] = 30.0                       # current differs after restart: must be ignored (duty dependent)
    z["steady"] &= ~((z.index >= 2100) & (z.index < 2110))      # first 10 min after restart are not steady
    return z


class T02Tests(unittest.TestCase):
    asset = Asset("P", "centrifugal_pump", "train", ["discharge_pressure_kpa", "vibration_accel_rms_g", "motor_current_a"], {})

    def test_lockout_end_found(self):
        r = lockout_restarts(frame())
        self.assertEqual(len(r), 1)
        self.assertEqual(r.restart.iloc[0], pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=2100))

    def test_step_after_restart_alerts_without_future_data(self):
        z = frame(step=40.0)
        steps = restart_steps(z, lockout_restarts(z))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps.window_end.iloc[0], pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=2100 + 130))   # 2 h + 10 min
        al = maintenance_alerts(steps, {"P": self.asset}, load_machine(), 8.0)
        self.assertEqual(len(al), 1)
        self.assertEqual(al[0].alert_time_utc, steps.window_end.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.assertTrue(any("vibration accel" in e.text for e in al[0].evidence))
        self.assertNotIn("motor_current_a", steps.steps.iloc[0])        # duty-dependent channel is not compared

    def test_no_step_no_alert(self):
        z = frame(step=0.5)
        self.assertEqual(maintenance_alerts(restart_steps(z, lockout_restarts(z)), {"P": self.asset}, load_machine(), 8.0), [])

    def test_restart_in_first_day_is_skipped(self):
        z = frame(step=40.0)
        z["operating_mode"] = z["operating_mode"].where(~((z.index >= 2000) & (z.index < 2100)), "RUNNING")
        z.loc[300:400, "operating_mode"] = "LOCKED_OUT"
        r = lockout_restarts(z)
        s = restart_steps(z, r)
        self.assertTrue((s.restart >= z.window_start_utc.min() + pd.Timedelta(hours=24)).all())


if __name__ == "__main__":
    unittest.main()
