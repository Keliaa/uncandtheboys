import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord, Evidence
from assist.config import load_machine
from assist.evidence.derived import channel_shift, explain_by_transmitter

N = 3000
H = 30.0
SHIFT_KPA = 10.0                      # suction transmitter reads 10 kPa high late in the window -> implied head residual about -3.4 %


def zframe(current_z=0.0):
    t = pd.date_range("2025-01-01", periods=N, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    ps = 40 + rng.normal(0, 0.3, N)
    ps[2000:] += SHIFT_KPA
    z = pd.DataFrame({"asset_id": "P", "window_start_utc": t, "steady": True, "regime": 10, "suction_pressure_kpa__mean": ps,
                      "discharge_pressure_kpa__mean": 300 + rng.normal(0, 0.3, N), "fluid_temp_c__mean": 20.0})
    for k in ("ps", "pd", "i", "n", "pw", "q"):
        z[f"z_{k}"] = 0.0
    z["z_i"] = current_z
    return z


assets = pd.DataFrame({"asset_id": ["P"], "pump_rated_head_m": [H], "fluid": ["Fresh water"]})


def d04_alert(residual_pct):
    when = (pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=2500)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return AlertRecord("P", when, "MACHINE", "IMPELLER_WEAR", 0.4, "Medium", source="d04", evidence=[Evidence("discharge_pressure_kpa", residual_pct, "x")])


class DerivedTests(unittest.TestCase):
    machine = load_machine()

    def test_channel_shift_measures_the_drift_in_native_units(self):
        z = zframe()
        d = channel_shift(z, "P", "suction_pressure_kpa", pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=2500))
        self.assertAlmostEqual(d, SHIFT_KPA, delta=0.5)

    def test_head_residual_explained_by_the_transmitter_becomes_a_sensor_alert(self):
        a = d04_alert(-3.4)
        out = explain_by_transmitter([a], zframe(), assets, self.machine)
        self.assertEqual((a.event_class, a.fault_type), ("SENSOR", "SENSOR_CALIBRATION_SHIFT"))
        self.assertIn("explained by the suction pressure", out[0].evidence[0].text)

    def test_unexplained_residual_or_active_machine_channels_stay_machine(self):
        big = d04_alert(-12.0)                                   # far more than the transmitter implies
        explain_by_transmitter([big], zframe(), assets, self.machine)
        self.assertEqual(big.event_class, "MACHINE")
        loud = d04_alert(-3.4)                                   # motor current also deviates: not a lone transmitter
        explain_by_transmitter([loud], zframe(current_z=6.0), assets, self.machine)
        self.assertEqual(loud.event_class, "MACHINE")
        wrong_sign = d04_alert(+3.4)
        explain_by_transmitter([wrong_sign], zframe(), assets, self.machine)
        self.assertEqual(wrong_sign.event_class, "MACHINE")


if __name__ == "__main__":
    unittest.main()
