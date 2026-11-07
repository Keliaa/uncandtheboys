import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.splits import DATA_FULL

T0 = {"ambient_temp_c", "discharge_pressure_kpa", "fluid_temp_c", "motor_current_a"}


@unittest.skipUnless((DATA_FULL / "features_1min.parquet").exists(), "data/full missing")
class T04Tests(unittest.TestCase):
    def test_pump_with_only_legacy_channels_never_names_vibration_faults(self):
        from assist.pipeline import run_pipeline
        det, alerts = run_pipeline(["PUMP-09"], ("validation",), channel_filter=lambda a: T0, with_what_if=False)
        self.assertTrue(len(alerts) > 0)
        blind = {"BEARING_WEAR", "IMBALANCE", "MISALIGNMENT", "CAVITATION"}
        self.assertFalse([a for a in alerts if a.fault_type in blind])           # needs a vibration or suction sensor
        self.assertFalse([a for a in alerts if any(e.channel.startswith("vibration") for e in a.evidence)])
        self.assertFalse([a for a in alerts if a.fault_type == "SENSOR_STUCK" and any(e.channel not in T0 for e in a.evidence)])


if __name__ == "__main__":
    unittest.main()
