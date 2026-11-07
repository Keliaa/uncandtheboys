#!/usr/bin/env python3
"""C02: run the frozen version 1 system on PUMP-10..12 and write the submission CSV.

Settings are the ones chosen on train (D01 k=8, persist=10). Nothing is tuned or scored on test data.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in

TAXONOMY = {"BEARING_WEAR", "SUCTION_STRAINER_CLOGGING", "CAVITATION", "DISCHARGE_RESTRICTION", "MISALIGNMENT", "IMBALANCE",
            "IMPELLER_WEAR", "AIR_ENTRAINMENT", "SENSOR_DRIFT", "SENSOR_STUCK", "SENSOR_CALIBRATION_SHIFT",
            "SENSOR_NOISE_SPIKES", "COMMUNICATION_LOSS", "UNKNOWN"}


def main():
    _, alerts = run_pipeline(pumps_in("test"), splits=("test",))
    df = predictions_frame(alerts)
    assert set(df.event_class) <= {"MACHINE", "SENSOR"} and set(df.fault_type) <= TAXONOMY
    out = ROOT / "results" / "test_predictions.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} alerts for {sorted(df.asset_id.unique())} -> {out}")


if __name__ == "__main__":
    main()
