import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.ingest.quality import _runs, trust_frame, sensor_alerts
from assist.config import load_machine


def zf(n=60, frozen_from=None, gap_from=None):
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    d = pd.DataFrame({"asset_id": "P", "window_start_utc": t, "steady": True, "gap": False, "quality_any": 0})
    from retrofit_sim.baselines import CH
    for k in CH:
        d[f"flat_{k}"] = False
        d[f"zs_{k}"] = 0.0
    if frozen_from is not None:
        d.loc[frozen_from:, "flat_pd"] = True
    if gap_from is not None:
        d.loc[gap_from:, "gap"] = True
    return d


class Q01Tests(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(_runs(np.array([1, 1, 0, 1], bool)).tolist(), [1, 2, 0, 1])

    def test_frozen_channel_loses_trust_only_for_itself(self):
        z = zf(frozen_from=10)
        tf = trust_frame(z, flat_min=15)
        self.assertEqual(tf.trust_pd.iloc[23], 1.0)       # 14 frozen minutes: still trusted
        self.assertEqual(tf.trust_pd.iloc[-1], 0.0)
        self.assertEqual(tf.trust_i.iloc[-1], 1.0)
        a = sensor_alerts(z, tf, load_machine(), flat_min=15)
        self.assertEqual([x.fault_type for x in a], ["SENSOR_STUCK"])
        self.assertEqual(a[0].event_class, "SENSOR")

    def test_gap_gives_communication_loss(self):
        z = zf(gap_from=20)
        tf = trust_frame(z)
        self.assertTrue((tf.filter(like="trust_").iloc[-1] == 0.0).all())
        a = sensor_alerts(z, tf, load_machine(), gap_persist=10)
        self.assertEqual([x.fault_type for x in a], ["COMMUNICATION_LOSS"])


if __name__ == "__main__":
    unittest.main()
