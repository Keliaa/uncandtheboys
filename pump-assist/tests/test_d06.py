import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.detect.benign import _tail, benign_minutes
from assist.splits import DATA_FULL


class D06Tests(unittest.TestCase):
    def test_tail_extends_mask(self):
        m = np.array([0, 1, 0, 0, 0, 0], bool)
        self.assertEqual(_tail(m, 2).tolist(), [False, True, True, True, False, False])
        self.assertEqual(_tail(m, 0).tolist(), m.tolist())

    @unittest.skipUnless((DATA_FULL / "operating_context.csv").exists(), "data/full missing")
    def test_marks_manoeuvring_not_sea_passage(self):
        import pandas as pd
        b = benign_minutes(["PUMP-01"], tail_min=0)
        ctx = pd.read_csv(DATA_FULL / "operating_context.csv", usecols=["asset_id", "timestamp_utc", "schedule_state"])
        ctx = ctx[ctx.asset_id == "PUMP-01"].reset_index(drop=True)
        man = ctx.schedule_state == "arrival manoeuvring"
        self.assertTrue(man.any() and b.benign[man.to_numpy()].all())
        self.assertLess(b.benign.mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
