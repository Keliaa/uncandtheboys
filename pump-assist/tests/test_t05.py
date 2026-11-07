import sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assist.splits import DATA_FULL


class T05Tests(unittest.TestCase):
    @unittest.skipUnless((DATA_FULL / "assets.csv").exists(), "data/full missing")
    def test_site_groups_match_assets_and_use_train_pumps_only(self):
        sites = {"Vessel A": ["PUMP-01", "PUMP-02", "PUMP-03"], "Plant C": ["PUMP-04", "PUMP-05"], "Plant D": ["PUMP-06"], "Vessel B": ["PUMP-07"]}
        a = pd.read_csv(DATA_FULL / "assets.csv")
        for name, ids in sites.items():
            sel = a[a.asset_id.isin(ids)]
            self.assertEqual(set(sel.split), {"train"})                       # no validation or test pump in a fold
            self.assertTrue(sel.site.str.startswith(name).all(), name)         # all pumps of the group are on that site
        # every train pump is in exactly one group
        self.assertEqual(sorted(p for v in sites.values() for p in v), sorted(a[a.split == "train"].asset_id))


if __name__ == "__main__":
    unittest.main()
