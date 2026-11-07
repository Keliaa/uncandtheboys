import csv, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assist.alert import AlertRecord, Evidence, write_submission_csv, SUBMISSION_COLUMNS
from assist.splits import EXPECTED, pumps_in, require_split, load_split_map


def sample(**kw):
    d = dict(asset_id="PUMP-08", alert_time_utc="2025-09-10T06:12:00Z", event_class="MACHINE",
             fault_type="SUCTION_STRAINER_CLOGGING", probability=0.64, certainty="Medium",
             evidence=[Evidence("suction_pressure_kpa", -6.1, "suction pressure low for this speed")],
             recommended_action="Inspect and clean the suction strainer.")
    d.update(kw)
    return AlertRecord(**d)


class AlertTests(unittest.TestCase):
    def test_csv_matches_template_columns(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "p.csv"
            self.assertEqual(write_submission_csv([sample()], p), 1)
            rows = list(csv.DictReader(open(p, newline="")))
        self.assertEqual(list(rows[0]), SUBMISSION_COLUMNS)
        self.assertEqual(rows[0]["confidence"], "0.64")
        template = Path(__file__).resolve().parents[2] / "original-challenge/submission_template/predictions_template.csv"
        self.assertEqual(next(csv.reader(open(template))), SUBMISSION_COLUMNS)

    def test_validation(self):
        for bad in (dict(event_class="PUMP"), dict(probability=1.2), dict(alert_time_utc="2025-09-10 06:12"),
                    dict(certainty="Huge"), dict(fault_type="")):
            with self.assertRaises(ValueError):
                sample(**bad).validate()


class SplitTests(unittest.TestCase):
    def test_splits_disjoint_and_complete(self):
        allp = [a for ids in EXPECTED.values() for a in ids]
        self.assertEqual(len(allp), len(set(allp)))
        self.assertEqual(len(allp), 12)

    def test_guard(self):
        m = load_split_map("/nonexistent/assets.csv")
        self.assertEqual(pumps_in("validation", m), ["PUMP-08", "PUMP-09"])
        require_split(["PUMP-01"], ("train",), "fit", m)
        with self.assertRaises(ValueError):
            require_split(["PUMP-01", "PUMP-10"], ("train",), "fit", m)


if __name__ == "__main__":
    unittest.main()
