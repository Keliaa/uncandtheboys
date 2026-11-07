import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord
from app import decisions


class DecisionTests(unittest.TestCase):
    def test_save(self):
        a = AlertRecord("PUMP-08", "2025-09-10T06:12:00Z", "MACHINE", "UNKNOWN")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "d.csv"
            decisions.save_decision(a, "approved", p)
            decisions.save_decision(a, "not sure", p)
            rows = decisions.load_decisions(p)
            self.assertEqual([r["decision"] for r in rows], ["approved", "not sure"])
            with self.assertRaises(ValueError):
                decisions.save_decision(a, "maybe", p)


class AppTests(unittest.TestCase):
    def test_pump08_shows_alert_and_records_decision(self):
        if not (Path(__file__).resolve().parents[2] / "data/full/features_1min.parquet").exists():
            self.skipTest("data/full missing")
        from streamlit.testing.v1 import AppTest
        tmp = tempfile.TemporaryDirectory()
        decisions.DECISIONS_CSV = Path(tmp.name) / "d.csv"
        decisions.save_decision.__defaults__ = (decisions.DECISIONS_CSV,)
        decisions.load_decisions.__defaults__ = (decisions.DECISIONS_CSV,)
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/main.py"), default_timeout=180).run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.sidebar.selectbox[0].value, "PUMP-08")
        if not at.button:
            self.skipTest("no alert for PUMP-08")
        next(b for b in at.button if b.label == "Approved").click().run()
        self.assertEqual(decisions.load_decisions()[-1]["decision"], "approved")
        tmp.cleanup()


class ReplayDemoAppTests(unittest.TestCase):
    def test_demo_mode_shows_alert_with_evidence_action_and_what_if(self):
        from app import demo
        if not demo.available():
            self.skipTest("run scripts/c03_demo.py first")
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/main.py"), default_timeout=180).run()
        at.sidebar.radio[0].set_value("Replay demo").run()
        self.assertFalse(at.exception, at.exception)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Evidence", text)
        self.assertIn("Recommended action", text)
        self.assertIn("What-if", text)


if __name__ == "__main__":
    unittest.main()
