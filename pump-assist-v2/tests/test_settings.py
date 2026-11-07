import os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.settings import CONFIG_PATH, DEFAULTS, load_settings


class SettingsTests(unittest.TestCase):
    def test_yaml_file_matches_the_built_in_defaults(self):
        """configs/pipeline.yaml documents the frozen settings: it must not drift away from the defaults the code falls back on."""
        missing = Path(tempfile.mkdtemp()) / "none.yaml"
        self.assertEqual(dict(load_settings(path=CONFIG_PATH)), dict(load_settings(path=missing)))
        self.assertEqual(load_settings(path=missing)["v2"]["detect"]["k"], DEFAULTS["v2"]["detect"]["k"])

    def test_precedence_argument_over_environment_over_file(self):
        old = os.environ.pop("PUMP_ASSIST_ENGINE", None)
        try:
            self.assertEqual(load_settings().engine, "v2")
            os.environ["PUMP_ASSIST_ENGINE"] = "b4"
            self.assertEqual(load_settings().engine, "b4")
            self.assertEqual(load_settings(engine="v2").engine, "v2")
            self.assertEqual(load_settings(overrides={"engine": "v2"}).engine, "v2")
        finally:
            os.environ.pop("PUMP_ASSIST_ENGINE", None)
            if old is not None:
                os.environ["PUMP_ASSIST_ENGINE"] = old

    def test_nested_overrides_and_bad_values(self):
        s = load_settings(overrides={"v2": {"detect": {"k": 6.0}, "evidence": {"stable_min": 3}}, "shared": {"episodes_quiet_min": {"v2": 0}}, "b4": {"params": {"stable_min": 10}}})
        self.assertEqual(s.v2.detect.k, 6.0)
        self.assertEqual(s.v2.detect.persist, 10)             # untouched keys keep their value
        self.assertEqual((s.v2.evidence.stable_min, s.shared.episodes_quiet_min), (3, {"v2": 0, "b4": 0}))
        self.assertEqual(s["b4"]["params"], {"stable_min": 10})
        with self.assertRaises(KeyError):
            load_settings(overrides={"v2": {"detect": {"kk": 1}}})
        with self.assertRaises(ValueError):
            load_settings(engine="b5")


if __name__ == "__main__":
    unittest.main()
