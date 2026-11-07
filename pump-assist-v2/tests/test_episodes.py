import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord
from assist.evidence.episodes import family, manage_episodes

N = 3 * 1440


def det(flag_ranges):
    t = pd.date_range("2025-01-01", periods=N, freq="1min", tz="UTC")
    flag = np.zeros(N, bool)
    for a, b in flag_ranges:
        flag[a:b] = True
    return pd.DataFrame({"asset_id": "P", "window_start_utc": t, "available_utc": t + pd.Timedelta(minutes=1), "steady": True, "flag": flag})


def alert(minute, ftype, cls="MACHINE", source="d01"):
    t = (pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return AlertRecord("P", t, cls, ftype, 0.8, "High", source=source)


class EpisodeTests(unittest.TestCase):
    def test_families(self):
        self.assertEqual(family("CAVITATION"), "hydraulic")
        self.assertEqual(family("BEARING_WEAR"), "mechanical")
        self.assertEqual(family("UNKNOWN"), "other")

    def test_same_problem_changing_type_is_one_alert_but_escalation_passes(self):
        d = det([(100, 1500)])                                   # flags on for a long time
        al = [alert(200, "SUCTION_STRAINER_CLOGGING"), alert(600, "IMPELLER_WEAR"), alert(900, "CAVITATION"), alert(1000, "CAVITATION")]
        out = manage_episodes(al, d)
        self.assertEqual([a.fault_type for a in out], ["SUCTION_STRAINER_CLOGGING", "CAVITATION"])

    def test_new_episode_after_twelve_quiet_hours_and_families_are_independent(self):
        d = det([(100, 300), (2400, 2500)])                      # quiet from 300 to 2400 (35 h)
        al = [alert(200, "DISCHARGE_RESTRICTION"), alert(250, "BEARING_WEAR"), alert(2450, "DISCHARGE_RESTRICTION")]
        out = manage_episodes(al, d)
        self.assertEqual([(a.fault_type, a.alert_time_utc[:13]) for a in out],
                         [("DISCHARGE_RESTRICTION", "2025-01-01T03"), ("BEARING_WEAR", "2025-01-01T04"), ("DISCHARGE_RESTRICTION", "2025-01-02T16")])

    def test_sensor_and_other_sources_untouched(self):
        d = det([(100, 1500)])
        al = [alert(200, "SENSOR_STUCK", "SENSOR"), alert(300, "SENSOR_STUCK", "SENSOR"), alert(400, "IMPELLER_WEAR", source="d04"),
              alert(500, "IMPELLER_WEAR", source="d04")]
        self.assertEqual(len(manage_episodes(al, d)), 4)

    def test_stop_does_not_count_as_quiet(self):
        d = det([(100, 200)])
        d.loc[300:1800, "steady"] = False                         # 25 h stopped, then running again and quiet
        al = [alert(150, "SUCTION_STRAINER_CLOGGING"), alert(1900, "IMPELLER_WEAR")]
        self.assertEqual(len(manage_episodes(al, d)), 1)          # only 100 quiet running minutes after the stop: still the same episode


if __name__ == "__main__":
    unittest.main()
