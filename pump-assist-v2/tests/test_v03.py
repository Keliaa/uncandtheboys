import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.alert import AlertRecord
from assist.config import load_machine
from assist.detect.envelope import add_envelope_evidence, bearing_alerts, envelope_spectrum, peak_snr

FS, RPM, BPFO = 10240.0, 1200.0, 3.057


def signal(defect: bool, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(10240) / FS
    x = 0.3 * np.sin(2 * np.pi * (RPM / 60) * t) + rng.normal(0, 0.2, len(t))     # 1x shaft tone + noise
    if defect:
        f = BPFO * RPM / 60
        for k in range(int(f)):                                                      # impulse train at BPFO, ringing at 3 kHz
            n0 = int(k / f * FS)
            seg = x[n0:n0 + 200]
            tt = np.arange(len(seg)) / FS
            x[n0:n0 + 200] += 2.0 * np.exp(-tt * 900) * np.sin(2 * np.pi * 3000 * tt)
    return x


class V03Tests(unittest.TestCase):
    def test_defect_peak_found_and_healthy_quiet(self):
        o, a = envelope_spectrum(signal(True), FS, RPM)
        bad = peak_snr(o, a, BPFO)
        o, a = envelope_spectrum(signal(False), FS, RPM)
        ok = peak_snr(o, a, BPFO)
        self.assertGreater(bad, 6.0)
        self.assertLess(ok, 4.0)

    def test_alerts_need_persistence_and_evidence_marks_low(self):
        t = pd.date_range("2025-01-01", periods=6, freq="4h", tz="UTC")
        env = pd.DataFrame({"asset_id": "P", "timestamp_utc": t, "snr_bpfo": [2, 2, 8, 20, 25, 30.0], "snr_bpfi": 2.0})
        al = bearing_alerts(env, load_machine())
        self.assertEqual(len(al), 1)                                   # two snapshots in a row, then one alert per episode
        self.assertEqual(al[0].fault_type, "BEARING_WEAR")
        self.assertEqual(al[0].alert_time_utc, "2025-01-01T12:01:00Z")
        early = AlertRecord("P", "2025-01-01T05:00:00Z", "MACHINE", "BEARING_WEAR", 0.8, "High")
        add_envelope_evidence([early], env)
        self.assertIn("argues against", early.evidence[-1].text)


if __name__ == "__main__":
    unittest.main()
