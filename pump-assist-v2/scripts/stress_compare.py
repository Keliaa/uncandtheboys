#!/usr/bin/env python3
"""Shadow comparison on the same generated stress fleets (seeds 101-103, built for T03 in ../pump-assist/build): B4 (read only), B1, v2 without
and with the derived-channel-aware sensor diagnosis. All scored with the official scorer and the full private labels. Settings are the frozen
ones of each system (nothing is tuned here)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B4 = ROOT.parent / "physics-context-detector"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "original-challenge" / "src")); sys.path.insert(0, str(B4)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
import b4
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
import eval as ev          # baseline-reproduction/scripts/eval.py (score_stress_dir, b1_predictions)

KEYS = ["composite_score", "event_recall", "alert_precision", "false_alerts_per_asset_day", "timeliness", "fault_type_recall", "class_recall"]


def main():
    rows = []
    for seed in (101, 102, 103):
        sdir = ROOT.parent / "pump-assist" / "build" / f"stress_{seed}"
        pub = str(sdir / "public")
        ids = sorted(pd.read_csv(sdir / "public" / "assets.csv").asset_id)
        systems = {"B1": ev.b1_predictions(pub),
                   "B4": b4.run_b4(b4.load_inputs(pub, ids)),
                   "v2 without derived": predictions_frame(run_pipeline(ids, ("stress",), data=sdir / "public")[1]),
                   "v2 with derived": predictions_frame(run_pipeline(ids, ("stress",), data=sdir / "public", derived_sensor=True)[1])}
        for name, pred in systems.items():
            r = ev.score_stress_dir(pred, str(sdir))
            rows.append({"seed": seed, "system": name, "events": r["events_scored"], **{k: round(r[k], 3) for k in KEYS},
                         "early_warning_rate": r["early_warning_rate"], "median_lead_h": r["median_lead_time_h"]})
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    m = d.groupby("system")[KEYS].mean().round(3).reset_index().assign(seed="mean")
    pd.concat([d, m], ignore_index=True).to_csv(ROOT / "results" / "stress_compare.csv", index=False)
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
