#!/usr/bin/env python3
"""T03: the frozen system (no change of any setting or model) on generated stress fleets, scored with the private labels.

Fleets: build/stress_<seed>/ made with
  python original-challenge/make_dataset.py --tier stress --seed <seed> --stress-assets 8 --days 7 --output pump-assist/build/stress_<seed>
(run from pump-assist/; seeds 101-103). Stress pumps have unseen sizes, seasons and missing sensors; only about 30 % of events are
published as labels, the scorer here uses all of them. B1 (k=6, persist 15) is scored on the same fleets for comparison.
Writes results/t03_stress.csv and results/t03_stress.md."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
import eval as ev

SEEDS = [101, 102, 103]
KEYS = ["composite_score", "event_f1", "event_recall", "alert_precision", "false_alerts_per_asset_day", "timeliness", "fault_type_recall", "class_recall"]


def main():
    rows = []
    for seed in SEEDS:
        sdir = ROOT / "build" / f"stress_{seed}"
        pub = sdir / "public"
        ids = sorted(pd.read_csv(pub / "assets.csv").asset_id)
        _, alerts = run_pipeline(ids, ("stress",), data=pub)
        pred = predictions_frame(alerts)
        pred.to_csv(ROOT / "results" / f"t03_stress_{seed}_predictions.csv", index=False)
        r = ev.score_stress_dir(pred, str(sdir))
        b = ev.score_stress_dir(ev.b1_predictions(str(pub)), str(sdir))
        for name, x in (("frozen system", r), ("B1 baseline", b)):
            rows.append({"seed": seed, "system": name, "events": x["events_scored"], "detected": x["true_positive_events"],
                         **{k: round(x[k], 3) for k in KEYS}})
        print(rows[-2], "\n", rows[-1], flush=True)
    t = pd.DataFrame(rows)
    mean = t.groupby("system")[KEYS].mean().round(3).reset_index().assign(seed="mean")
    t = pd.concat([t, mean], ignore_index=True)
    t.to_csv(ROOT / "results" / "t03_stress.csv", index=False)
    md = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    (ROOT / "results" / "t03_stress.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
