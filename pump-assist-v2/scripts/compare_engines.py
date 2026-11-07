#!/usr/bin/env python3
"""Run both engines of configs/pipeline.yaml (v2 and b4) through the same pipeline entry point and the same official scorer.

Scores train and validation, the three generated stress fleets (seeds 101-103, built for T03 in ../pump-assist/build, if present) and checks that
the b4 engine's test-pump alerts equal B4's own frozen file. Writes results/engines_compare.csv and results/test_predictions_b4_engine.csv."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev

KEYS = ["composite_score", "event_recall", "alert_precision", "false_alerts_per_asset_day", "timeliness", "fault_type_recall", "class_recall"]


def main():
    rows = []
    for engine in ("v2", "b4"):
        p = pd.concat([predictions_frame(run_pipeline(pumps_in(s), (s,), engine=engine)[1]) for s in ("train", "validation")], ignore_index=True)
        r = ev.score_splits(p)
        for sp in ("train", "validation"):
            rows.append({"engine": engine, "set": sp, "events": r[sp]["events_scored"], **{k: round(r[sp][k], 3) for k in KEYS}})
            print(rows[-1], flush=True)
        for seed in (101, 102, 103):
            sdir = ROOT.parent / "pump-assist" / "build" / f"stress_{seed}"
            if not sdir.exists():
                continue
            ids = sorted(pd.read_csv(sdir / "public" / "assets.csv").asset_id)
            ps = predictions_frame(run_pipeline(ids, ("stress",), data=sdir / "public", engine=engine)[1])
            r = ev.score_stress_dir(ps, str(sdir))
            rows.append({"engine": engine, "set": f"stress_{seed}", "events": r["events_scored"], **{k: round(r[k], 3) for k in KEYS}})
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    stress = d[d.set.str.startswith("stress")].groupby("engine")[KEYS].mean().round(3).reset_index().assign(set="stress mean")
    d = pd.concat([d, stress], ignore_index=True)
    d.to_csv(ROOT / "results" / "engines_compare.csv", index=False)
    # the b4 engine on the test pumps must equal B4's own frozen predictions (alert time, class, type, confidence)
    test = predictions_frame(run_pipeline(pumps_in("test"), ("test",), engine="b4")[1])
    test.to_csv(ROOT / "results" / "test_predictions_b4_engine.csv", index=False)
    ref = pd.read_csv(ROOT.parent / "physics-context-detector" / "results" / "b4_test_predictions.csv")
    cols = ["asset_id", "alert_time_utc", "event_class", "fault_type"]
    same = test[cols].sort_values(cols).reset_index(drop=True).equals(ref[cols].sort_values(cols).reset_index(drop=True))
    print("b4 engine test alerts equal B4's frozen file:", same, f"({len(test)} vs {len(ref)} alerts)")
    print(d[d.set.isin(["train", "validation", "stress mean"])].to_string(index=False))


if __name__ == "__main__":
    main()
