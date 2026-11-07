#!/usr/bin/env python3
"""E03: sensor vs machine. Choose how quiet the related channels must be (sensor_alone, as a fraction of the alert limit) on
TRAIN pumps only, then check shape typing (drift vs shift). Validation is scored once. Test pumps are not used.

Reports the class recall (MACHINE or SENSOR) and the composite for each variant."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
import eval as ev

ALONE = [None, 0.25, 0.15, 0.1]      # None = current rule (1/3 of the limit)


def run(split, alone, shape):
    return predictions_frame(run_pipeline(pumps_in(split), (split,), sensor_alone=alone, e03_shape=shape, with_what_if=False)[1])


def main():
    tr = pumps_in("train")
    rows = []
    for alone in ALONE:
        for shape in (False, True):
            r = ev.score_splits(run("train", alone, shape), assets_subset=tr)["train"]
            rows.append({"sensor_alone": alone if alone is not None else 1 / 3, "shape_typing": shape, "train_composite": round(r["composite_score"], 1),
                         "class_recall": round(r["class_recall"], 3), "type_recall": round(r["fault_type_recall"], 3), "event_recall": round(r["event_recall"], 3)})
            print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(ROOT / "results" / "e03_grid.csv", index=False)
    best = g.sort_values(["train_composite", "class_recall"], ascending=False).iloc[0]
    print("chosen on train:", dict(best))
    alone = None if abs(best.sensor_alone - 1 / 3) < 1e-9 else float(best.sensor_alone)
    out = []
    for label, a, s in (("current rule", None, False), ("E03 (chosen on train)", alone, bool(best.shape_typing))):
        p = pd.concat([run("train", a, s), run("validation", a, s)], ignore_index=True)
        r = ev.score_splits(p)
        out.append(p)
        print(f"{label}: train {r['train']['composite_score']:.1f} class recall {r['train']['class_recall']:.2f} type recall {r['train']['fault_type_recall']:.2f} | "
              f"validation {r['validation']['composite_score']:.1f} class recall {r['validation']['class_recall']:.2f} type recall {r['validation']['fault_type_recall']:.2f}", flush=True)
    out[1].to_csv(ROOT / "results" / "e03_trainval_predictions.csv", index=False)


if __name__ == "__main__":
    main()
