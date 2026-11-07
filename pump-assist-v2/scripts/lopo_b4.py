#!/usr/bin/env python3
"""Leave-one-pump-out on B4 (physics-context-detector, read only, imported unchanged).

For each of the 9 labelled pumps (PUMP-01..09), B4's stability window N (stable_min in 5, 10, 20, 30) is chosen on the other 8 pumps (mean of the
train and validation composites there) and B4 is scored on the held-out pump alone. B4 runs once per N on all 9 pumps (every pump is detected
independently, so that is equivalent). Writes results/lopo_b4.csv and per-event lead times. Test pumps are never run."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B4 = ROOT.parent / "physics-context-detector"
sys.path.insert(0, str(ROOT.parent / "original-challenge" / "src")); sys.path.insert(0, str(B4)); sys.path.insert(0, str(B4 / "scripts"))

import pandas as pd
import b4
import eval as ev

DEV = [f"PUMP-0{i}" for i in range(1, 10)]
GRID = [5, 10, 20, 30]
DATA = str(ROOT.parent / "data" / "full")


def pump_report(pred, pump):
    r = ev.score_splits(pred, DATA, assets_subset=[pump])
    sp = "train" if "train" in r else "validation"
    return r[sp]


def main():
    inp = b4.load_inputs(DATA, DEV)
    preds = {}
    for n in GRID:
        preds[n] = b4.run_b4(inp, {"stable_min": n})
        print("ran B4 with stable_min", n, len(preds[n]), "alerts", flush=True)
    per = {n: {p: pump_report(preds[n], p) for p in DEV} for n in GRID}
    rows, events = [], []
    for held in DEV:
        rest = [p for p in DEV if p != held]
        score_rest = {}
        for n in GRID:
            r = ev.score_splits(preds[n], DATA, assets_subset=rest)
            score_rest[n] = r["mean"]
        n_best = max(GRID, key=lambda n: (score_rest[n], n))
        r = per[n_best][held]
        rows.append({"held_out": held, "chosen_N": n_best, "composite": round(r["composite_score"], 1), "recall": round(r["event_recall"], 2),
                     "type_recall": round(r["fault_type_recall"], 2), "class_recall": round(r["class_recall"], 2),
                     "FA_per_day": round(r["false_alerts_per_asset_day"], 2), "events": r["events_scored"], "early_warning_rate": r["early_warning_rate"],
                     "median_lead_h": r["median_lead_time_h"]})
        for e in r["per_event"]:
            events.append({"held_out": held, **{k: e.get(k) for k in ("event_id", "fault_type", "detected", "type_correct", "class_correct", "lead_time_to_deadline_h", "timeliness")}})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    mean = {"held_out": "mean", "chosen_N": "", "composite": round(d.composite.mean(), 1), "recall": round(d.recall.mean(), 2), "type_recall": round(d.type_recall.mean(), 2),
            "class_recall": round(d.class_recall.mean(), 2), "FA_per_day": round(d.FA_per_day.mean(), 2), "events": int(d.events.sum()), "early_warning_rate": "", "median_lead_h": ""}
    pd.concat([d, pd.DataFrame([mean])], ignore_index=True).to_csv(ROOT / "results" / "lopo_b4.csv", index=False)
    pd.DataFrame(events).to_csv(ROOT / "results" / "lopo_b4_events.csv", index=False)
    print("mean held-out composite", mean["composite"])


if __name__ == "__main__":
    main()
