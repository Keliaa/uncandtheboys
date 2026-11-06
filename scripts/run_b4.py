#!/usr/bin/env python3
"""Run detector B4.

  python scripts/run_b4.py                 # train + validation only (development): writes results/b4_trainval_predictions.csv
                                           # and results/b4_validation_predictions.csv, prints scores vs B1
  python scripts/run_b4.py --final         # ALSO runs the frozen detector once on the test pumps and writes
                                           # results/b4_test_predictions.csv (no tuning happens in this mode)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pandas as pd  # noqa: E402

from retrofit_sim import b4  # noqa: E402

DEV_IDS = [f"PUMP-0{i}" for i in range(1, 10)]
TEST_IDS = ["PUMP-10", "PUMP-11", "PUMP-12"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "full"))
    p.add_argument("--out", default=os.path.join(ROOT, "results"))
    p.add_argument("--final", action="store_true", help="also predict the test pumps with the frozen parameters")
    p.add_argument("--params", default=None, help="JSON file with parameter overrides")
    p.add_argument("--stress", default="", help="comma-separated stress seeds in build/stress_<seed> to score as extra dev data")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    P = json.load(open(a.params)) if a.params else None
    t0 = time.time()
    inp = b4.load_inputs(a.data, DEV_IDS)
    pred = b4.run_b4(inp, P)
    pred.to_csv(os.path.join(a.out, "b4_trainval_predictions.csv"), index=False)
    val_ids = set(inp["assets"][inp["assets"].split == "validation"].asset_id)
    pred[pred.asset_id.isin(val_ids)].to_csv(os.path.join(a.out, "b4_validation_predictions.csv"), index=False)
    print(f"B4 train+validation: {len(pred)} alerts in {time.time() - t0:.0f} s")
    import eval as ev  # noqa: E402
    res = {"B4": ev.score_splits(pred, a.data)}
    print(ev.summary_table(res))
    if a.stress:
        rows = []
        for sd in a.stress.split(","):
            d = os.path.join(ROOT, "build", f"stress_{sd}")
            assets = pd.read_csv(os.path.join(d, "public", "assets.csv"))
            inp_s = b4.load_inputs(os.path.join(d, "public"), list(assets.asset_id))
            ps = b4.run_b4(inp_s, P)
            r = ev.score_stress_dir(ps, d)
            rows.append({"seed": sd, **{k: r[m] for k, m, _ in ev.KEYS}, "events": r["events_scored"]})
        st = pd.DataFrame(rows)
        print(st.round(3).to_string(index=False))
        print(f"stress mean composite {st.composite.mean():.1f}  type_rec {st.type_rec.mean():.3f}  recall {st.recall.mean():.3f}")
    if a.final:
        inp_t = b4.load_inputs(a.data, TEST_IDS)
        pt = b4.run_b4(inp_t, P)
        pt.to_csv(os.path.join(a.out, "b4_test_predictions.csv"), index=False)
        print(f"B4 test: {len(pt)} alerts written to results/b4_test_predictions.csv (not scored: labels are hidden)")
    print(f"total runtime {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
