#!/usr/bin/env python3
"""Development helper: score B4 parameter variants on train, validation and stress fleets (no test data).

  python scripts/sweep.py '{"k_fast": 5}' '{"k_fast": 7}'
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "original-challenge", "src"))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pandas as pd
import b4
import eval as ev

SEEDS = os.environ.get("STRESS", "101,102,103,104").split(",")


def run(over):
    P = dict(over)
    pred = b4.run_b4(b4.load_inputs(os.path.join(os.path.dirname(ROOT), "data", "full"), [f"PUMP-0{i}" for i in range(1, 10)]), P)
    r = ev.score_splits(pred)
    st = []
    for sd in SEEDS:
        d = os.path.join(ROOT, "build", f"stress_{sd}")
        assets = pd.read_csv(os.path.join(d, "public", "assets.csv"))
        ps = b4.run_b4(b4.load_inputs(os.path.join(d, "public"), list(assets.asset_id)), P)
        st.append(ev.score_stress_dir(ps, d))
    sm = sum(x["composite_score"] for x in st) / len(st)
    stt = sum(x["fault_type_recall"] for x in st) / len(st)
    return (json.dumps(over), r["train"]["composite_score"], r["validation"]["composite_score"], r["mean"], sm, stt,
            r["train"]["false_alerts_per_asset_day"], r["train"]["fault_type_recall"], r["train"]["event_recall"])


if __name__ == "__main__":
    vs = [json.loads(x) for x in sys.argv[1:]] or [{}]
    with ProcessPoolExecutor(max_workers=min(len(vs), 8)) as ex:
        res = list(ex.map(run, vs))
    print(f"{'override':60s} train   val  mean  stress s_type  FA/d  tr_type tr_rec")
    for o, tr, va, m, sm, stt, fa, tt, rc in res:
        print(f"{o[:60]:60s} {tr:5.1f} {va:5.1f} {m:5.1f} {sm:6.1f} {stt:5.2f} {fa:5.2f} {tt:5.2f} {rc:5.2f}")
