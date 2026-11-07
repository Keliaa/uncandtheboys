#!/usr/bin/env python3
"""D08: how close is the 'changed at' time in the evidence to the true onset of the matching labelled event?

Alerts of the frozen system on train and validation that carry a change time are matched to the labelled event of the same type
that is active at the alert time. Error = estimated change time - true onset (hours). Events that started before the data window
(pre-existing) have no change inside it and are skipped. Test pumps are not used. Labels are used only for this check."""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from assist.pipeline import run_pipeline
from assist.splits import pumps_in

PAT = re.compile(r"changed at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC")


def main():
    ev = pd.concat([pd.read_csv(ROOT.parent / "data" / "full" / "events_train.csv"), pd.read_csv(ROOT.parent / "data" / "full" / "events_validation.csv")])
    for c in ("onset_utc", "end_utc"):
        ev[c] = pd.to_datetime(ev[c], utc=True)
    rows, n_alerts, n_with = [], 0, 0
    for s in ("train", "validation"):
        _, alerts = run_pipeline(pumps_in(s), (s,), with_what_if=False)
        for a in alerts:
            n_alerts += 1
            m = [PAT.search(e.text) for e in a.evidence]
            m = [x for x in m if x]
            if not m:
                continue
            n_with += 1
            est = pd.Timestamp(m[0].group(1), tz="UTC")
            t = pd.Timestamp(a.alert_time_utc)
            w = ev[(ev.asset_id == a.asset_id) & (ev.onset_utc <= t) & (ev.end_utc.isna() | (ev.end_utc + pd.Timedelta(minutes=30) >= t)) & (ev.fault_type == a.fault_type)]
            if w.empty:
                continue
            e = w.iloc[0]
            rows.append({"asset_id": a.asset_id, "alert_time_utc": a.alert_time_utc, "fault_type": a.fault_type, "estimated_change": est.strftime("%Y-%m-%d %H:%M"),
                         "true_onset": e.onset_utc.strftime("%Y-%m-%d %H:%M"), "error_h": round((est - e.onset_utc).total_seconds() / 3600, 2)})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "d08_change_times.csv", index=False)
    print(f"{n_with} of {n_alerts} alerts carry a change time; {len(d)} matched a labelled event of the same type")
    print(d.to_string(index=False))
    if len(d):
        print("median |error| h:", round(d.error_h.abs().median(), 2), "| by type:", d.groupby("fault_type").error_h.apply(lambda x: round(x.abs().median(), 2)).to_dict())


if __name__ == "__main__":
    main()
