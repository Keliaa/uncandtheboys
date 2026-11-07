#!/usr/bin/env python3
"""Merge the leave-one-pump-out results and the stress comparison into results/LOPO_REPORT.md, including warning time against the
conventional alarm deadline (lead time to `action_deadline_utc`, scored from the events)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import pandas as pd


def lead_stats(ev: pd.DataFrame) -> dict:
    d = ev[ev.lead_time_to_deadline_h.notna()]
    det = ev[ev.detected.astype(bool)]
    return {"events": len(ev), "detected": int(ev.detected.astype(bool).sum()), "type_correct": int(ev.type_correct.astype(bool).sum()),
            "events_with_deadline": len(d), "alerted_before_deadline": int((d.lead_time_to_deadline_h > 0).sum()),
            "median_lead_h": round(float(d.lead_time_to_deadline_h.median()), 1) if len(d) else None, "mean_timeliness": round(float(det.timeliness.mean()), 2) if len(det) else None}


def table(df: pd.DataFrame) -> str:
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main():
    b4 = pd.read_csv(ROOT / "results" / "lopo_b4.csv")
    v2 = pd.read_csv(ROOT / "results" / "lopo_v2.csv")
    b4e = pd.read_csv(ROOT / "results" / "lopo_b4_events.csv")
    v2e = pd.read_csv(ROOT / "results" / "lopo_v2_events.csv")
    st = pd.read_csv(ROOT / "results" / "stress_compare.csv")
    per = pd.DataFrame({"held_out": b4.held_out[:-1], "B4": b4.composite[:-1]})
    for name in v2.variant.dropna().unique():
        x = v2[(v2.variant == name) & (v2.held_out != "mean")].set_index("held_out").composite
        per[name] = per.held_out.map(x)
    means = {"held_out": "mean", "B4": round(b4.composite[:-1].astype(float).mean(), 1)}
    for name in v2.variant.dropna().unique():
        means[name] = round(v2[(v2.variant == name) & (v2.held_out != "mean")].composite.astype(float).mean(), 1)
    per = pd.concat([per, pd.DataFrame([means])], ignore_index=True)
    leads = [{"system": "B4", **lead_stats(b4e)}] + [{"system": n, **lead_stats(v2e[v2e.variant == n])} for n in v2.variant.dropna().unique()]
    stress_mean = st[st.seed.astype(str) == "mean"][["system", "composite_score", "event_recall", "alert_precision", "false_alerts_per_asset_day", "timeliness", "fault_type_recall", "class_recall"]]
    out = f"""# Leave-one-pump-out and shadow comparison

Pumps PUMP-01 to 09 (the labelled ones). Each pump is held out in turn; B4 picks its stability window N on the other 8 pumps, v2 picks its limits
(D01 k, D04 head limit) and trains its naive Bayes model on the other 8 pumps' events. The held-out pump is scored alone with the official scorer.
B4's other parameters were tuned on the train pumps during its development, so its fold results are optimistic for pumps 01 to 07.

## Held-out composite per pump

{table(per)}

## Warning against the conventional alarm deadline (all 9 held-out pumps)

{table(pd.DataFrame(leads))}

`alerted_before_deadline` counts events whose first alert came before the time a conventional limit would have been reached; most events have no
deadline. Timeliness can be high even when an alert comes after the deadline, so the lead time is the figure to watch.

## Shared stress fleets (seeds 101 to 103, frozen settings, 8 unseen pumps each)

{table(stress_mean)}

Source files: `lopo_b4.csv`, `lopo_v2.csv`, `lopo_*_events.csv`, `stress_compare.csv`.
"""
    (ROOT / "results" / "LOPO_REPORT.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
