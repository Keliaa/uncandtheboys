"""Operator app (U01). Read only: no path back to the pump.

  streamlit run app/main.py
Pumps from the train and validation splits are listed. Test pumps are excluded from the app.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import demo
from app.decisions import CHOICES, DECISIONS_CSV, load_decisions, save_decision
from assist.pipeline import run_pipeline
from assist.splits import pumps_in

CHANNELS = ["discharge_pressure_kpa", "suction_pressure_kpa", "motor_current_a", "motor_speed_rpm",
            "vibration_rms_mm_s", "vibration_accel_rms_g", "bearing_temp_c", "flow_m3_h"]


@st.cache_data(show_spinner="Reading pump data")
def load(asset_id: str):
    det, alerts = run_pipeline([asset_id], splits=("train", "validation"))
    from assist.ingest.features import load_features
    feats = load_features(asset_id, splits=("train", "validation"))
    return det, alerts, feats


def alert_panel(a, key: str, buttons: bool = True, note: str | None = None):
    """Evidence, certainty, action and what-if of one alert; decision buttons write results/decisions.csv."""
    st.subheader(f"{a.fault_type}  ({a.event_class})")
    if note:
        st.caption(note)
    st.metric("Probability", f"{a.probability:.0%}", a.certainty)
    st.markdown("**Evidence**")
    for e in a.evidence:
        st.write(f"- {e.text}")
    if a.top_types:
        st.markdown("**Other possible causes**")
        st.write(", ".join(f"{t} {p:.0%}" for t, p in a.top_types))
    st.markdown("**Recommended action**")
    st.info(a.recommended_action)
    w = a.what_if
    if a.event_class == "MACHINE" and w:
        st.markdown("**What-if (simulation)**")
        if "flow_change_pct" in w:
            st.write(f"Flow: {w['flow_change_pct']:+.1f} % · Risk: {w['risk_before']:.2f} → {w['risk_after']:.2f} · "
                     f"Cost: approx. {w['cost_eur']:.0f} EUR")
        if not w["accepted"]:
            st.warning("Sandbox does not accept this action: " + "; ".join(w["reasons"]))
        for msg in w["warnings"]:
            st.caption("⚠ " + msg)
        with st.expander("Caveats"):
            for c in w["caveats"]:
                st.write("- " + c)
    if not buttons:
        return
    prev = [d for d in load_decisions() if d["asset_id"] == a.asset_id and d["alert_time_utc"] == a.alert_time_utc]
    if prev:
        st.write(f"Last decision: **{prev[-1]['decision']}**")
    cols = st.columns(len(CHOICES))
    for col, choice in zip(cols, CHOICES):
        if col.button(choice.capitalize(), key=f"{key}-{choice}"):
            save_decision(a, choice)
            st.success(f"Saved: {choice}")
            st.rerun()


def replay_demo():
    """C03: the alerts the system showed while PUMP-08 was replayed (run scripts/c03_demo.py first)."""
    meta, items, rows = demo.load_demo()
    st.subheader(f"Replay demo: {meta['asset_id']}, system checked every {meta['check_min']} simulated minutes")
    t0, t1 = rows.window_start_utc.min().to_pydatetime(), rows.window_start_utc.max().to_pydatetime()
    now = pd.Timestamp(st.slider("Replay time (UTC)", min_value=t0, max_value=t1, value=t1, step=pd.Timedelta(hours=1).to_pytimedelta()))
    seen = rows[rows.window_start_utc <= now]
    shown = [(when, a) for when, a in items if when <= now]
    left, right = st.columns([3, 2])
    with left:
        have = [c for c in CHANNELS if f"{c}__mean" in seen.columns and seen[f"{c}__mean"].notna().any()]
        pick = st.multiselect("Channels", have, default=have[:2], key="demo_ch")
        if pick:
            st.line_chart(seen.set_index("window_start_utc")[[f"{c}__mean" for c in pick]].resample("10min").mean())
    with right:
        st.write(f"{len(shown)} alert(s) shown at this replay time")
        if not shown:
            st.success("No alerts yet.")
            return
        idx = st.selectbox("Alert", range(len(shown)), index=len(shown) - 1,
                           format_func=lambda i: f"{shown[i][1].alert_time_utc}  {shown[i][1].fault_type}", key="demo_alert")
        when, a = shown[idx]
        alert_panel(a, f"demo{idx}", buttons=False,
                    note=f"Shown at replay time {when:%Y-%m-%d %H:%M}. Scripted operator decision: {a.operator_decision}")


def main():
    st.set_page_config(page_title="Pump assistant", layout="wide")
    st.title("Pump assistant")
    st.caption("Read only. The assistant gives advice; a person decides.")
    modes = ["Batch (stored data)"] + (["Replay demo"] if demo.available() else [])
    if st.sidebar.radio("Mode", modes) == "Replay demo":
        replay_demo()
        return
    pumps = pumps_in("train") + pumps_in("validation")
    asset = st.sidebar.selectbox("Pump", pumps, index=pumps.index("PUMP-08"))
    det, alerts, feats = load(asset)
    st.sidebar.write(f"{len(alerts)} alert(s) for {asset}")

    left, right = st.columns([3, 2])
    with left:
        have = [c for c in CHANNELS if f"{c}__mean" in feats.columns and feats[f"{c}__mean"].notna().any()]
        shown = st.multiselect("Channels", have, default=have[:2])
        if shown:
            st.line_chart(feats.set_index("window_start_utc")[[f"{c}__mean" for c in shown]].resample("10min").mean())
        st.subheader("Detector score")
        st.line_chart(det.set_index("window_start_utc")["score"].resample("10min").max())
    with right:
        if not alerts:
            st.success("No alerts for this pump.")
            return
        idx = st.selectbox("Alert", range(len(alerts)),
                           format_func=lambda i: f"{alerts[i].alert_time_utc}  {alerts[i].fault_type}")
        alert_panel(alerts[idx], str(idx))


main()
