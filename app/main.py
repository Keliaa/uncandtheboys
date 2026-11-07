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


def main():
    st.set_page_config(page_title="Pump assistant", layout="wide")
    st.title("Pump assistant")
    st.caption("Read only. The assistant gives advice; a person decides.")
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
        a = alerts[idx]
        st.subheader(f"{a.fault_type}  ({a.event_class})")
        st.metric("Probability", f"{a.probability:.0%}", a.certainty)
        st.markdown("**Evidence**")
        for e in a.evidence:
            st.write(f"- {e.text}")
        if a.top_types:
            st.markdown("**Other possible causes**")
            st.write(", ".join(f"{t} {p:.0%}" for t, p in a.top_types))
        st.markdown("**Recommended action**")
        st.info(a.recommended_action)
        prev = [d for d in load_decisions() if d["asset_id"] == a.asset_id and d["alert_time_utc"] == a.alert_time_utc]
        if prev:
            st.write(f"Last decision: **{prev[-1]['decision']}**")
        cols = st.columns(len(CHOICES))
        for col, choice in zip(cols, CHOICES):
            if col.button(choice.capitalize(), key=f"{idx}-{choice}"):
                save_decision(a, choice)
                st.success(f"Saved: {choice}")
                st.rerun()


main()
