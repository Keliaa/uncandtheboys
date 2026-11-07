"""B4 operator demo: quiet monitoring, useful actions.

    streamlit run app.py        (or ./run_demo.sh)

Presenter-controlled replay of three prepared chapters. Reads only the prepared bundles in bundles/ and the local
interaction log in state/; it never runs the detector, never reads labels and makes no network calls.
"""
from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import channels as chn  # noqa: E402
from demo.adapter import FAULT_PLAIN, OUTCOMES, build_cards, parse_context, split_cards  # noqa: E402
from demo.interactions import InteractionStore  # noqa: E402
from demo.paths import BUNDLE_DIR  # noqa: E402
from demo.replay import MINUTE, fmt, load_scenes, ts  # noqa: E402

REPLAY_LABEL = "Recorded replay · synthetic pump data"
TICK_SECONDS = 1.0            # wall-clock seconds per replay step while playing
TREND_MINUTES = 90            # main-view trend window (ending at the replay clock)

C_MEAS = "#35577a"
C_EXP = "#9aa5b1"
C_ALERT = "#b7791f"

st.set_page_config(page_title="Pump monitoring · operator demo", page_icon="🛠️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
  .replay-label {display:inline-block; padding:2px 10px; border-radius:12px; background:#eef1f5; color:#3d4a5c;
                 font-size:0.85rem; border:1px solid #d6dce4;}
  .clock {font-size:1.15rem; white-space:nowrap; font-weight:600; color:#26323f; font-variant-numeric: tabular-nums;}
  .muted {color:#5f6b7a; font-size:0.9rem;}
  .pill {display:inline-block; padding:1px 9px; border-radius:10px; font-size:0.8rem; font-weight:600;
         margin-left:6px; vertical-align:middle;}
  .pill-new {background:#fbf0d9; color:#7a5212; border:1px solid #ecd3a0;}
  .pill-acknowledged {background:#e6eef8; color:#2c4f7c; border:1px solid #c5d5ea;}
  .pill-check_recorded {background:#eceff2; color:#3f4a56; border:1px solid #d3d9df;}
  .card-title {font-size:1.2rem; font-weight:650; color:#1f2a36;}
  .card-label {font-weight:600; color:#3d4a5c; margin-top:0.4rem;}
  .quiet-box {padding:1.1rem 1.2rem; border-radius:8px; background:#f3f6f8; border:1px solid #dde3e8;}
  .quiet-title {font-size:1.15rem; font-weight:650; color:#26323f;}
  .limited-box {padding:1.1rem 1.2rem; border-radius:8px; background:#f7f4ee; border:1px solid #e3dccd;}
  .mon-head {font-weight:650; font-size:1.05rem; color:#26323f;}
  .mon-dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;}
  .chapter {font-size:1.05rem; color:#3d4a5c;}
  .jump {padding:0.5rem 0.8rem; border-left:3px solid #8aa1b8; background:#f4f7fa; color:#3d4a5c;
         font-size:0.92rem; margin:0.3rem 0 0.6rem 0;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------------- state
@st.cache_resource(show_spinner=False)
def scenes_cached():
    return load_scenes(BUNDLE_DIR)


try:
    SCENES = scenes_cached()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

store = InteractionStore()
ss = st.session_state
ss.setdefault("idx", 0)
ss.setdefault("playing", False)
ss.setdefault("clocks", {s.id: s.start.isoformat() for s in SCENES})
ss.setdefault("flash", None)
ss.setdefault("evidence_open", {})


def scene() -> "object":
    return SCENES[ss.idx]


def clock() -> pd.Timestamp:
    s = scene()
    return s.clamp(ss.clocks[s.id])


def set_clock(t) -> None:
    ss.clocks[scene().id] = ts(t).isoformat()


# ---------------------------------------------------------------------------------------------- callbacks
def on_play():
    s = scene()
    if s.at_end(clock()):
        return
    ss.playing = not ss.playing


def go_to(i: int):
    ss.playing = False
    ss.idx = max(0, min(len(SCENES) - 1, i))
    s = scene()
    set_clock(s.start)                          # each chapter starts from its historical state
    store.restart_scene(s.id)
    ss.evidence_open = {}


def on_restart():
    go_to(ss.idx)


def on_reset():
    store.reset()
    ss.playing = False
    ss.clocks = {s.id: s.start.isoformat() for s in SCENES}
    ss.idx = 0
    ss.evidence_open = {}
    ss.flash = "Demo reset: all recorded actions cleared; replay back at chapter 1."


def on_chapter_pick():
    go_to(ss.chapter_pick)


def on_seek():
    s = scene()
    old, new = clock(), s.clamp(s.start + pd.Timedelta(minutes=int(ss.seek)))
    ss.playing = False
    if new > old and len(s.alerts):
        crossed = s.alerts[(s.alerts.time > old) & (s.alerts.time <= new)]
        if len(crossed):
            new = crossed.time.min()        # stop at the first new action, like Play does
            ss.flash = "New check recommended — replay paused."
    set_clock(new)


def on_ack(card_id: str, refs: list):
    store.acknowledge(scene().id, card_id, clock(), refs)
    ss.flash = "Acknowledged. Monitoring continues."


def on_record(card_id: str, refs: list, key: str):
    outcome = ss.get(f"out_{key}")
    if outcome not in OUTCOMES:
        ss.flash = "Choose a finding before saving."
        return
    store.record_check(scene().id, card_id, clock(), outcome, ss.get(f"notes_{key}", ""), refs)
    ss.flash = "Check recorded; monitoring continues."


def toggle_evidence(card_id: str):
    ss.evidence_open[card_id] = not ss.evidence_open.get(card_id, False)


# ------------------------------------------------------------------------------------------------- charts
def _naive(t) -> pd.Timestamp:
    return ts(t).tz_convert(None)


def trend_figure(s, view, keys: list, since, height_per: int = 120, mark_alerts: bool = True,
                 demand: bool = False):
    df = view.minutes[view.minutes.t >= ts(since)]
    keys = [k for k in keys if f"m_{k}" in df]
    titles = (["Cooling demand (%) · operating context"] if demand else []) + [s.channel_label(k) for k in keys]
    nrows = max(1, len(titles))
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=titles)
    x = df.t.dt.tz_convert(None)
    off = 0
    if demand:
        fig.add_trace(go.Scatter(x=x, y=df["demand_pct"], name="Cooling demand", mode="lines", showlegend=False,
                                 line=dict(color="#7d8a99", width=1.6, shape="hv"),
                                 hovertemplate="demand %{y:.0f} %<extra></extra>"), row=1, col=1)
        off = 1
    for r, k in enumerate(keys, start=1 + off):
        fig.add_trace(go.Scatter(x=x, y=df[f"e_{k}"], name="Expected for current conditions", mode="lines",
                                 line=dict(color=C_EXP, width=1.6, dash="dash"), showlegend=(r == 1 + off),
                                 hovertemplate="expected %{y:.2f}<extra></extra>"), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=df[f"m_{k}"], name="Measured (1-min average)", mode="lines",
                                 line=dict(color=C_MEAS, width=1.8), showlegend=(r == 1 + off),
                                 hovertemplate="measured %{y:.2f}<extra></extra>"), row=r, col=1)
    if mark_alerts and len(view.alerts):
        for t in view.alerts.time:
            if t >= ts(since):
                fig.add_vline(x=_naive(t), line=dict(color=C_ALERT, width=1, dash="dot"))
    fig.update_xaxes(range=[_naive(since), _naive(view.clock)], showgrid=True, gridcolor="#eef1f4")
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f4", zeroline=False)
    fig.update_annotations(font_size=12, x=0, xanchor="left")
    fig.update_layout(height=74 + height_per * nrows, margin=dict(l=10, r=10, t=64, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
                      legend=dict(orientation="h", yref="container", y=0.995, yanchor="top", x=1, xanchor="right",
                                  font_size=11))
    return fig


def show_fig(fig, key: str):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


# ------------------------------------------------------------------------------------------------ helpers
def esc(x) -> str:
    return html.escape(str(x))


def context_now(view) -> dict:
    if not len(view.minutes):
        return {}
    last = view.minutes.iloc[-1]
    prev = view.minutes[view.minutes.t <= last.t - pd.Timedelta(minutes=30)]
    out = {"mode": str(last.get("mode", "")).replace("_", " ").capitalize(),
           "schedule": last.get("schedule"), "demand": last.get("demand_pct"),
           "supply": last.get("supply_temp_c"), "as_of": last.available_at}
    if len(prev):
        out["demand_30"] = prev.iloc[-1].get("demand_pct")
    return out


def num(x, nd=0):
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


# ------------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"<span class='replay-label'>{REPLAY_LABEL}</span>", unsafe_allow_html=True)
    st.markdown("### Chapters")
    ss.chapter_pick = ss.idx
    st.radio("Chapter", options=list(range(len(SCENES))), key="chapter_pick", on_change=on_chapter_pick,
             format_func=lambda i: f"{SCENES[i].chapter}. {SCENES[i].meta['title']}", label_visibility="collapsed")
    s0 = scene()
    st.markdown("**Presenter note**")
    st.caption(s0.meta.get("question", ""))
    if s0.meta.get("context_note"):
        st.caption(s0.meta["context_note"])
    st.divider()
    st.caption("Actions are stored only on this laptop (operator-demo/state/). No work orders, device commands or "
               "notifications are sent.")
    st.button("Reset demo (clears all recorded actions)", on_click=on_reset, use_container_width=True)


# ------------------------------------------------------------------------------------------------ header
s = scene()
c = clock()
view = s.view(c)
mon = view.monitoring
asset = s.asset

if ss.flash:
    st.toast(ss.flash)
    ss.flash = None

h1, h2 = st.columns([3, 2])
with h1:
    st.markdown("## What needs my attention?")
with h2:
    st.markdown(f"<div style='text-align:right;padding-top:0.9rem'><span class='replay-label'>{REPLAY_LABEL}"
                f"</span></div>", unsafe_allow_html=True)

st.markdown(f"<div class='chapter'>Chapter {s.chapter} of {len(SCENES)} · <b>{esc(s.meta['title'])}</b> · "
            f"{esc(asset['asset_id'])} · {esc(asset.get('system', ''))}, {esc(asset.get('site', ''))}</div>",
            unsafe_allow_html=True)
if ss.idx > 0:
    p = SCENES[ss.idx - 1]
    same_pump = p.asset["asset_id"] == asset["asset_id"]
    st.markdown(f"<div class='jump'>Replay jumps {'forward in time on the same pump' if same_pump else 'to a different pump'}: "
                f"previous chapter showed {esc(p.asset['asset_id'])} around {fmt(p.start)}; this chapter starts at "
                f"{esc(asset['asset_id'])} · {fmt(s.start)}.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------- replay controls
b1, b2, b3, b4c, b5 = st.columns([1.05, 1.35, 1.1, 2.3, 3.2])
at_end = s.at_end(c)
b1.button("⏸ Pause" if ss.playing else "▶ Play", on_click=on_play, use_container_width=True, disabled=at_end,
          type="primary")
b2.button("⏭ Next scene", on_click=go_to, args=(ss.idx + 1,), use_container_width=True,
          disabled=ss.idx >= len(SCENES) - 1)
b3.button("↺ Restart", on_click=on_restart, use_container_width=True, help="Restart this chapter from its beginning "
          "and clear the actions recorded in it.")
b4c.markdown(f"<div class='muted'>Replay time</div><div class='clock'>{fmt(c, seconds=True)}</div>",
             unsafe_allow_html=True)
total = int((s.end - s.start) / MINUTE)
ss.seek = int((c - s.start) / MINUTE)
with b5:
    st.slider("Replay position", 0, total, key="seek", on_change=on_seek,
              format="%d min", help="Seek within this chapter. Only data available at the selected time is shown.")
    if at_end:
        st.caption("End of chapter." + (" Press **Next scene** to continue." if ss.idx < len(SCENES) - 1 else ""))
    elif ss.playing:
        st.caption(f"Playing at {s.speed:g} replay minutes per second; pauses automatically when a new check appears.")

st.divider()

# ------------------------------------------------------------------------------------------------ main
cards = build_cards(view.alerts, asset, store.visible_actions(s.id, c), s.start)
current, existing = split_cards(cards, s.start)

left, right = st.columns([3, 2], gap="large")


def render_card(card, compact: bool = False):
    key = card.card_id.replace("|", "_").replace(":", "_")
    refs = [o.source_ref for o in card.occurrences]
    with st.container(border=True):
        st.markdown(f"<span class='card-title'>{esc(card.title)}</span>"
                    f"<span class='pill pill-{card.status}'>{esc(card.status_label)}</span>",
                    unsafe_allow_html=True)
        n = len(card.occurrences)
        when = f"First raised {fmt(card.first.time)}"
        if n > 1:
            when += f" · latest {fmt(card.latest.time)} · raised {n} times"
        st.markdown(f"<div class='muted'>{esc(card.asset_id)} · {esc(card.location)}<br>{when}</div>",
                    unsafe_allow_html=True)
        if not compact:
            st.markdown("<div class='card-label'>Why this check?</div>", unsafe_allow_html=True)
            st.markdown(esc(card.why))
            st.markdown("<div class='card-label'>Next step</div>", unsafe_allow_html=True)
            st.markdown(esc(card.next_step))
        note = card.status_note()
        if note:
            st.caption(note)
        a1, a2, a3 = st.columns([1, 1.3, 1.3])
        if card.status == "new":
            a1.button("Acknowledge", key=f"ack_{key}", on_click=on_ack, args=(card.card_id, refs),
                      use_container_width=True)
        if card.status in ("acknowledged", "check_recorded") or (card.status == "new" and card.actions):
            label = "Record check" if card.status != "check_recorded" else "Add a check"
            with a2.popover(label, use_container_width=True):
                st.radio("What did you find?", OUTCOMES, key=f"out_{key}", index=None)
                st.text_area("Notes (optional)", key=f"notes_{key}", height=80)
                st.button("Save check", key=f"save_{key}", on_click=on_record, args=(card.card_id, refs, key),
                          type="primary")
                st.caption("Recording a check does not mark the pump healthy or stop monitoring.")
        open_ = ss.evidence_open.get(card.card_id, False)
        a3.button("Hide evidence" if open_ else "Show evidence", key=f"ev_{key}", on_click=toggle_evidence,
                  args=(card.card_id,), use_container_width=True)
        if open_:
            render_evidence(card, key)


def render_evidence(card, key: str):
    st.markdown("---")
    st.markdown("**Evidence** · measured readings against the values expected for the current conditions")
    ev_keys = []
    for o in reversed(card.occurrences):
        row = s.alerts[s.alerts.source_ref == o.source_ref]
        if len(row):
            for k in row.iloc[0].get("evidence_channels", []) or []:
                if k not in ev_keys:
                    ev_keys.append(k)
    since = max(view.minutes.t.min(), card.first.time - pd.Timedelta(hours=6)) if len(view.minutes) else c
    if ev_keys and len(view.minutes):
        show_fig(trend_figure(s, view, ev_keys, since, height_per=110), key=f"evfig_{key}_{c.value}")
        st.caption(f"From {fmt(since)} to the replay time. Dotted lines mark when a check was raised.")
    else:
        st.caption("No measured-versus-expected trend is available for this check.")
    ctx = parse_context(card.latest.evidence)
    if ctx:
        st.markdown(f"**Operating context when raised:** demand {ctx['demand_pct']:.0f} %, supply water "
                    f"{ctx['supply_temp_c']:.1f} °C.")
    st.markdown(f"**Suspected fault type:** {esc(', '.join(card.suspected_faults))}")
    st.markdown("**Update history**")
    for o in card.occurrences:
        st.markdown(f"- {fmt(o.time)} — {esc(FAULT_PLAIN.get(o.fault_type, o.fault_type))}")
    for a in card.actions:
        what = "Acknowledged" if a["action"] == "acknowledge" else f"Check recorded: {a.get('outcome', '')}"
        extra = f" — “{esc(a['notes'])}”" if a.get("notes") else ""
        st.markdown(f"- {fmt(a['replay_time'])} — {what}{extra}")
    with st.expander("Source detector evidence (verbatim)"):
        for o in card.occurrences:
            row = s.alerts[s.alerts.source_ref == o.source_ref]
            loc = f"{row.iloc[0]['source_file']}, line {row.iloc[0]['source_row']}" if len(row) else ""
            st.markdown(f"`{esc(o.source_ref)}` · {esc(loc)}")
            st.code(o.evidence, language=None, wrap_lines=True)


with left:
    if mon.available and not current:
        detail = ("Monitoring is active and nothing new has persisted long enough to be worth checking. Earlier "
                  "items are listed below." if existing else
                  "Monitoring is active and nothing has persisted long enough to be worth checking.")
        st.markdown("<div class='quiet-box'><div class='quiet-title'>No new checks recommended</div>"
                    f"<div class='muted' style='margin-top:0.3rem'>{detail} This is not a statement that the pump "
                    "is fault-free.</div></div>", unsafe_allow_html=True)
    elif not mon.available:
        st.markdown(f"<div class='limited-box'><div class='quiet-title'>{esc(mon.headline)}</div>"
                    f"<div class='muted' style='margin-top:0.3rem'>{esc(mon.detail)} New checks may not be raised "
                    f"while monitoring is limited.</div></div>", unsafe_allow_html=True)
    if current:
        st.markdown(f"#### Checks for this pump · {len(current)}")
        for card in current:
            render_card(card)
    if existing:
        st.markdown(f"#### Existing items · {len(existing)}")
        st.caption("Raised before this replay window and not yet closed in this demo. Shown so earlier concerns "
                   "are not hidden.")
        for card in existing:
            render_card(card, compact=False)

    keys = s.meta.get("main_channels") or s.channel_keys[:4]
    if len(view.minutes):
        since = max(c - pd.Timedelta(minutes=TREND_MINUTES), view.minutes.t.min())
        st.markdown(f"#### Recent readings · last {TREND_MINUTES} min")
        show_fig(trend_figure(s, view, keys, since, height_per=105, demand=True), key=f"main_{s.id}_{c.value}")

with right:
    dot = {"active": "#5b7f9e", "calibrating": "#b08a3e", "settling": "#b08a3e", "not_running": "#8a96a3",
           "unavailable": "#a0643c"}.get(mon.status, "#8a96a3")
    with st.container(border=True):
        st.markdown("<div class='muted'>Monitoring availability</div>"
                    f"<div class='mon-head'><span class='mon-dot' style='background:{dot}'></span>"
                    f"{esc(mon.headline)}</div>", unsafe_allow_html=True)
        st.caption(mon.detail)
        if mon.total:
            since_txt = f" · since {fmt(mon.since)}" if mon.since is not None else ""
            st.caption(f"{mon.reporting} of {mon.total} measurements reporting{since_txt}")
        st.caption("Monitoring status is independent of the checks: acknowledging or recording a check never "
                   "changes it.")

    ctx = context_now(view)
    with st.container(border=True):
        st.markdown("<div class='muted'>Operating context</div>", unsafe_allow_html=True)
        if ctx:
            d30 = ctx.get("demand_30")
            chg = f" (was {num(d30)} % 30 min earlier)" if d30 is not None and num(d30) != num(ctx['demand']) else ""
            st.markdown(f"**{esc(ctx['mode'])}**" + (f" · {esc(str(ctx['schedule']).replace('_', ' ').lower())}"
                                                      if ctx.get("schedule") else ""))
            st.markdown(f"Cooling demand **{num(ctx['demand'])} %**{chg}  \nSupply water **{num(ctx['supply'], 1)} °C**")
            st.caption(f"As of {fmt(ctx['as_of'])}")
        else:
            st.caption("No operating data received yet.")


# ------------------------------------------------------------------------------------------------ playback
if ss.playing:
    time.sleep(TICK_SECONDS)
    new, triggered = s.advance(clock(), s.speed)
    set_clock(new)
    if len(triggered):
        ss.playing = False
        ss.flash = "New check recommended — replay paused."
    elif s.at_end(new):
        ss.playing = False
    st.rerun()
