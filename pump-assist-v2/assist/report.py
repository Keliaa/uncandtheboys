"""U02: static HTML report, one file per pump. No server, no internet (charts are embedded PNG images).

Same panels as the app: status, data chart, alert cards (probability, evidence, action, what-if with caveats),
sensor health and blind spots. No buttons: the person decides outside the report.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .alert import AlertRecord
from .config import Asset
from .splits import EXP_ROOT

REPORT_DIR = EXP_ROOT / "results" / "reports"
CHART_CHANNELS = [("discharge_pressure_kpa", "Discharge pressure [kPa]"), ("suction_pressure_kpa", "Suction pressure [kPa]"),
                  ("motor_current_a", "Motor current [A]"), ("motor_speed_rpm", "Speed [rpm]"),
                  ("vibration_accel_rms_g", "Vibration acceleration [g]"), ("bearing_temp_c", "Bearing temperature [°C]")]
BLIND_SPOTS = {"vibration_rms_mm_s": "no vibration sensor", "suction_pressure_kpa": "no suction pressure",
               "flow_m3_h": "no flow meter", "motor_speed_rpm": "no speed signal"}

CSS = """
:root{--bg:#fff;--fg:#1c2330;--mut:#5b6678;--card:#f4f6fa;--line:#d5dbe6;--ok:#1a7f4b;--warn:#b7791f;--bad:#b42318;--acc:#2456c8}
@media (prefers-color-scheme:dark){:root{--bg:#10141c;--fg:#e6eaf2;--mut:#9aa6ba;--card:#1a2130;--line:#2c3649;--ok:#4cc38a;--warn:#e0a84a;--bad:#f0776b;--acc:#7da2ff}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin-inline:auto}
h1{font-size:1.4rem;margin:.2rem 0}h2{font-size:1.1rem;margin:1.4rem 0 .5rem}.mut{color:var(--mut)}
.pill{display:inline-block;padding:1px 10px;border-radius:99px;border:1px solid var(--line);font-size:.85rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.card h3{margin:0 0 4px;font-size:1rem}.bad{color:var(--bad)}.warn{color:var(--warn)}.ok{color:var(--ok)}
img{max-width:100%;height:auto}ul{margin:.3rem 0;padding-left:1.2rem}table{border-collapse:collapse}td,th{padding:2px 10px 2px 0;text-align:left}
details{margin-top:6px}small{color:var(--mut)}
"""


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _chart(feats: pd.DataFrame, det: pd.DataFrame, alerts: list[AlertRecord]) -> str:
    have = [(c, t) for c, t in CHART_CHANNELS if f"{c}__mean" in feats and feats[f"{c}__mean"].notna().any()][:4]
    fig, axes = plt.subplots(len(have) + 1, 1, figsize=(9, 1.9 * (len(have) + 1)), sharex=True)
    f = feats.set_index("window_start_utc")
    times = [pd.Timestamp(a.alert_time_utc) for a in alerts]
    for ax, (c, title) in zip(axes, have):
        s = f[f"{c}__mean"].resample("30min").mean()
        ax.plot(s.index, s.values, lw=0.9)
        ax.set_ylabel(title, fontsize=7); ax.tick_params(labelsize=7)
    ax = axes[-1]
    sc = det.set_index("window_start_utc")["score"].resample("30min").max()
    ax.plot(sc.index, sc.values, lw=0.9, color="tab:orange")
    ax.set_ylabel("Detector score (max |z|)", fontsize=7); ax.tick_params(labelsize=7)
    for ax in axes:
        for t in times:
            ax.axvline(t, color="tab:red", lw=0.8, alpha=0.7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return _png(fig)


def _card(a: AlertRecord) -> str:
    cls = "bad" if a.event_class == "MACHINE" else "warn"
    ev = "".join(f"<li>{html.escape(e.text)}</li>" for e in a.evidence) or "<li>No channel above the limit.</li>"
    others = ", ".join(f"{html.escape(t)} {p:.0%}" for t, p in a.top_types[1:] if p >= 0.005)
    w = a.what_if
    wi = ""
    if a.event_class == "MACHINE" and w:
        line = ""
        if "flow_change_pct" in w:
            line = (f"Flow: {w['flow_change_pct']:+.1f} % · Risk: {w['risk_before']:.2f} → {w['risk_after']:.2f} · "
                    f"Cost: approx. {w['cost_eur']:.0f} EUR")
        if not w["accepted"]:
            line += " <span class='warn'>Sandbox does not accept this action: " + html.escape("; ".join(w["reasons"])) + "</span>"
        cav = "".join(f"<li>{html.escape(c)}</li>" for c in w["caveats"])
        wi = f"<p><b>What-if (simulation)</b><br>{line}</p><details><summary>Caveats</summary><ul>{cav}</ul></details>"
    return (f"<div class='card'><h3 class='{cls}'>{html.escape(a.fault_type)} <span class='pill'>{a.event_class}</span> "
            f"<span class='pill'>certainty {a.certainty} · {a.probability:.0%}</span></h3>"
            f"<small>{html.escape(a.alert_time_utc)}</small>"
            f"{'<p><small>Other possible causes: ' + others + '</small></p>' if others else ''}"
            f"<p><b>Evidence</b></p><ul>{ev}</ul>"
            f"<p><b>Recommended action</b><br>{html.escape(a.recommended_action)}</p>{wi}</div>")


def pump_report(asset_id: str, feats: pd.DataFrame, det: pd.DataFrame, alerts: list[AlertRecord], asset: Asset,
                low_trust: dict[str, float], generated: str) -> str:
    mine = sorted((a for a in alerts if a.asset_id == asset_id), key=lambda a: a.alert_time_utc)
    n_m = sum(a.event_class == "MACHINE" for a in mine)
    n_s = len(mine) - n_m
    status = ("<span class='bad'>machine alert</span>" if n_m else "<span class='warn'>sensor alert</span>" if n_s
              else "<span class='ok'>normal</span>")
    sick = [f"{html.escape(c)}: low trust {f:.0%} of the time" for c, f in sorted(low_trust.items()) if f > 0.05]
    blind = [txt for ch, txt in BLIND_SPOTS.items() if ch not in asset.channels_available]
    npl = asset.nameplate
    cards = "".join(_card(a) for a in mine) or "<p class='ok'>No alerts for this pump.</p>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(asset_id)} report</title><style>{CSS}</style></head><body>
<p class="mut"><a href="index.html">All pumps</a> · READ ONLY: advice for a person, no path back to the pump</p>
<h1>{html.escape(asset_id)} <span class="pill">{status}</span></h1>
<p class="mut">{html.escape(str(npl.get('site', '')))} · {html.escape(str(npl.get('system', '')))} · {html.escape(str(npl.get('profile', '')))} ·
{html.escape(str(npl.get('control', '')))} · {html.escape(str(npl.get('instrumentation_tier', '')))} · split {html.escape(asset.split)}</p>
<h2>Data and detector</h2><img alt="Charts of the measured channels and the detector score; red lines mark alerts" src="{_chart(feats, det, mine)}">
<h2>Sensor health</h2><ul>{''.join(f'<li>{s}</li>' for s in sick) or '<li>All channels trusted.</li>'}</ul>
<p class="mut">Blind spots: {html.escape('; '.join(blind)) if blind else 'none'}</p>
<h2>Alerts ({len(mine)})</h2>{cards}
<p><small>Generated {generated}. Simulation results are indicative only; see the caveats on each card.</small></p></body></html>"""


def write_reports(asset_ids, splits, out_dir: Path | str = REPORT_DIR, data=None) -> list[Path]:
    from datetime import datetime, timezone
    from .config import _asset_from_dict, assets_from_csv
    from .ingest.features import load_features
    from .pipeline import run_pipeline
    from .splits import DATA_FULL
    data = data or DATA_FULL
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    det, alerts = run_pipeline(asset_ids, splits, data)
    feats = load_features(asset_ids, data, splits)
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(f"{data}/assets.csv")}
    low = det.attrs.get("low_trust_fraction", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    paths, rows = [], []
    for aid in sorted(set(feats.asset_id)):
        h = pump_report(aid, feats[feats.asset_id == aid], det[det.asset_id == aid], alerts, cfg[aid], low.get(aid, {}), now)
        p = out / f"{aid}.html"
        p.write_text(h, encoding="utf-8")
        paths.append(p)
        mine = [a for a in alerts if a.asset_id == aid]
        rows.append(f"<tr><td><a href='{aid}.html'>{aid}</a></td><td>{cfg[aid].split}</td><td>{sum(a.event_class == 'MACHINE' for a in mine)}</td>"
                    f"<td>{sum(a.event_class == 'SENSOR' for a in mine)}</td></tr>")
    (out / "index.html").write_text(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Pump reports</title><style>{CSS}</style></head><body><h1>Pump reports</h1>"
        f"<table><tr><th>Pump</th><th>Split</th><th>Machine alerts</th><th>Sensor alerts</th></tr>{''.join(rows)}</table>"
        f"<p><small>Generated {now}. Read only.</small></p></body></html>", encoding="utf-8")
    return paths
