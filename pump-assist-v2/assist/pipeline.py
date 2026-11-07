"""The assistant, end to end. Every method is switched in configs/pipeline.yaml (assist/settings.py).

    shared load  ->  engine: detect + diagnose  ->  shared stages  ->  alert records
    (split guard,     v2: D01/D04/Q01/T02 + E01/E02/E03        (vibration evidence V01-V03, derived-channel check,
     features,        b4: the B4 detector, fed with the        episodes, recommend, what-if)
     replay rows,         same shared inputs
     channel filter)

Both engines read the same inputs and return the same records, so the app, the reports and the scorer do not care which one ran.
Reads telemetry only. Nothing here writes to a pump. The split guard decides which pumps may be used.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .alert import AlertRecord, SUBMISSION_COLUMNS, to_submission_rows
from .config import Asset, _asset_from_dict, assets_from_csv, load_machine
from .detect.benign import benign_minutes
from .detect.physics import detect_physics, physics_alerts
from .detect.zscore import detect, to_alerts_with_rules, z_frame
from .evidence.bayes import GaussianNB
from .ingest.features import load_assets, load_features
from .ingest.quality import sensor_alerts, trust_frame
from .recommend.actions import add_actions
from .settings import for_engine, load_settings
from .splits import DATA_FULL, EXP_ROOT

_UNSET = object()      # 'use the value from configs/pipeline.yaml'

# Kept for scripts that import them; the pipeline itself reads configs/pipeline.yaml.
PHYS_LIMIT, PHYS_PERSIST = 2.0, 30
GAP_PERSIST = FLAT_MIN = 30
BENIGN_TAIL, HS_LIMIT = 60, 2.2
E02_MODEL = EXP_ROOT / "results" / "e02_model.json"


@dataclass
class Shared:
    """Inputs both engines use: one split-guarded read of the data."""
    assets: pd.DataFrame
    machine: dict
    cfg: dict[str, Asset]
    feats: pd.DataFrame
    vib_ids: list[str]
    data: object
    splits: tuple


def load_shared(asset_ids, splits, data, features=None, channel_filter=None) -> Shared:
    assets, machine = load_assets(data), load_machine()
    cfg = {d["asset_id"]: _asset_from_dict(d) for d in assets_from_csv(f"{data}/assets.csv")}
    # features: 1-minute rows from the replay stream (I02); otherwise the rows are read from the feature file (I01)
    feats = load_features(asset_ids, data, splits) if features is None else features.sort_values(["asset_id", "window_start_utc"]).reset_index(drop=True)
    vib_ids = sorted(feats.asset_id.unique())
    if channel_filter is not None:      # T04: pretend a pump has only some of its sensors (others become missing data)
        feats = feats.copy()
        for aid in feats.asset_id.unique():
            keep = set(channel_filter(aid))
            rows = feats.asset_id == aid
            for ch in {c.split("__")[0] for c in feats.columns if "__" in c} - keep:
                feats.loc[rows, [c for c in feats.columns if c.startswith(ch + "__")]] = float("nan")
            cfg[aid].channels_available = [c for c in cfg[aid].channels_available if c in keep]
        vib_ids = [a for a in vib_ids if "vibration_rms_mm_s" in cfg[a].channels_available]
    return Shared(assets, machine, cfg, feats, vib_ids, data, tuple(splits))


def _low_trust(zf, tf) -> dict:
    from retrofit_sim.baselines import CH
    low = {}
    for aid, idx in zf.groupby("asset_id").groups.items():
        low[aid] = {CH[k]: float((tf.loc[idx, f"trust_{k}"] < 0.5).mean()) for k in CH if f"z_{k}" in zf and zf.loc[idx, f"z_{k}"].notna().any()}
    return low


# ----------------------------------------------------------------------------------------------------------- engine: v2
def detect_v2(sh: Shared, S, a: dict):
    """D01 robust z + D04 + Q01 + T02 with E01/E02/E03 diagnosis. Returns (det, alerts, zf)."""
    V = S.v2
    assets, machine, cfg, feats = sh.assets, sh.machine, sh.cfg, sh.feats
    k_alert, persist = float(V.detect.k), int(V.detect.persist)
    zf = z_frame(feats, assets, a["regime"], sh.data)
    tf = trust_frame(zf, int(V.quality.flat_min))
    if a["baseline"] != "static":      # T01: online expected values (always learning or guarded)
        from .detect.baseline import adaptive_z
        zf = adaptive_z(zf, a["baseline"], trust=tf)
    ben = (benign_minutes(sorted(feats.asset_id.unique()), sh.data, tail_min=int(V.detect.benign.tail_min), hs_limit=float(V.detect.benign.hs_limit))
           if V.detect.benign.enabled else None)
    extra = None
    if a["d02_limit"] is not None:      # D02: add EWMA-of-regression-residual flags to D01
        from .detect.regression import ewma_scores, load_context
        sc = ewma_scores(zf, load_context(sorted(feats.asset_id.unique()), sh.data))
        extra = sc.assign(flag=sc.d02_score > a["d02_limit"])[["asset_id", "window_start_utc", "flag"]]

    def add_extra(extra, frame):
        return frame if extra is None else pd.concat([extra, frame]).groupby(["asset_id", "window_start_utc"], as_index=False).flag.max()

    if a["d03_models"] is not None:     # D03: PCA/T2 flags (models fitted on healthy train minutes only)
        from .detect.pca import pca_flags
        extra = add_extra(extra, pca_flags(zf, assets, a["d03_models"])[["asset_id", "window_start_utc", "flag"]])
    if a["ewma"] is not None:           # smoothed z flags (span, limit), segments reset at stops
        from .detect.smoothing import smoothed_scores
        sc = smoothed_scores(zf, int(a["ewma"][0]))
        extra = add_extra(extra, sc.assign(flag=sc.ewma_score > float(a["ewma"][1]))[["asset_id", "window_start_utc", "flag"]])
    if a["d07_models"] is not None:     # D07: autoencoder reconstruction-error flags
        from .detect.autoencoder import autoencoder_flags
        extra = add_extra(extra, autoencoder_flags(zf, assets, a["d07_models"])[["asset_id", "window_start_utc", "flag"]])
    det = detect(feats, assets, k_alert if a["d01_flags"] else 1e9, persist, trust_gate=bool(V.detect.trust_gate), zf=zf, benign=ben, extra=extra)
    det.attrs["low_trust_fraction"] = _low_trust(zf, tf)      # share of minutes with trust < 0.5, per pump and channel
    clf = a["classifier"]               # callable(asset_id, z, time) -> {fault type: probability}; overrides the E02 model
    if clf is None and a["use_e02"] and E02_MODEL.exists():
        model = GaussianNB.load(E02_MODEL)
        clf = lambda asset, z, t: model.predict_proba(z)
    alerts = add_actions(to_alerts_with_rules(det, cfg, machine, k_alert, classifier=clf, gate=a["gate"], sensor_alone=a["sensor_alone"],
                                              stable_min=a["stable_min"]), machine)
    if a["e03_shape"]:      # E03: slow sensor movement is a drift, a jump is a calibration shift
        from .evidence.sensor import refine_sensor_types
        alerts = refine_sensor_types(alerts, zf, machine)
    from .evidence.changepoint import add_change_times      # D08: when did the deviation start?
    alerts = add_change_times(alerts, zf)
    if V.physics.enabled:
        alerts += add_actions(physics_alerts(detect_physics(feats, assets, machine, float(V.physics.limit_pct), int(V.physics.persist), benign=ben), machine), machine)
    alerts += sensor_alerts(zf, tf, machine, int(V.quality.gap_persist), int(V.quality.flat_min))
    if a["t02_alerts"]:      # T02: new baseline after maintenance lockouts
        from .detect.maintenance import lockout_restarts, maintenance_alerts, restart_steps
        steps = restart_steps(zf, lockout_restarts(zf))
        alerts += add_actions(maintenance_alerts(steps, cfg, machine, float(a["t02_step_z"])), machine)
    return det, alerts, zf


# ----------------------------------------------------------------------------------------------------------- engine: b4
def detect_b4(sh: Shared, S):
    """B4 (physics-context-detector, read only) on the shared inputs. Returns (det, alerts, zf)."""
    from .engines.b4_engine import run_b4_on_inputs
    det, alerts = run_b4_on_inputs(sh, S["b4"])
    zf = None
    if S.shared.quality_report:      # the same trust values as the v2 engine, for the report's sensor-health panel
        zf = z_frame(sh.feats, sh.assets, "speed", sh.data)
        det.attrs["low_trust_fraction"] = _low_trust(zf, trust_frame(zf, int(S.v2.quality.flat_min)))
    return det, alerts, zf


# ----------------------------------------------------------------------------------------------------------- shared stages
def shared_stages(alerts, det, sh: Shared, S, zf, a: dict):
    engine = S.engine
    machine = sh.machine
    if for_engine(S.shared.vibration_evidence, engine) or a["v03_alerts"] or a["v02_typing"]:
        from .detect.envelope import add_envelope_evidence, bearing_alerts, bearing_envelope      # V03: bearing envelope spectrum
        from .detect.orders import add_order_evidence, snapshot_orders, spectral_typing           # V02: FFT orders
        from .detect.vibration import add_snapshot_evidence, snapshot_stats                       # V01: snapshot change
        if for_engine(S.shared.vibration_evidence, engine):
            alerts = add_snapshot_evidence(alerts, snapshot_stats(sh.vib_ids, sh.data, sh.splits))
        env = bearing_envelope(sh.vib_ids, sh.data, sh.splits)
        if for_engine(S.shared.vibration_evidence, engine):
            alerts = add_envelope_evidence(alerts, env)
        if a["v03_alerts"] and not env.empty:
            alerts += bearing_alerts(env, machine)
        orders = snapshot_orders(sh.vib_ids, sh.data, sh.splits)
        if a["v02_typing"] and not orders.empty:
            alerts = spectral_typing(alerts, orders, machine)
        if for_engine(S.shared.vibration_evidence, engine):
            alerts = add_order_evidence(alerts, orders)
    if a["derived_sensor"] and zf is not None and engine == "v2":      # a head residual that one pressure transmitter explains is a sensor fault
        from .evidence.derived import explain_by_transmitter
        alerts = explain_by_transmitter(alerts, zf, sh.assets, machine)
    if a["episodes"]:      # one alert per developing problem (per fault family); value = minutes of steady quiet that close an episode
        from .evidence.episodes import manage_episodes
        alerts = manage_episodes(alerts, det, quiet_min=a["episodes"])
    for al in alerts:      # R01: an alert without an action gets the action table's text
        if not al.recommended_action:
            from .config import action_for
            al.recommended_action = action_for(machine, "UNKNOWN" if al.certainty == "Low" else al.fault_type)["text"]
    if a["with_what_if"]:
        from .recommend.whatif import add_what_if
        alerts = add_what_if(alerts, machine, sh.data)
    else:      # the sandbox uses simulator profile defaults, so it stays off unless asked; drop the stub R01 put there
        for al in alerts:
            al.what_if = None
    alerts.sort(key=lambda x: (x.asset_id, x.alert_time_utc))
    return alerts


def run_pipeline(asset_ids, splits=("train", "validation", "test"), data=DATA_FULL, use_e02=_UNSET,
                 d02_limit=_UNSET, with_what_if=_UNSET, d03_models=None, d01_flags: bool = True,
                 v03_alerts=_UNSET, v02_typing=_UNSET, baseline=_UNSET, t02_alerts=_UNSET,
                 t02_step_z=_UNSET, channel_filter=None, sensor_alone=_UNSET, e03_shape=_UNSET, classifier=None,
                 features: pd.DataFrame | None = None, gate=_UNSET, regime=_UNSET, d07_models=None, episodes=_UNSET, stable_min=_UNSET,
                 ewma=_UNSET, derived_sensor=_UNSET, engine: str | None = None, settings: dict | None = None):
    """Returns (detection frame with per-minute scores, list of AlertRecord) for the given pumps.

    Every method is set in configs/pipeline.yaml (see assist/settings.py); a keyword argument overrides the file. `engine` picks the detector:
    'v2' (this folder's methods) or 'b4' (the B4 detector). Both run on the same shared inputs and through the same shared stages.
    """
    S = load_settings(settings, engine=engine)
    V = S.v2
    pick = lambda arg, value: value if arg is _UNSET else arg
    a = {
        "use_e02": pick(use_e02, V.evidence.e02), "d02_limit": pick(d02_limit, V.options.d02_limit),
        "with_what_if": pick(with_what_if, S.shared.what_if), "v03_alerts": pick(v03_alerts, V.options.v03_alerts),
        "v02_typing": pick(v02_typing, V.options.v02_typing), "baseline": pick(baseline, V.detect.baseline),
        "t02_alerts": pick(t02_alerts, V.maintenance.enabled), "t02_step_z": pick(t02_step_z, V.maintenance.step_z),
        "sensor_alone": pick(sensor_alone, V.evidence.sensor_alone), "e03_shape": pick(e03_shape, V.evidence.shape_typing),
        "gate": pick(gate, V.evidence.gate), "regime": pick(regime, V.detect.regime),
        "episodes": pick(episodes, for_engine(S.shared.episodes_quiet_min, S.engine)) or None,
        "stable_min": pick(stable_min, V.evidence.stable_min), "derived_sensor": pick(derived_sensor, V.evidence.derived_sensor),
        "ewma": pick(ewma, V.options.ewma), "d01_flags": d01_flags, "d03_models": d03_models, "d07_models": d07_models, "classifier": classifier,
    }
    sh = load_shared(asset_ids, splits, data, features, channel_filter)
    det, alerts, zf = detect_b4(sh, S) if S.engine == "b4" else detect_v2(sh, S, a)
    alerts = shared_stages(alerts, det, sh, S, zf, a)
    return det, alerts


def predictions_frame(alerts: list[AlertRecord]) -> pd.DataFrame:
    return pd.DataFrame(to_submission_rows(alerts), columns=SUBMISSION_COLUMNS)
