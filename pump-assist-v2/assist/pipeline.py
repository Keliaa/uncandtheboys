"""The assistant, end to end, with the settings frozen from the train/validation work.

ingest (I01) -> quality (Q01) -> detect (D01 robust z, D04 physics, with D06 benign-period hold)
-> evidence (E01 rules, E02 naive Bayes) -> recommend (R01).
Reads telemetry only. Nothing here writes to a pump. The split guard decides which pumps may be used.

Frozen settings: D01 k=8, persist 10 | T02 after-maintenance step 8 spreads (2 h learning window) | D04 limit 2 %, persist 30 | Q01 gap 30, frozen 30 |
D06 tail 60 min, Hs 2.2 m | E02 model trained on train events only (results/e02_model.json).
"""
from __future__ import annotations

import pandas as pd

from .alert import AlertRecord, SUBMISSION_COLUMNS, to_submission_rows
from .config import _asset_from_dict, assets_from_csv, load_machine
from .detect.benign import benign_minutes
from .detect.physics import detect_physics, physics_alerts
from .detect.zscore import DEFAULT_K, DEFAULT_PERSIST, detect, to_alerts_with_rules, z_frame
from .evidence.bayes import GaussianNB
from .ingest.features import load_assets, load_features
from .ingest.quality import sensor_alerts, trust_frame
from .recommend.actions import add_actions
from .settings import load_settings
from .splits import DATA_FULL, EXP_ROOT

_UNSET = object()      # 'use the value from configs/pipeline.yaml'

PHYS_LIMIT, PHYS_PERSIST = 2.0, 30
GAP_PERSIST = FLAT_MIN = 30
BENIGN_TAIL, HS_LIMIT = 60, 2.2
E02_MODEL = EXP_ROOT / "results" / "e02_model.json"


def run_pipeline(asset_ids, splits=("train", "validation", "test"), data=DATA_FULL, use_e02=_UNSET,
                 d02_limit=_UNSET, with_what_if=_UNSET, d03_models=None, d01_flags: bool = True,
                 v03_alerts=_UNSET, v02_typing=_UNSET, baseline=_UNSET, t02_alerts=_UNSET,
                 t02_step_z=_UNSET, channel_filter=None, sensor_alone=_UNSET, e03_shape=_UNSET, classifier=None,
                 features: pd.DataFrame | None = None, gate=_UNSET, regime=_UNSET, d07_models=None, episodes=_UNSET, stable_min=_UNSET,
                 ewma=_UNSET, derived_sensor=_UNSET, engine: str | None = None, settings: dict | None = None):
    """Returns (detection frame with per-minute scores, list of AlertRecord) for the given pumps.

    Every method is set in configs/pipeline.yaml (see assist/settings.py); a keyword argument overrides the file. engine='b4' runs the
    B4 detector instead of the v2 methods (reads stored files only; no replayed rows, no channel filter).
    """
    S = load_settings(settings, engine=engine)
    if S.engine == "b4":
        if features is not None or channel_filter is not None:
            raise ValueError("engine b4 reads the stored files: it cannot take replayed rows or a channel filter")
        from .engines.b4_engine import run_b4_engine
        return run_b4_engine(asset_ids, splits, data, S["b4"])
    V = S.v2
    pick = lambda arg, value: value if arg is _UNSET else arg
    use_e02 = pick(use_e02, V.evidence.e02)
    d02_limit = pick(d02_limit, V.options.d02_limit)
    with_what_if = pick(with_what_if, V.recommend.what_if)
    v03_alerts = pick(v03_alerts, V.options.v03_alerts)
    v02_typing = pick(v02_typing, V.options.v02_typing)
    baseline = pick(baseline, V.detect.baseline)
    t02_alerts = pick(t02_alerts, V.maintenance.enabled)
    t02_step_z = pick(t02_step_z, V.maintenance.step_z)
    sensor_alone = pick(sensor_alone, V.evidence.sensor_alone)
    e03_shape = pick(e03_shape, V.evidence.shape_typing)
    gate = pick(gate, V.evidence.gate)
    regime = pick(regime, V.detect.regime)
    episodes = pick(episodes, V.evidence.episodes_quiet_min) or None
    stable_min = pick(stable_min, V.evidence.stable_min)
    derived_sensor = pick(derived_sensor, V.evidence.derived_sensor)
    ewma = pick(ewma, V.options.ewma)
    k_alert, persist = float(V.detect.k), int(V.detect.persist)
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
    zf = z_frame(feats, assets, regime, data)
    tf = trust_frame(zf, int(V.quality.flat_min))
    if baseline != "static":      # T01: online expected values (always learning or guarded)
        from .detect.baseline import adaptive_z
        zf = adaptive_z(zf, baseline, trust=tf)
    ben = (benign_minutes(sorted(feats.asset_id.unique()), data, tail_min=int(V.detect.benign.tail_min), hs_limit=float(V.detect.benign.hs_limit))
           if V.detect.benign.enabled else None)
    extra = None
    if d02_limit is not None:      # D02: add EWMA-of-regression-residual flags to D01
        from .detect.regression import ewma_scores, load_context
        sc = ewma_scores(zf, load_context(sorted(feats.asset_id.unique()), data))
        extra = sc.assign(flag=sc.d02_score > d02_limit)[["asset_id", "window_start_utc", "flag"]]
    if d03_models is not None:     # D03: add PCA/T2 flags (models fitted on healthy train minutes only)
        from .detect.pca import pca_flags
        pf = pca_flags(zf, assets, d03_models)[["asset_id", "window_start_utc", "flag"]]
        extra = pf if extra is None else pd.concat([extra, pf]).groupby(["asset_id", "window_start_utc"], as_index=False).flag.max()
    if ewma is not None:       # smoothed z flags (span, limit), segments reset at stops
        from .detect.smoothing import smoothed_scores
        sc = smoothed_scores(zf, int(ewma[0]))
        ef = sc.assign(flag=sc.ewma_score > float(ewma[1]))[["asset_id", "window_start_utc", "flag"]]
        extra = ef if extra is None else pd.concat([extra, ef]).groupby(["asset_id", "window_start_utc"], as_index=False).flag.max()
    if d07_models is not None:     # D07: add autoencoder reconstruction-error flags
        from .detect.autoencoder import autoencoder_flags
        af = autoencoder_flags(zf, assets, d07_models)[["asset_id", "window_start_utc", "flag"]]
        extra = af if extra is None else pd.concat([extra, af]).groupby(["asset_id", "window_start_utc"], as_index=False).flag.max()
    det = detect(feats, assets, k_alert if d01_flags else 1e9, persist, trust_gate=bool(V.detect.trust_gate), zf=zf, benign=ben, extra=extra)
    from retrofit_sim.baselines import CH
    low = {}
    for aid, idx in zf.groupby('asset_id').groups.items():
        low[aid] = {CH[k]: float((tf.loc[idx, f'trust_{k}'] < 0.5).mean()) for k in CH if f'z_{k}' in zf and zf.loc[idx, f'z_{k}'].notna().any()}
    det.attrs['low_trust_fraction'] = low      # share of minutes with trust < 0.5, per pump and channel
    clf = classifier        # callable(asset_id, z, time) -> {fault type: probability}; overrides the E02 model
    if clf is None and use_e02 and E02_MODEL.exists():
        model = GaussianNB.load(E02_MODEL)
        clf = lambda a, z, t: model.predict_proba(z)
    alerts = add_actions(to_alerts_with_rules(det, cfg, machine, k_alert, classifier=clf, gate=gate, sensor_alone=sensor_alone, stable_min=stable_min), machine)
    if e03_shape:      # E03: slow sensor movement is a drift, a jump is a calibration shift
        from .evidence.sensor import refine_sensor_types
        alerts = refine_sensor_types(alerts, zf, machine)
    from .evidence.changepoint import add_change_times      # D08: when did the deviation start?
    alerts = add_change_times(alerts, zf)
    if V.physics.enabled:
        alerts += add_actions(physics_alerts(detect_physics(feats, assets, machine, float(V.physics.limit_pct), int(V.physics.persist), benign=ben), machine), machine)   # R01 also for D04 alerts
    alerts += sensor_alerts(zf, tf, machine, int(V.quality.gap_persist), int(V.quality.flat_min))
    from .detect.vibration import add_snapshot_evidence, snapshot_stats      # V01: snapshot change as bearing-wear evidence
    alerts = add_snapshot_evidence(alerts, snapshot_stats(vib_ids, data, splits))
    if t02_alerts:      # T02: new baseline after maintenance lockouts
        from .detect.maintenance import lockout_restarts, maintenance_alerts, restart_steps
        steps = restart_steps(zf, lockout_restarts(zf))
        m_alerts = add_actions(maintenance_alerts(steps, cfg, machine, float(t02_step_z)), machine)
        alerts += m_alerts
    from .detect.envelope import add_envelope_evidence, bearing_alerts, bearing_envelope      # V03: bearing envelope spectrum
    env = bearing_envelope(vib_ids, data, splits)
    alerts = add_envelope_evidence(alerts, env)
    if v03_alerts and not env.empty:
        alerts += bearing_alerts(env, machine)
    from .detect.orders import add_order_evidence, snapshot_orders, spectral_typing      # V02: FFT orders
    orders = snapshot_orders(vib_ids, data, splits)
    if v02_typing and not orders.empty:
        alerts = spectral_typing(alerts, orders, machine)
    alerts = add_order_evidence(alerts, orders)
    if derived_sensor:      # B4 idea #6: a head residual that one pressure transmitter explains is a sensor fault, not impeller wear
        from .evidence.derived import explain_by_transmitter
        alerts = explain_by_transmitter(alerts, zf, assets, machine)
    if episodes:      # B4 idea: one alert per developing problem (per fault family); value = minutes of steady quiet that close an episode
        from .evidence.episodes import manage_episodes
        alerts = manage_episodes(alerts, det, quiet_min=episodes)
    if with_what_if:
        from .recommend.whatif import add_what_if
        alerts = add_what_if(alerts, machine, data)
    else:      # the sandbox uses simulator profile defaults, so v2 leaves it out unless asked; drop the stub R01 put there
        for a in alerts:
            a.what_if = None
    alerts.sort(key=lambda a: (a.asset_id, a.alert_time_utc))
    return det, alerts


def predictions_frame(alerts: list[AlertRecord]) -> pd.DataFrame:
    return pd.DataFrame(to_submission_rows(alerts), columns=SUBMISSION_COLUMNS)
