"""Opt-in B4 relabeling. Publication-time features; no detector or episode changes.

Artifacts are trusted local pickle files, not an interchange format for untrusted models.
Override confidence 0.6 is deliberately fixed and UNCALIBRATED.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import b4

VERSION = 1
CATEGORIES = ['control', 'tier', 'sensor_channel']
SENSOR = {'SENSOR_DRIFT', 'SENSOR_CALIBRATION_SHIFT'}
IDENTITY = ['asset_id', 'alert_time_utc', 'event_class']

def versions():
    return {k: importlib.metadata.version(k) for k in ('numpy', 'pandas', 'scipy', 'scikit-learn')}

def b4_hash():
    return hashlib.sha256(Path(b4.__file__).read_bytes()).hexdigest()

def feature_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def sensor_channel(row):
    detector = str(row.get('detector', ''))
    if detector == 'single':
        for k, name in b4.NAME.items():
            if str(row.get('evidence', '')).startswith(name + ' deviates alone:'):
                return 'n' if k == 'rx' else k
    return detector.removeprefix('single_') if detector.startswith('single_') else ''

def features(inputs, pred, frames=None):
    """Explicit feature allowlist. Frame rows represent COMPLETED minute windows.

    No IDs, timestamps, original diagnosis, event labels or simulator truth enter X.
    B4 preparation is causal; all reductions below are prefix-only.
    """
    assets = inputs['assets'].set_index('asset_id', drop=False)
    frames = frames or {a.asset_id: b4.prepare_asset(inputs, a, b4.PARAMS)
                        for _, a in inputs['assets'].iterrows()}
    snapshots = {aid: b4.snapshot_features(inputs.get('snaps'), a)
                 for aid, a in assets.iterrows()}
    rows = []
    for _, alert in pred.iterrows():
        a = assets.loc[alert.asset_id]
        f, channels = frames[alert.asset_id]
        cut = pd.Timestamp(alert.alert_time_utc)
        h = f[pd.to_datetime(f.t, utc=True) + pd.Timedelta(minutes=1) <= cut]
        r = dict(control=a.control, tier=a.instrumentation_tier, sensor_channel=sensor_channel(alert))
        available, _ = b4.static_channels(a)
        for k in list(b4.CH) + ['rx', 'hr']:
            r[f'available_{k}'] = float(k in channels)
            for prefix in ['e_', 'z_']:
                col = prefix + k
                v = h[col] if col in h else pd.Series(dtype=float)
                r[col + '_now'] = v.iloc[-1] if len(v) else np.nan
                for n in (20, 60):
                    w = v.tail(n)
                    r[f'{col}_mean{n}'] = w.mean()
                    r[f'{col}_std{n}'] = w.std()
                    r[f'{col}_maxabs{n}'] = w.abs().max()
            z = h['z_' + k] if 'z_' + k in h else pd.Series(dtype=float)
            for span in (5, 60):
                sm = b4.ewm_seg(np.where(h.steady, z, np.nan), span, ~h.run.to_numpy()) if len(z) else []
                value = sm[-1] if len(sm) else np.nan
                if span == 5 and len(h):
                    value = float(np.nan_to_num(value)) if h.steady.iloc[-1] else 0.
                r[f'z_{k}_ewm{span}'] = value
            tail = z.tail(360).to_numpy()
            above = np.flatnonzero(np.isfinite(tail) & (np.abs(tail) <= 1.5))
            r[f'rise_{k}'] = float(len(tail) - 1 - above[-1]) if len(above) else np.nan
            r[f'physics_{k}'] = np.nan
            if len(h) and 'ph_' + k in h and 'x_' + k in h:
                ph = h['ph_' + k].iloc[-1]
                r[f'physics_{k}'] = (h['x_' + k].iloc[-1] - ph) / max(abs(ph), 1e-6)
        # Positive-amplitude least-squares signature fits in standardized residual space.
        keys = [k for k in b4.HYD if k in channels and 'sig_SUCTION_STRAINER_CLOGGING_' + k in h]
        y = np.array([r[f'z_{k}_mean20'] for k in keys])
        fits = []
        for fault in b4.SIG_FAULTS:
            s = np.array([h[f'sig_{fault}_{k}'].iloc[-1] / h['s_' + k].iloc[-1] for k in keys]) if len(h) else np.array([])
            ok = np.isfinite(y) & np.isfinite(s)
            yy, ss = y[ok], s[ok]
            amp = max(0., float(yy @ ss) / max(float(ss @ ss), 1e-12))
            err = float(np.mean((yy - amp * ss) ** 2)) if len(yy) else np.nan
            r['fit_' + fault] = err
            r['amplitude_' + fault] = amp
            fits.append(err)
        fs = sorted(x for x in fits if np.isfinite(x))
        r['signature_margin'] = fs[1] - fs[0] if len(fs) > 1 else np.nan
        # Qualitative B4 vibration / transient hydraulic signatures complement its
        # quantitative pump-curve signatures. These are residual-space fit errors.
        patterns = {'BEARING_WEAR': {'a':1,'c':1,'bt':1,'v':.5},
                    'IMBALANCE': {'v':1}, 'MISALIGNMENT': {'v':1,'bt':.5,'wt':.5},
                    'CAVITATION': {'a':1,'c':1,'v':.5,'ps':-1},
                    'AIR_ENTRAINMENT': {'ps':-1,'pd':-1,'q':-1}}
        for fault, pattern in patterns.items():
            kk = [k for k in list(b4.VIB) + b4.TEMP + ['ps','pd','q'] if np.isfinite(r[f'z_{k}_mean20'])]
            yy = np.array([r[f'z_{k}_mean20'] for k in kk])
            ss = np.array([pattern.get(k,0.) for k in kk])
            amp = max(0., float(yy @ ss) / max(float(ss @ ss),1e-12))
            r['fit_' + fault] = float(np.mean((yy-amp*ss)**2)) if len(yy) else np.nan
            r['amplitude_' + fault] = amp
        for k in b4.HYD:
            r['sensor_fit_' + k] = float(np.nansum([r[f'z_{q}_mean20'] ** 2 for q in keys if q != k])) if keys else np.nan
        errors = {k[4:]:v for k,v in r.items() if k.startswith('fit_')}
        for fault, error in errors.items():
            others = [v for k,v in errors.items() if k != fault and np.isfinite(v)]
            r['fit_margin_' + fault] = min(others) - error if others else np.nan
        for col in ['u', 'ctx_dem', 'ctx_tsup', 'ctx_amb', 'ctx_hs', 'ctx_lvl', 'since_start']:
            r[col] = h[col].iloc[-1] if len(h) and col in h else np.nan
        sf = snapshots[alert.asset_id]
        sf = sf[pd.to_datetime(sf.t, utc=True) <= cut]
        age = (cut - pd.Timestamp(sf.t.iloc[-1])).total_seconds() / 60 if len(sf) else np.nan
        fresh = len(sf) and age <= b4.PARAMS['snap_valid_min']
        r['snapshot_available'] = float(bool(fresh))
        r['snapshot_age_min'] = age if fresh else np.nan
        for col in ['rpm', 'v1', 'v2', 'v3', 'env_bpfo', 'env_bpfi', 'hf']:
            r['snapshot_' + col] = sf[col].iloc[-1] if fresh else np.nan
        rows.append(r)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

def targets(pred, events, assets, benign, seed):
    """Use exactly the scorer windows, but reject ALL overlaps before looking at type."""
    from retrofit_sim.scoring import PRE_TOL, POST_TOL, SOFT_END
    assets = assets.set_index('asset_id')
    windows = []
    for _, e in events.iterrows():
        a = assets.loc[e.asset_id]
        onset, detectable, end = (pd.to_datetime(e[c], utc=True) for c in ('onset_utc', 'detectable_utc', 'end_utc'))
        start = max(onset, pd.Timestamp(a.window_start_utc))
        if pd.notna(detectable):
            start = max(start, detectable - PRE_TOL)
        stop = pd.Timestamp(a.window_end_utc) if pd.isna(end) else min(pd.Timestamp(a.window_end_utc), end + (POST_TOL if e.fault_type in SOFT_END or e.event_class == 'SENSOR' else pd.Timedelta(0)))
        windows.append((e, start, stop))
    rows = []
    for _, r in pred.iterrows():
        t = pd.Timestamp(r.alert_time_utc)
        candidates = [e for e, s, z in windows if e.asset_id == r.asset_id and s <= t <= z]
        bn = any(b.asset_id == r.asset_id and pd.Timestamp(b.start_utc) <= t <= pd.Timestamp(b.end_utc) for _, b in benign.iterrows())
        reason = 'benign' if bn else 'unmatched' if not candidates else 'overlap' if len(candidates) > 1 else 'matched'
        e = candidates[0] if len(candidates) == 1 else None
        if reason == 'matched' and str(e.observable).lower() not in ('true', '1'):
            reason = 'unobservable'
        eligible = r.event_class == (e.event_class if e is not None else '')
        if r.event_class == 'SENSOR':
            eligible = eligible and r.fault_type in SENSOR
        rows.append(dict(seed=seed, asset_id=r.asset_id, event_id=e.event_id if e is not None else '',
                         target=e.fault_type if reason == 'matched' else '',
                         target_class=e.event_class if reason == 'matched' else '', reason=reason, eligible=eligible))
    out = pd.DataFrame(rows, columns=['seed', 'asset_id', 'event_id', 'target', 'target_class', 'reason', 'eligible'])
    out['weight'] = 0.
    ok = out.reason == 'matched'
    out.loc[ok, 'weight'] = 1 / out[ok].groupby(['seed', 'asset_id', 'event_id']).event_id.transform('size')
    return out

def fit(X, y, training_manifest):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer, MissingIndicator
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    if 'training_seeds' in training_manifest:
        allowed = set(training_manifest['training_seeds'])
        if not allowed <= set(range(201, 231)) or not set(y.seed.astype(int)) <= allowed:
            raise ValueError('split isolation: only seeds 201..230 may fit preprocessing or forests')
    models = {}
    for cls in ['MACHINE', 'SENSOR']:
        mask = (y.reason == 'matched') & (y.target_class == cls)
        if 'eligible' in y:
            mask &= y.eligible.astype(str).str.lower().isin(['true', '1'])
        if cls == 'SENSOR':
            mask &= y.target.isin(SENSOR)
        counts = y[mask].drop_duplicates(['seed', 'asset_id', 'event_id']).groupby('target').size()
        supported = sorted(counts[counts >= 10].index)
        # Fit all represented classes; acceptance is separately restricted to >=10 events.
        if not mask.any():
            continue
        nums = [c for c in X if c not in CATEGORIES]
        prep = ColumnTransformer([
            ('numeric', FeatureUnion([('median', SimpleImputer(strategy='median', keep_empty_features=True)),
                                      ('missing', MissingIndicator(features='all'))]), nums),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORIES)])
        model = Pipeline([('preprocess', prep), ('forest', RandomForestClassifier(n_estimators=200, max_depth=8,
                          min_samples_leaf=5, class_weight='balanced', random_state=2026, n_jobs=1))])
        # Normalize within the actual eligible fitting rows, including events with
        # other emitted classes that are intentionally ineligible for this model.
        yy = y.loc[mask]
        weights = 1 / yy.groupby(['seed', 'asset_id', 'event_id']).event_id.transform('size')
        model.fit(X.loc[mask], yy.target, forest__sample_weight=weights)
        models[cls] = dict(model=model, supported=supported, counts=counts.to_dict(),
                           combinations=sorted(set(zip(X.loc[mask, 'control'], X.loc[mask, 'tier']))))
    return dict(version=VERSION, versions=versions(), b4_hash=b4_hash(), feature_hash=feature_hash(), schema=list(X), models=models,
                training_manifest=training_manifest, gate=None, confidence='0.6 fixed, uncalibrated')

def compatible(artifact, X):
    validate_structure(artifact)
    if artifact['version'] != VERSION or artifact['versions'] != versions() or artifact['b4_hash'] != b4_hash() or artifact['feature_hash'] != feature_hash():
        raise ValueError('model version, dependency versions, or frozen B4 hash mismatch')
    if artifact['schema'] != list(X):
        raise ValueError('ordered feature schema mismatch')

def validate_structure(artifact):
    if not isinstance(artifact, dict) or not isinstance(artifact['models'], dict):
        raise ValueError('invalid model structure')
    gate = artifact['gate']
    if gate is not None and (len(gate) != 2 or gate[0] not in (.6,.7,.8,.9) or gate[1] not in (.1,.2,.3)):
        raise ValueError('invalid acceptance gate')
    for cls,family in artifact['models'].items():
        if cls not in ('MACHINE','SENSOR') or not hasattr(family['model'], 'predict_proba'):
            raise ValueError('invalid classifier family')
        classes = set(family['model'].classes_)
        allowed = SENSOR if cls == 'SENSOR' else {k for k,v in b4.CLASS.items() if v == 'MACHINE'}
        if not classes <= allowed or not set(family['supported']) <= classes:
            raise ValueError('invalid supported classes')
        if not isinstance(family['combinations'], (list,tuple)):
            raise ValueError('invalid control/tier support')

def load_model(path):
    try:
        with open(path, 'rb') as f:
            artifact = pickle.load(f)
        validate_structure(artifact)
        if artifact['version'] != VERSION or artifact['versions'] != versions() or artifact['b4_hash'] != b4_hash() or artifact['feature_hash'] != feature_hash():
            raise ValueError('incompatible artifact')
        return artifact, ''
    except Exception as exc:
        reason = f'model unavailable/incompatible: {exc}'
        warnings.warn(reason + '; retaining unchanged B4 output', RuntimeWarning)
        return None, reason

def relabel(pred, X, artifact, failure='', gate=None):
    out = pred.copy(deep=True)
    diag = []
    if artifact is not None:
        try:
            compatible(artifact, X)
        except Exception as exc:
            failure, artifact = str(exc), None
            warnings.warn(f'{failure}; retaining unchanged B4 output', RuntimeWarning)
    for pos, (idx, r) in enumerate(pred.iterrows()):
        reason, scores, candidate, accepted = failure or 'fallback_only', {}, '', False
        selected_gate = gate if gate is not None else artifact.get('gate') if artifact else None
        if artifact is not None and selected_gate:
            family = artifact['models'].get(r.event_class)
            reason = 'rule_based' if r.event_class == 'SENSOR' and r.fault_type not in SENSOR else 'no_model'
            if family and not (r.event_class == 'SENSOR' and r.fault_type not in SENSOR):
                xx = X.iloc[[pos]]
                pp = family['model'].predict_proba(xx)[0]
                scores = dict(zip(family['model'].classes_, map(float, pp)))
                order = np.argsort(pp)[::-1]
                candidate = str(family['model'].classes_[order[0]])
                margin = pp[order[0]] - (pp[order[1]] if len(pp) > 1 else 0.)
                reason = 'low_score'
                if (xx.control.iloc[0], xx.tier.iloc[0]) not in family['combinations']:
                    reason = 'unsupported_control_tier'
                elif candidate not in family['supported']:
                    reason = 'unsupported_class'
                elif pp[order[0]] >= selected_gate[0] and margin >= selected_gate[1]:
                    reason, accepted = 'accepted', True
        channel = sensor_channel(r)
        parts = []
        if accepted and candidate != r.fault_type:
            measured = []
            for k in b4.CH:
                for suffix in ('now', 'mean20', 'mean60'):
                    e = X.iloc[pos].get(f'e_{k}_{suffix}')
                    z = X.iloc[pos].get(f'z_{k}_{suffix}')
                    if pd.notna(e) and pd.notna(z):
                        measured.append((abs(z), f'{b4.NAME[k]} deviation {e:+.3g} {b4.UNIT[k]} (z={z:+.2f}, {suffix})'))
                        break
            parts = [s for _,s in sorted(measured,reverse=True)[:5]]
            if not parts:
                accepted, reason = False, 'no_measured_evidence'
            elif r.event_class == 'SENSOR' and not channel:
                accepted, reason = False, 'missing_sensor_attribution'
        changed = accepted and candidate != r.fault_type
        if changed:
            out.at[idx, 'fault_type'] = candidate
            out.at[idx, 'confidence'] = 0.6
            out.at[idx, 'evidence'] = f'{candidate}: ' + '; '.join(parts) + (f'. Attributed channel: {channel}.' if channel else '.')
            out.at[idx, 'recommended_action'] = b4.ACTIONS[candidate].replace('the transmitter', f'the {b4.NAME.get(channel, channel)} transmitter') if channel else b4.ACTIONS[candidate]
        diag.append(dict(original_diagnosis=r.fault_type, candidate=candidate, scores=__import__('json').dumps(scores, sort_keys=True),
                         accepted=accepted, changed=changed, fallback_reason='' if accepted else reason,
                         sensor_channel=sensor_channel(r)))
    pd.testing.assert_frame_equal(pred[IDENTITY], out[IDENTITY])
    return out, pd.concat([pred[IDENTITY].reset_index(drop=True), pd.DataFrame(diag)], axis=1)
