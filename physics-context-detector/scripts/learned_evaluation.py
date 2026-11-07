"""Frozen split protocol and unchanged official scorer; generated artifacts only in build/."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'original-challenge' / 'src'), str(ROOT / 'scripts')]
import b4
import learned_diagnosis as ld
from learned_experiment import fleet, generate, extract, digest, write_json
from retrofit_sim.scoring import score
OUT = ROOT / 'build' / 'learned'
MODEL = OUT / 'model.pkl'
METRICS = ['composite_score', 'fault_type_recall', 'event_recall', 'false_alerts_per_asset_day']
SUBMIT = ['asset_id', 'alert_time_utc', 'event_class', 'fault_type', 'confidence', 'evidence', 'recommended_action']

def cached(seed):
    d = fleet(seed)
    expected = dict(generation=digest(d / 'generation.json'), b4=digest(ROOT / 'b4.py'),
                    features=digest(ROOT / 'learned_diagnosis.py'))
    if json.loads((d / 'features.json').read_text()) != expected:
        raise ValueError(f'Stale feature cache for {seed}; rerun prepare')
    return (pd.read_csv(d / 'baseline.csv'), pd.read_parquet(d / 'features.parquet'),
            pd.read_csv(d / 'targets.csv', keep_default_na=False))

def data(seed):
    d = fleet(seed)
    return (pd.read_csv(d / 'private' / 'events_stress_all.csv'), pd.read_csv(d / 'public' / 'assets.csv'),
            pd.read_csv(d / 'public' / 'benign_windows_stress.csv'))

def pooled(predictions, seeds, control_tier=None):
    """Prefix provenance IDs to prevent collisions between independently sampled fleets."""
    pp, ee, aa, bb = [], [], [], []
    for p, seed in zip(predictions, seeds):
        e, a, b = data(seed)
        p = p.copy()
        for frame in (p, e, a, b):
            frame['asset_id'] = str(seed) + ':' + frame.asset_id.astype(str)
        if control_tier:
            a = a[(a.control == control_tier[0]) & (a.instrumentation_tier == control_tier[1])]
        e['event_id'] = str(seed) + ':' + e.event_id.astype(str)
        pp.append(p); ee.append(e); aa.append(a); bb.append(b)
    return score(pd.concat(pp), pd.concat(ee), pd.concat(aa), pd.concat(bb), 'stress')

def improves(base, new):
    return (new['fault_type_recall'] > base['fault_type_recall'] and
            new['composite_score'] >= base['composite_score'] and
            new['event_recall'] >= base['event_recall'] and
            new['false_alerts_per_asset_day'] <= base['false_alerts_per_asset_day'])

def train(train_seeds=None, selection=None, holdout=None, protocol='full 50-fleet experiment'):
    if MODEL.exists() or (OUT / 'freeze.json').exists():
        raise ValueError('Model already frozen; refusing retraining in this experiment directory')
    train_seeds = list(range(201,231)) if train_seeds is None else list(train_seeds)
    selection = list(range(231,241)) if selection is None else list(selection)
    holdout = list(range(241,251)) if holdout is None else list(holdout)
    groups = [set(train_seeds),set(selection),set(holdout)]
    if any(not g for g in groups) or any(groups[i] & groups[j] for i,j in [(0,1),(0,2),(1,2)]):
        raise ValueError('training, selection and holdout fleets must be nonempty and disjoint')
    OUT.mkdir(parents=True,exist_ok=True)
    triples = [cached(s) for s in train_seeds]
    X = pd.concat([x for _, x, _ in triples], ignore_index=True)
    y = pd.concat([y for _, _, y in triples], ignore_index=True)
    assert set(y.seed.astype(int)) <= set(train_seeds)
    manifest = dict(protocol=protocol, training_seeds=train_seeds, selection_seeds=selection, holdout_seeds=holdout,
                    inputs={str(s): digest(fleet(s) / 'features.json') for s in train_seeds + selection})
    artifact = ld.fit(X, y, manifest)
    sel = [cached(s) for s in selection]
    base = pooled([p for p, _, _ in sel], selection)
    candidates = []
    best = base['fault_type_recall']
    for top in (.60, .70, .80, .90):
        for margin in (.10, .20, .30):
            gate = [top, margin]
            ps = [ld.relabel(p, x, artifact, gate=gate)[0] for p, x, _ in sel]
            r = pooled(ps, selection)
            valid = improves(base, r)
            candidates.append(dict(gate=gate, acceptable=valid, **{m:r[m] for m in METRICS}))
            if valid and r['fault_type_recall'] > best:
                artifact['gate'], best = gate, r['fault_type_recall']
    artifact['selection'] = dict(baseline={m:base[m] for m in METRICS}, candidates=candidates)
    with MODEL.open('wb') as f:
        pickle.dump(artifact, f)
    freeze = {k:v for k,v in artifact.items() if k != 'models'}
    freeze['models'] = {k:{z:v for z,v in fam.items() if z != 'model'} for k,fam in artifact['models'].items()}
    freeze['model_sha256'] = digest(MODEL)
    write_json(OUT / 'freeze.json', freeze)
    print(f'Frozen gate: {artifact["gate"]}', flush=True)

def frozen():
    m = json.loads((OUT / 'freeze.json').read_text())
    if digest(MODEL) != m['model_sha256']:
        raise ValueError('frozen model hash mismatch')
    artifact, error = ld.load_model(MODEL)
    if error:
        raise ValueError(error)
    return artifact

def bootstrap(base, new, n=2000):
    """Paired fleet bootstrap, using additive official-score sufficient statistics."""
    def metrics(rs):
        ne = sum(r['events_scored'] for r in rs)
        tp = sum(r['true_positive_events'] for r in rs)
        fa = sum(r['false_alert_episodes'] + r['excess_repeat_episodes'] for r in rs)
        days = sum(r['asset_days'] for r in rs)
        recall = tp / ne if ne else 0
        precision = tp / (tp + fa) if tp + fa else 0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0
        tr = sum(sum(e['type_correct'] for e in r['per_event']) for r in rs) / ne
        cr = sum(sum(e['class_correct'] for e in r['per_event']) for r in rs) / ne
        tl = sum(sum(e['timeliness'] for e in r['per_event']) for r in rs) / ne
        rate = fa / days
        return np.array([100 * (.35*f1 + .20*tl + .25*tr + .10*cr + .10*max(0, 1-rate/2)), tr, recall, rate])
    rng = np.random.default_rng(2026)
    draws = []
    for _ in range(n):
        ix = rng.integers(0, len(base), len(base))
        draws.append(metrics([new[i] for i in ix]) - metrics([base[i] for i in ix]))
    return {m:dict(zip(['low', 'high'], map(float, np.quantile(np.array(draws)[:,j], [.025,.975])))) for j,m in enumerate(METRICS)}

def evaluate():
    artifact = frozen()  # no holdout labels are accessed before this check
    seeds = artifact['training_manifest']['holdout_seeds']
    baselines, learned, br, lr, rows, diagnostics, accuracies, breakdown = [], [], [], [], [], [], [], []
    for s in seeds:
        p, x, y = cached(s)
        q, d = ld.relabel(p, x, artifact)
        # Even a fallback-only frozen gate should expose classifier scores for
        # review. This does not change acceptance or the submission predictions.
        for cls,family in artifact['models'].items():
            eligible = p.event_class == cls
            if cls == 'SENSOR':
                eligible &= p.fault_type.isin(ld.SENSOR)
            positions = np.flatnonzero(eligible.to_numpy())
            if len(positions):
                probabilities = family['model'].predict_proba(x.iloc[positions])
                for pos,prob in zip(positions,probabilities):
                    d.at[pos,'scores'] = json.dumps(dict(zip(family['model'].classes_,map(float,prob))),sort_keys=True)
                    d.at[pos,'candidate'] = str(family['model'].classes_[np.argmax(prob)])
        e, a, b = data(s)
        r0, r1 = score(p, e, a, b, 'stress'), score(q, e, a, b, 'stress')
        baselines.append(p); learned.append(q); br.append(r0); lr.append(r1)
        result_dir = OUT / f'stress_{s}'
        result_dir.mkdir(parents=True,exist_ok=True)
        q[SUBMIT].to_csv(result_dir / 'learned_predictions.csv', index=False)
        d.to_csv(result_dir / 'diagnostics.csv', index=False)
        diagnostics.append(d.assign(seed=s))
        for method, r in [('baseline',r0), ('learned',r1)]:
            rows.append(dict(seed=s, method=method, **{m:r[m] for m in METRICS}))
        ok = y.reason == 'matched'
        accuracies.extend([dict(seed=s, truth=t, baseline=bp, learned=lp, raw_classifier=raw,
                                fitting_eligible=str(eligible).lower() in ('true','1')) for t,bp,lp,raw,eligible in
                           zip(y.loc[ok,'target'], p.loc[ok,'fault_type'], q.loc[ok,'fault_type'],
                               d.loc[ok,'candidate'],y.loc[ok,'eligible'])])
        for (ctrl,tier), aa in a.groupby(['control','instrumentation_tier']):
            for method, pp in [('baseline',p), ('learned',q)]:
                rr = score(pp,e,aa,b,'stress')
                breakdown.append(dict(seed=s, control=ctrl,tier=tier,method=method,**{m:rr[m] for m in METRICS}))
    base, new = pooled(baselines,seeds), pooled(learned,seeds)
    combinations = sorted(set((ctrl,tier) for s in seeds for ctrl,tier in
                              data(s)[1][['control','instrumentation_tier']].itertuples(index=False,name=None)))
    for ctrl,tier in combinations:
        for method,ps in [('baseline',baselines),('learned',learned)]:
            rr = pooled(ps,seeds,(ctrl,tier))
            breakdown.append(dict(seed='aggregate',control=ctrl,tier=tier,method=method,**{m:rr[m] for m in METRICS}))
    ds, ac = pd.concat(diagnostics), pd.DataFrame(accuracies)
    raw_mask = ac.fitting_eligible & (ac.raw_classifier != '')
    report = dict(protocol=artifact['training_manifest'], baseline=base, learned=new, delta={m:new[m]-base[m] for m in METRICS},
                  paired_fleet_bootstrap_95=bootstrap(br,lr), bootstrap_replicates=2000,
                  override_coverage=float(ds.changed.mean()), fallback_frequency=float((~ds.accepted).mean()),
                  fallback_reasons=ds.loc[~ds.accepted,'fallback_reason'].value_counts(dropna=False).to_dict(),
                  unambiguous_alert_accuracy={m:float((ac.truth == ac[m]).mean()) for m in ('baseline','learned')},
                  raw_classifier_matched_accuracy=float((ac.loc[raw_mask,'truth'] == ac.loc[raw_mask,'raw_classifier']).mean()) if raw_mask.any() else None,
                  raw_classifier_matched_alerts=int(raw_mask.sum()),
                  acceptance_pass=bool(new['fault_type_recall']-base['fault_type_recall'] >= .05-1e-12 and
                    new['composite_score'] >= base['composite_score'] and new['event_recall'] >= base['event_recall'] and
                    new['false_alerts_per_asset_day'] <= base['false_alerts_per_asset_day']), model_sha256=digest(MODEL))
    write_json(OUT / 'holdout_report.json',report)
    pd.DataFrame(rows).to_csv(OUT / 'per_seed.csv',index=False)
    pd.DataFrame(breakdown).to_csv(OUT / 'control_tier.csv',index=False)
    ac.to_csv(OUT / 'matched_alerts.csv',index=False)
    for method, rr in [('baseline',base),('learned',new)]:
        pe = pd.DataFrame(rr['per_event'])
        pd.crosstab(pe.fault_type,pe.predicted_fault_type.fillna('MISSED'),dropna=False).to_csv(OUT / f'{method}_confusion.csv')
    lines = ['# Learned B4 holdout experiment', '', f'Acceptance target passed: {report["acceptance_pass"]}.',
             f'Protocol: {artifact["training_manifest"].get("protocol","full experiment")}.',
             f'Training seeds: {artifact["training_manifest"]["training_seeds"]}; selection: {artifact["training_manifest"]["selection_seeds"]}; holdout: {seeds}.',
             f'Frozen gate: {artifact["gate"]}. Confidence 0.6 is fixed and uncalibrated.', '',
             '| metric | B4 | learned | change |', '|---|---:|---:|---:|']
    lines += [f'| {m} | {base[m]:.6f} | {new[m]:.6f} | {new[m]-base[m]:+.6f} |' for m in METRICS]
    lines += ['', 'Paired fleet bootstrap 95% intervals (2,000 resamples):', '```json',
              json.dumps(report['paired_fleet_bootstrap_95'],indent=2), '```', '',
              f'Override coverage: {report["override_coverage"]:.3%}; fallback frequency: {report["fallback_frequency"]:.3%}.',
              f'Unambiguous matched-alert accuracy: {report["unambiguous_alert_accuracy"]}.',
              'Official matching may change after relabeling despite identical alert identities. See per_seed.csv, control_tier.csv, matched_alerts.csv and *_confusion.csv.',
              'The experiment remains opt-in regardless of outcome. No official test pumps were loaded.']
    lines += ['', 'Distinct eligible training events per class:', '```json',
              json.dumps({cls:family['counts'] for cls,family in artifact['models'].items()},indent=2), '```',
              f'Supported override classes: { {cls:family["supported"] for cls,family in artifact["models"].items()} }.',
              'Classes require at least ten distinct training events; no supported classes means fallback-only output.',
              f'Exploratory raw classifier accuracy on {report["raw_classifier_matched_alerts"]} eligible unambiguous matched alerts: {report["raw_classifier_matched_accuracy"]}. These raw diagnoses are not accepted overrides and do not enter official scores.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({m:report[m] for m in ['delta','acceptance_pass']},indent=2),flush=True)

def pilot():
    """Reuse eight completed fleets; never generate additional data or relax support gates."""
    global OUT, MODEL
    OUT = ROOT / 'build' / 'learned' / 'pilot_8'
    MODEL = OUT / 'model.pkl'
    training, selection, holdout = list(range(201,206)), [206], [207,208]
    protocol = '8-fleet pilot: 5 training / 1 selection / 2 holdout; not the original 50-fleet acceptance study'
    OUT.mkdir(parents=True,exist_ok=True)
    plan = dict(protocol=protocol, training_seeds=training,selection_seeds=selection,holdout_seeds=holdout,
                support_minimum_events=10, new_generation=False,
                limitations=['Holdout fleets were generated and feature/label caches prepared before designation; only aggregate cache sanity checks were inspected.',
                             'No model fitting, gate selection, or outcome comparison used these holdout fleets.',
                             'One selection fleet and two holdout fleets provide limited statistical evidence.'])
    path = OUT / 'pilot_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('existing pilot split differs; refusing to overwrite')
    write_json(path,plan)  # commit split before inspecting targets or fitting
    for s in training + selection + holdout:
        m = fleet(s) / 'generation.json'
        if not m.exists() or not json.loads(m.read_text()).get('complete'):
            raise ValueError(f'pilot requires already completed fleet {s}; generation is disabled')
    if not MODEL.exists():
        train(training,selection,holdout,protocol)
    else:
        a = frozen()['training_manifest']
        if (a['training_seeds'],a['selection_seeds'],a['holdout_seeds']) != (training,selection,holdout):
            raise ValueError('frozen pilot split mismatch')
    evaluate()
    with (OUT / 'REPORT.md').open('a',encoding='utf-8') as f:
        f.write('\n\nPilot limitations:\n\n' + '\n'.join('- ' + s for s in plan['limitations']))

def regression():
    artifact = frozen()
    reports = {}
    for s in range(101,105):
        generate(s); extract(s)
        p,x,_ = cached(s)
        q,d = ld.relabel(p,x,artifact)
        e,a,b = data(s)
        reports[str(s)] = dict(baseline=score(p,e,a,b,'stress'),learned=score(q,e,a,b,'stress'))
        q[SUBMIT].to_csv(fleet(s) / 'learned_predictions.csv',index=False)
        d.to_csv(fleet(s) / 'diagnostics.csv',index=False)
    import eval as ev
    inputs = b4.load_inputs(str(ROOT.parent / 'data' / 'full'),[f'PUMP-0{i}' for i in range(1,10)])
    p,frames = b4.run_b4(inputs,return_frames=True)
    q,d = ld.relabel(p,ld.features(inputs,p,frames),artifact)
    q[SUBMIT].to_csv(OUT / 'official_trainval_learned.csv',index=False)
    d.to_csv(OUT / 'official_trainval_diagnostics.csv',index=False)
    reports['official_trainval'] = dict(baseline=ev.score_splits(p),learned=ev.score_splits(q))
    original = pd.read_csv(ROOT / 'results' / 'b4_trainval_predictions.csv')
    try:
        pd.testing.assert_frame_equal(p,original,check_dtype=False)
        reports['original_runner_predictions_unchanged'] = True
    except AssertionError as exc:
        reports['original_runner_predictions_unchanged'] = str(exc)
    write_json(OUT / 'regression_report.json',reports)
    lines = ['\n## Post-freeze regression checks', '', '| fleet | composite change | type recall change | event recall change | false+excess/day change |',
             '|---|---:|---:|---:|---:|']
    for name,r in reports.items():
        if name == 'original_runner_predictions_unchanged':
            continue
        pairs = [(name,r['baseline'],r['learned'])] if name != 'official_trainval' else [
            (f'{name}:{s}',r['baseline'][s],r['learned'][s]) for s in ('train','validation')]
        for label,base,new in pairs:
            lines.append('| ' + label + ' | ' + ' | '.join(f'{new[m]-base[m]:+.6f}' for m in METRICS) + ' |')
    lines += ['', f'Frozen B4 prediction equality: {reports["original_runner_predictions_unchanged"]}.',
              'Regression failures do not change the frozen model or gate.']
    with (OUT / 'REPORT.md').open('a',encoding='utf-8') as f:
        f.write('\n'.join(lines))
