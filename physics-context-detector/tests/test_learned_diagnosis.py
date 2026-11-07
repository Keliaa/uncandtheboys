"""Safety and causal contracts for the opt-in diagnosis layer."""
from pathlib import Path
import sys
import copy
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT.parent / 'original-challenge' / 'src')]
import b4
import learned_diagnosis as ld
import learned_evaluation as le
import learned_experiment as experiment
from test_b4_causality import truncate

def predictions():
    return pd.DataFrame([dict(asset_id='A', alert_time_utc='2026-01-01T02:00:00Z', event_class='SENSOR',
                             fault_type='SENSOR_DRIFT', confidence=.7, detector='single',
                             evidence='discharge pressure deviates alone: +2', recommended_action='old')])

def assets():
    return pd.DataFrame([dict(asset_id='A',window_start_utc='2026-01-01T00:00:00Z',window_end_utc='2026-01-08T00:00:00Z')])

def event(eid='e', **kw):
    return dict(event_id=eid,asset_id='A',onset_utc='2026-01-01T00:00:00Z',detectable_utc='2026-01-01T01:00:00Z',
                end_utc='',event_class='MACHINE',fault_type='BEARING_WEAR',observable=True,**kw)

def test_targets_do_not_use_predicted_type_and_weight_events():
    p = pd.concat([predictions()]*3,ignore_index=True)
    y = ld.targets(p,pd.DataFrame([event()]),assets(),pd.DataFrame(),201)
    assert set(y.target) == {'BEARING_WEAR'}
    assert y.weight.sum() == pytest.approx(1.)
    y2 = ld.targets(p,pd.DataFrame([event(),event('other')]),assets(),pd.DataFrame(),201)
    assert set(y2.reason) == {'overlap'}
    assert y2.weight.sum() == 0

def test_exclusions():
    p = predictions()
    e = event(); e['observable'] = False
    assert ld.targets(p,pd.DataFrame([e]),assets(),pd.DataFrame(),201).reason.iloc[0] == 'unobservable'
    bn = pd.DataFrame([dict(asset_id='A',start_utc='2026-01-01T00:00:00Z',end_utc='2026-01-02T00:00:00Z')])
    assert ld.targets(p,pd.DataFrame([event()]),assets(),bn,201).reason.iloc[0] == 'benign'
    p.alert_time_utc = '2025-12-31T00:00:00Z'
    assert ld.targets(p,pd.DataFrame([event()]),assets(),pd.DataFrame(),201).reason.iloc[0] == 'unmatched'

class Scores:
    classes_ = np.array(['SENSOR_CALIBRATION_SHIFT','SENSOR_DRIFT'])
    def predict_proba(self,X):
        return np.array([[.95,.05]])

def artifact(X):
    return dict(version=ld.VERSION,versions=ld.versions(),b4_hash=ld.b4_hash(),feature_hash=ld.feature_hash(),schema=list(X),gate=[.6,.1],
                models={'SENSOR':dict(model=Scores(),supported=list(Scores.classes_),combinations=[('FIXED_SPEED','T2')])})

def feature_row():
    return pd.DataFrame([dict(control='FIXED_SPEED',tier='T2',sensor_channel='pd',e_pd_now=2.,z_pd_now=6.)])

def test_override_and_fallback_contract(tmp_path):
    p,x = predictions(),feature_row()
    a = artifact(x)
    q,d = ld.relabel(p,x,a)
    pd.testing.assert_frame_equal(p[ld.IDENTITY],q[ld.IDENTITY])
    assert len(q) == len(p) and q.confidence.iloc[0] == .6
    assert q.fault_type.iloc[0] == 'SENSOR_CALIBRATION_SHIFT'
    assert 'discharge pressure' in q.evidence.iloc[0]
    assert 'discharge pressure transmitter' in q.recommended_action.iloc[0]
    assert d.sensor_channel.iloc[0] == 'pd' and d.changed.iloc[0]
    empty = x.copy(); empty[['e_pd_now','z_pd_now']] = np.nan
    qq,dd = ld.relabel(p,empty,a)
    pd.testing.assert_frame_equal(p,qq)
    assert dd.fallback_reason.iloc[0] == 'no_measured_evidence'
    unattributed = p.copy(); unattributed.evidence = 'unattributed deviation'
    qq,dd = ld.relabel(unattributed,x,a)
    pd.testing.assert_frame_equal(unattributed,qq)
    assert dd.fallback_reason.iloc[0] == 'missing_sensor_attribution'
    for field,value,reason in [('supported',[],'unsupported_class'),('combinations',[],'unsupported_control_tier')]:
        aa = copy.deepcopy(a); aa['models']['SENSOR'][field] = value
        qq,dd = ld.relabel(p,x,aa)
        pd.testing.assert_frame_equal(p,qq)
        assert dd.fallback_reason.iloc[0] == reason
    aa = copy.deepcopy(a); aa['gate'] = None
    pd.testing.assert_frame_equal(p,ld.relabel(p,x,aa)[0])
    aa = copy.deepcopy(a); aa['version'] = -1
    with pytest.warns(RuntimeWarning):
        pd.testing.assert_frame_equal(p,ld.relabel(p,x,aa)[0])
    with pytest.warns(RuntimeWarning,match='schema'):
        pd.testing.assert_frame_equal(p,ld.relabel(p,x.assign(unexpected=1),a)[0])
    with pytest.warns(RuntimeWarning):
        aa,error = ld.load_model(tmp_path / 'missing.pkl')
    pd.testing.assert_frame_equal(p,ld.relabel(p,None,aa,error)[0])
    for ft in ['SENSOR_STUCK','COMMUNICATION_LOSS','SENSOR_NOISE_SPIKES']:
        pp = p.copy(); pp.fault_type = ft
        pd.testing.assert_frame_equal(pp,ld.relabel(pp,x,a)[0])

def test_preprocessing_and_minimum_distinct_events():
    x = pd.concat([feature_row()]*40,ignore_index=True)
    x['all_missing'] = np.nan
    y = pd.DataFrame([dict(reason='matched',target_class='SENSOR',target='SENSOR_DRIFT' if i<20 else 'SENSOR_CALIBRATION_SHIFT',
                           seed=201,asset_id='A',event_id=f'e{i//4}',weight=.25) for i in range(40)])
    a = ld.fit(x,y,{'training_seeds':[201]})
    assert a['models']['SENSOR']['supported'] == []  # five distinct events per class
    assert a['models']['SENSOR']['model'].predict_proba(x).shape == (40,2)
    y.loc[0,'seed'] = 231
    with pytest.raises(ValueError,match='split isolation'):
        ld.fit(x,y,{'training_seeds':[201]})

@pytest.mark.parametrize('asset',['PUMP-04','PUMP-06','PUMP-03'])
def test_causal_features_and_predictions(asset):
    inp = b4.load_inputs(str(ROOT.parent / 'data' / 'full'),[asset])
    p,frames = b4.run_b4(inp,return_frames=True)
    if not len(p):
        pytest.skip('no alerts')
    x = ld.features(inp,p,frames)
    cut = pd.Timestamp(p.alert_time_utc.iloc[len(p)//2]) + pd.Timedelta(seconds=17)
    partial = truncate(inp,cut)
    pp,ff = b4.run_b4(partial,return_frames=True)
    mask = pd.to_datetime(p.alert_time_utc,utc=True) <= cut
    pp = pp[pd.to_datetime(pp.alert_time_utc,utc=True)<=cut].reset_index(drop=True)
    xx = ld.features(partial,pp,ff)
    pd.testing.assert_frame_equal(x.loc[mask].reset_index(drop=True),xx,check_dtype=False)
    # Exercise fitted predictions even before a production experiment artifact exists.
    y = pd.DataFrame([dict(reason='matched',target_class=c,target=t,seed=201,asset_id='A',event_id=str(i),weight=1.)
                      for i,(c,t) in enumerate(zip(p.event_class,p.fault_type))])
    a = ld.fit(x,y,{})
    a['gate'] = [.6,.1]
    for family in a['models'].values():
        family['supported'] = list(family['model'].classes_)
    q,_ = ld.relabel(p,x,a)
    qq,_ = ld.relabel(pp,xx,a)
    pd.testing.assert_frame_equal(q.loc[mask].reset_index(drop=True),qq,check_dtype=False)
    missing = dict(inp); missing['snaps'] = None
    xm = ld.features(missing,p,frames)
    assert not xm.snapshot_available.any()
    for col in x.columns:
        assert col not in {'seed','asset_id','event_id','alert_time_utc','target','fault_type'}
    available, _ = b4.static_channels(inp['assets'].iloc[0])
    for k in b4.CH:
        if k not in available:
            assert not x[f'available_{k}'].any()
            assert x[f'e_{k}_now'].isna().all()

def test_gate_constraints():
    base = dict(composite_score=70.,fault_type_recall=.7,event_recall=.8,false_alerts_per_asset_day=.1)
    good = dict(base,fault_type_recall=.75)
    assert le.improves(base,good)
    for key,val in [('composite_score',69.),('event_recall',.79),('false_alerts_per_asset_day',.11)]:
        assert not le.improves(base,dict(good,**{key:val}))

def test_pilot_split_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(le,'OUT',tmp_path)
    monkeypatch.setattr(le,'MODEL',tmp_path / 'model.pkl')
    for train,selection,holdout in [([201],[201],[208]),([201],[206],[201]),([201],[],[208])]:
        with pytest.raises(ValueError,match='disjoint'):
            le.train(train,selection,holdout,'test pilot')

def test_generation_refuses_unowned_or_incompatible_fleets(tmp_path, monkeypatch):
    d = tmp_path / 'stress_201'
    d.mkdir()
    sentinel = d / 'sentinel.txt'
    sentinel.write_text('existing data')
    monkeypatch.setattr(experiment, 'fleet', lambda seed:d)
    with pytest.raises(ValueError,match='incompatible'):
        experiment.generate(201)
    experiment.write_json(d / 'generation.json',dict(settings={'seed':999},complete=True))
    with pytest.raises(ValueError,match='incompatible'):
        experiment.generate(201)
    assert sentinel.read_text() == 'existing data'
