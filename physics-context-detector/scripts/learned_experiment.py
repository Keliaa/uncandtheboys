#!/usr/bin/env python3
"""Sequential, resumable stress-fleet experiment. Never loads official test pumps."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'original-challenge' / 'src'), str(ROOT / 'scripts')]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)

def fleet(seed):
    return ROOT / 'build' / 'learned' / f'stress_{seed}'

def generate(seed):
    d = fleet(seed)
    settings = dict(seed=seed, days=7, assets=8, snapshots=True, tier='stress',
                    source={p.name: digest(p) for p in sorted((ROOT.parent / 'original-challenge' / 'src' / 'retrofit_sim').glob('*.py'))})
    m = d / 'generation.json'
    if d.exists():
        if not m.exists() or json.loads(m.read_text())['settings'] != settings:
            raise ValueError(f'Refusing incompatible existing fleet: {d}')
        if json.loads(m.read_text()).get('complete'):
            return
        # Dataset writer safely replaces only this manifest-owned incomplete fleet.
    d.mkdir(parents=True, exist_ok=True)
    write_json(m, dict(settings=settings, complete=False))
    subprocess.run([sys.executable, str(ROOT.parent / 'original-challenge' / 'make_dataset.py'),
                    '--tier', 'stress', '--seed', str(seed), '--stress-assets', '8', '--days', '7', '--output', str(d)], check=True)
    write_json(m, dict(settings=settings, complete=True))

def extract(seed):
    import pandas as pd
    import b4
    import learned_diagnosis as ld
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    d = fleet(seed)
    manifest = dict(generation=digest(d / 'generation.json'), b4=digest(ROOT / 'b4.py'),
                    features=digest(ROOT / 'learned_diagnosis.py'))
    cache = d / 'features.json'
    if cache.exists() and json.loads(cache.read_text()) == manifest:
        return
    assets = pd.read_csv(d / 'public' / 'assets.csv')
    inputs = b4.load_inputs(str(d / 'public'), assets.asset_id)
    pred, frames = b4.run_b4(inputs, return_frames=True)
    X = ld.features(inputs, pred, frames)
    labels = pd.read_csv(d / 'private' / 'events_stress_all.csv')
    benign = pd.read_csv(d / 'public' / 'benign_windows_stress.csv')
    target = ld.targets(pred, labels, assets, benign, seed)
    pred.to_csv(d / 'baseline.csv', index=False)
    X.to_parquet(d / 'features.parquet', index=False)
    target.to_csv(d / 'targets.csv', index=False)
    write_json(cache, manifest)

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'regression', 'all', 'pilot'])
    p.add_argument('--start', type=int, default=201)
    p.add_argument('--end', type=int, default=240)
    a = p.parse_args()
    if a.stage == 'pilot':
        import learned_evaluation as le
        le.pilot()
        return
    if a.stage in ('prepare', 'all'):
        if not 201 <= a.start <= a.end <= 250:
            p.error('experiment seeds must be within 201..250')
        if a.end > 240 and not (ROOT / 'build' / 'learned' / 'freeze.json').exists():
            p.error('freeze the model and gate before preparing holdout alerts')
        for seed in range(a.start, a.end + 1):
            print(f'Preparing seed {seed}', flush=True)
            generate(seed)
            extract(seed)
    if a.stage in ('train', 'all'):
        import learned_evaluation as le
        le.train()
    if a.stage in ('evaluate', 'all'):
        import learned_evaluation as le
        if a.stage == 'all':
            for seed in range(241,251):
                print(f'Preparing frozen holdout seed {seed}', flush=True)
                generate(seed)
                extract(seed)
        le.evaluate()
    if a.stage in ('regression', 'all'):
        import learned_evaluation as le
        le.regression()

if __name__ == '__main__':
    main()
