#!/usr/bin/env python3
"""Opt-in relabeling; explicit trusted model path required; never loads official test pumps."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'original-challenge' / 'src')]
import b4
import learned_diagnosis as ld

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', type=Path, default=ROOT.parent / 'data' / 'full')
    p.add_argument('--assets', nargs='+', default=[f'PUMP-0{i}' for i in range(1, 10)])
    p.add_argument('--out', type=Path, default=ROOT / 'build' / 'learned' / 'runner')
    a = p.parse_args()
    if set(a.assets) & {'PUMP-10', 'PUMP-11', 'PUMP-12'}:
        p.error('official test pumps are excluded from this experiment')
    out = a.out.resolve()
    if not out.is_relative_to((ROOT / 'build').resolve()):
        p.error('--out must be under the detector build directory')
    out.mkdir(parents=True, exist_ok=True)
    inputs = b4.load_inputs(str(a.data), a.assets)
    baseline, frames = b4.run_b4(inputs, return_frames=True)
    artifact, failure = ld.load_model(a.model)
    X = ld.features(inputs, baseline, frames) if artifact else None
    learned, diagnostics = ld.relabel(baseline, X, artifact, failure)
    cols = ['asset_id', 'alert_time_utc', 'event_class', 'fault_type', 'confidence', 'evidence', 'recommended_action']
    learned[cols].to_csv(out / 'learned_predictions.csv', index=False)
    diagnostics.to_csv(out / 'diagnostics.csv', index=False)
    (out / 'run.json').write_text(json.dumps(dict(model=str(Path(a.model).resolve()), fallback=failure,
                                                confidence='0.6 fixed, uncalibrated'), indent=2))
    print(f'{len(learned)} alerts; {diagnostics.changed.sum() if len(diagnostics) else 0} changed; {out}')

if __name__ == '__main__':
    main()
