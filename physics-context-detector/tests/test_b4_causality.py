"""Causality check for detector B4 (BENCHMARK_RULES.md section 2).

An alert raised at time t may only depend on data with timestamp <= t (plus static assets.csv). The test runs B4 on a
pump's full week, then again on inputs truncated at 3 random cut times (features whose minute has fully elapsed by the
cut, context rows and vibration snapshots stamped <= cut). Every alert at or before each cut must be identical
(time, class, type, confidence, evidence, action) in both runs.

Run:  python -m pytest -q tests/test_b4_causality.py      (or: python tests/test_b4_causality.py)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "original-challenge", "src"))
sys.path.insert(0, ROOT)

import b4  # noqa: E402

DATA = os.path.join(os.path.dirname(ROOT), "data", "full")
# development pumps only: a VFD pressure-controlled fully instrumented pump with snapshots, a wireless-retrofit
# process pump with cavitation episodes, and a legacy fixed-speed pump (test pumps are never used here)
PUMPS = ["PUMP-04", "PUMP-06", "PUMP-03"]


def truncate(inp: dict, cut: pd.Timestamp) -> dict:
    out = dict(inp)
    f = inp["feat"].copy()
    ws = pd.to_datetime(f["window_start_utc"], utc=True)
    out["feat"] = f[ws + pd.Timedelta(minutes=1) <= cut]          # the minute [w, w+1) is complete at w + 1 min
    c = inp["ctx"].copy()
    out["ctx"] = c[pd.to_datetime(c["timestamp_utc"], utc=True) <= cut]
    if inp.get("snaps") is not None:
        s = inp["snaps"]
        out["snaps"] = s[pd.to_datetime(s["timestamp_utc"], utc=True) <= cut]
    return out


def _before(pred: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    t = pd.to_datetime(pred["alert_time_utc"], utc=True)
    return pred[t <= cut].reset_index(drop=True)


@pytest.mark.parametrize("asset", PUMPS)
def test_alerts_identical_before_cut(asset):
    if not os.path.exists(os.path.join(DATA, "features_1min.parquet")):
        pytest.skip("full data tier not present")
    inp = b4.load_inputs(DATA, [asset])
    full = b4.run_b4(inp)
    a = inp["assets"].iloc[0]
    ws, we = pd.Timestamp(a.window_start_utc), pd.Timestamp(a.window_end_utc)
    rng = np.random.default_rng([2026, int(asset[-2:])])
    span_min = int((we - ws).total_seconds() // 60)
    cuts = sorted(ws + pd.Timedelta(minutes=int(m)) + pd.Timedelta(seconds=int(s))
                  for m, s in zip(rng.integers(6 * 60, span_min - 60, 3), rng.integers(0, 60, 3)))
    for cut in cuts:
        part = b4.run_b4(truncate(inp, cut))
        a_full, a_part = _before(full, cut), _before(part, cut)
        assert len(a_full) == len(a_part), f"{asset} cut {cut}: {len(a_full)} alerts in full run, {len(a_part)} truncated"
        pd.testing.assert_frame_equal(a_full, a_part, check_dtype=False, obj=f"{asset} alerts before {cut}")
        print(f"{asset}: cut {cut} -> {len(a_full)} alerts before the cut identical")


if __name__ == "__main__":
    for p in PUMPS:
        test_alerts_identical_before_cut(p)
    print("causality OK")
