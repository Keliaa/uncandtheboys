"""Alert management by episodes (idea from the B4 detector, physics-context-detector/EXPERIMENTS.md #3).

A developing problem should raise one alert, not one per fault type it passes through. Per pump and per fault family
(hydraulic or mechanical) the first machine alert opens an episode. While it is open, later alerts of that family are suppressed,
except escalation types (cavitation, air entrainment) that were not yet alerted in the episode. An episode closes after
`quiet_min` minutes of steady running without a detector flag; quiet time is counted only while the pump runs steadily, so a stop
(e.g. for a repair) neither closes nor extends it. Sensor alerts are not touched (they have their own repeat memory).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HYDRAULIC = {"SUCTION_STRAINER_CLOGGING", "DISCHARGE_RESTRICTION", "IMPELLER_WEAR", "CAVITATION", "AIR_ENTRAINMENT"}
MECHANICAL = {"BEARING_WEAR", "MISALIGNMENT", "IMBALANCE"}
ESCALATION = {"CAVITATION", "AIR_ENTRAINMENT"}
QUIET_MIN = 720


def family(fault_type: str) -> str:
    return "hydraulic" if fault_type in HYDRAULIC else "mechanical" if fault_type in MECHANICAL else "other"


def manage_episodes(alerts, det: pd.DataFrame, quiet_min: int = QUIET_MIN, sources=("d01",)):
    """Returns the alerts with the suppressed ones removed. Alerts from sources not listed pass through unchanged."""
    keep, by_track = [], {}
    for a in alerts:
        if a.event_class != "MACHINE" or a.source not in sources:
            keep.append(a)
            continue
        by_track.setdefault((a.asset_id, family(a.fault_type)), []).append(a)
    for (aid, _fam), items in by_track.items():
        g = det[det.asset_id == aid].sort_values("window_start_utc")
        tser = g["available_utc"].reset_index(drop=True)
        steady, flag = g["steady"].to_numpy(bool), g["flag"].to_numpy(bool)
        qrun = np.zeros(len(g), dtype=int)          # steady quiet minutes in a row; a flag resets it, a stop leaves it unchanged
        run = 0
        for i in range(len(g)):
            if steady[i] and flag[i]:
                run = 0
            elif steady[i]:
                run += 1
            qrun[i] = run
        open_ep, alerted, start = False, set(), 0
        for a in sorted(items, key=lambda x: x.alert_time_utc):
            j = int(tser.searchsorted(pd.Timestamp(a.alert_time_utc), side="right")) - 1        # last minute available at the alert time
            if open_ep and j > start and qrun[start + 1: j + 1].max() >= quiet_min:
                open_ep, alerted = False, set()                  # the problem went quiet for long enough: a new alert is a new episode
            if not open_ep:
                open_ep, alerted, start = True, {a.fault_type}, max(j, 0)
                keep.append(a)
            elif a.fault_type in ESCALATION and a.fault_type not in alerted:
                alerted.add(a.fault_type)
                keep.append(a)
            # else: suppressed (the same developing problem)
    return sorted(keep, key=lambda x: (x.asset_id, x.alert_time_utc))
