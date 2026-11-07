"""R01: action table. Adds the recommended action to each alert record.

The table is in the machine file (worklist 7.1). Low certainty gives the UNKNOWN action.
"""
from __future__ import annotations

from typing import Any, Iterable

from ..alert import AlertRecord
from ..config import action_for


def add_actions(alerts: Iterable[AlertRecord], machine: dict[str, Any]) -> list[AlertRecord]:
    out = []
    for a in alerts:
        ft = "UNKNOWN" if a.certainty == "Low" else a.fault_type
        act = action_for(machine, ft)
        a.recommended_action = act["text"]
        a.what_if = None if act.get("sandbox_action") in (None, "none") else {
            "action": act["sandbox_action"], "hypothesis": act["sandbox_hypothesis"]}   # R02 fills the result
        out.append(a)
    return out
