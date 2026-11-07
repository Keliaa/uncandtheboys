"""R02: what-if check in the sealed sandbox (simulation only).

For a machine alert, ask the reduced-order pump model (original-challenge sandbox) what the recommended action
would change, using the sandbox action and hypothesis from the action table (machine file, 7.1). The machine
condition is a hypothesis, never the true state. Nothing is written to a pump; the function returns data only.
Sensor alerts and actions without a sandbox action get no what-if result.
"""
from __future__ import annotations

from typing import Any, Iterable

from .. import _ref  # noqa: F401
from ..alert import AlertRecord
from ..config import action_for
from ..splits import DATA_FULL

SPEED_STEP = 0.10     # speed_setpoint what-if: reduce speed by this ratio ("decrease the flow if operation permits")


def _hyp(text: str) -> dict[str, float]:
    h = {}
    for item in str(text).split():
        k, _, v = item.partition("=")
        if v:
            h[k.strip()] = float(v)
    return h


def what_if_for(alert: AlertRecord, machine: dict[str, Any], data=DATA_FULL) -> dict[str, Any] | None:
    if alert.event_class != "MACHINE":
        return None
    ftype = "UNKNOWN" if alert.certainty == "Low" else alert.fault_type
    act = action_for(machine, ftype)
    action = act.get("sandbox_action", "none")
    if action in (None, "none"):
        return None
    from retrofit_sim.sandbox import what_if
    hyp = _hyp(act.get("sandbox_hypothesis", ""))
    value = None
    if action == "speed_setpoint":      # needs a target: current speed ratio minus a step
        base = what_if(str(data), alert.asset_id, alert.alert_time_utc, "none", None, hyp)
        sr = (base.get("baseline") or {}).get("speed_ratio")
        value = round(sr - SPEED_STEP, 3) if sr else None
    r = what_if(str(data), alert.asset_id, alert.alert_time_utc, action, value, hyp)
    out: dict[str, Any] = {"action": action, "value": value, "hypothesis": hyp, "accepted": bool(r.get("accepted")),
                           "reasons": r.get("rejection_reasons", []), "warnings": r.get("warnings", []),
                           "caveats": r.get("caveats", [])}
    if "baseline" in r:
        b, p = r["baseline"], r["proposed"]
        out.update(flow_before_m3h=b["flow_m3h"], flow_after_m3h=p["flow_m3h"],
                   flow_change_pct=round(100.0 * (p["flow_m3h"] - b["flow_m3h"]) / max(b["flow_m3h"], 1e-6), 1),
                   risk_before=b["risk_proxy"], risk_after=p["risk_proxy"],
                   cost_eur=r["intervention"]["estimated_cost_eur"])
    return out


def add_what_if(alerts: Iterable[AlertRecord], machine: dict[str, Any], data=DATA_FULL) -> list[AlertRecord]:
    out = []
    for a in alerts:
        a.what_if = what_if_for(a, machine, data)
        out.append(a)
    return out
