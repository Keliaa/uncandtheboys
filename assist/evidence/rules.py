"""E01: rule table with a simple confidence.

Reads the symptom table from the machine file (S02). For each fault type the pump can show,
count how many of its symptoms match the z-scores at the alert minute. Pick the type with the most
matches (ties: higher mean |z|). confidence = matched / evaluable symptoms, reduced when |z| is small.
"""
from __future__ import annotations

from typing import Any

from ..alert import Evidence, UNKNOWN
from ..config import Asset, action_for, usable_symptoms

MATCH_FRAC = 0.5      # a symptom matches when |z| > MATCH_FRAC * k in the right direction
FLAT_FRAC = 1 / 3     # "unchanged" means |z| < FLAT_FRAC * k
SENSOR_ALONE = 1 / 3  # sensor suspicion: related channels all below this fraction of k


def certainty_label(p: float, machine: dict[str, Any]) -> str:
    c = machine.get("certainty", {"high": 0.70, "medium": 0.40})
    return "High" if p >= c["high"] else "Medium" if p >= c["medium"] else "Low"


def _match(direction: str, z: float, k: float) -> bool | None:
    """True/False, or None when the direction cannot be evaluated from a z-score."""
    if direction == "up":
        return z > MATCH_FRAC * k
    if direction == "down":
        return z < -MATCH_FRAC * k
    if direction == "strong_down":
        return z < -k
    if direction == "unchanged":
        return abs(z) < FLAT_FRAC * k
    return None


def _words(channel: str, z: float) -> str:
    name = channel.replace("_", " ")
    for suf in (" kpa", " c", " a", " rpm", " kw", " m3 h", " mm s", " g", " pct"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return f"{name} is {'high' if z > 0 else 'low'} (z = {z:+.1f})"


def classify(z: dict[str, float], asset: Asset, machine: dict[str, Any], k: float) -> dict[str, Any]:
    """z: channel name -> z-score at the alert minute. Returns the diagnosis fields of an alert record."""
    dev = {c: v for c, v in z.items() if abs(v) > k}
    dev_text = [Evidence(c, v, _words(c, v)) for c, v in sorted(dev.items(), key=lambda kv: -abs(kv[1]))[:3]]
    scores: dict[str, float] = {}
    classes: dict[str, str] = {}
    for ftype, ev in usable_symptoms(machine, asset).items():
        cls = machine["symptoms"][ftype]["class"]
        if cls == "SENSOR":
            continue
        got = [(e, _match(e["direction"], z[e["channel"]], k)) for e in ev if e["channel"] in z]
        got = [(e, m) for e, m in got if m is not None]
        hits = [e for e, m in got if m]
        if not hits or not any(e["direction"] != "unchanged" for e in hits):
            continue
        mean_z = sum(abs(z[e["channel"]]) for e in hits) / len(hits)
        scores[ftype] = (len(hits) / len(got)) * min(1.0, mean_z / (1.5 * k))
        classes[ftype] = cls
    # Sensor rule: one channel deviates while all its related channels stay quiet.
    related = machine.get("related_channels", {})
    for c, v in dev.items():
        rel = [r for r in related.get(c, []) if r in z]
        if rel and max(abs(z[r]) for r in rel) < SENSOR_ALONE * k and len(dev) == 1:
            scores["SENSOR_CALIBRATION_SHIFT"] = min(1.0, abs(v) / (1.5 * k))
            classes["SENSOR_CALIBRATION_SHIFT"] = "SENSOR"
            dev_text = [Evidence(c, v, f"{c.replace('_', ' ')} deviates alone (z = {v:+.1f}); related channels are normal")]
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        return dict(event_class="MACHINE", fault_type=UNKNOWN, probability=0.0, certainty="Low", top_types=[],
                    evidence=dev_text, recommended_action=action_for(machine, UNKNOWN)["text"])
    best, p = ranked[0]
    label = certainty_label(p, machine)
    if label == "Low":     # Low certainty -> UNKNOWN (worklist section 6)
        ft, cls = UNKNOWN, classes[best]
    else:
        ft, cls = best, classes[best]
    return dict(event_class=cls, fault_type=ft, probability=float(p), certainty=label,
                top_types=[(t, float(s)) for t, s in ranked[:3]], evidence=dev_text,
                recommended_action=action_for(machine, ft)["text"])


def apply_classifier(d: dict[str, Any], probs: dict[str, float], asset: Asset, machine: dict[str, Any]) -> dict[str, Any]:
    """E02: replace the rule-based type and confidence with classifier probabilities (machine faults only).

    Types the pump cannot show (symptom `needs` not met) are removed and the rest renormalised.
    """
    allowed = set(usable_symptoms(machine, asset))
    probs = {t: p for t, p in probs.items() if t in allowed}
    tot = sum(probs.values())
    if tot <= 0:
        return d
    ranked = sorted(((t, p / tot) for t, p in probs.items()), key=lambda kv: -kv[1])
    best, p = ranked[0]
    label = certainty_label(p, machine)
    ft = UNKNOWN if label == "Low" else best
    out = dict(d)
    out.update(fault_type=ft, probability=float(p), certainty=label, top_types=[(t, float(q)) for t, q in ranked[:3]],
               recommended_action=action_for(machine, ft)["text"])
    return out
