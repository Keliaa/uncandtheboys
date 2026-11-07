"""Presentation adapter: B4 alerts -> operator action cards.

B4's prediction rows are never modified. This module adds, for display only:

* a plain-language check title per (pump, recommended check), used to group repeated alerts into one card;
* a short "Why this check?" built only from facts stated in the alert's own evidence text (every sentence keeps a
  reference to the exact evidence span it came from, see `Observation.span`);
* task status from local operator interactions (New -> Acknowledged -> Check recorded).

Task status is never fed back into monitoring: acknowledging or recording a check does not hide alerts, evidence or
future occurrences. The detector's confidence value is deliberately not shown (it is not calibrated).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import channels as chn
from .replay import fmt, ts

# ------------------------------------------------------------------------------------------ recommended checks
MACHINE_CHECKS = {
    "SUCTION_STRAINER_CLOGGING": ("suction-strainer", "Check the suction strainer",
                                  "This pattern can indicate a partly blocked suction strainer."),
    "DISCHARGE_RESTRICTION": ("discharge-side", "Check the discharge side for a restriction",
                              "This pattern can indicate a restriction on the discharge side, such as a partly "
                              "closed valve or a fouled heat exchanger."),
    "IMPELLER_WEAR": ("pump-performance", "Check pump performance against its curve",
                      "This pattern can indicate reduced pump performance, such as impeller or wear-ring wear."),
    "CAVITATION": ("suction-conditions", "Check suction conditions for cavitation",
                   "This pattern can indicate cavitation at the pump inlet."),
    "AIR_ENTRAINMENT": ("air-ingress", "Check for air entering the suction",
                        "This pattern can indicate air being drawn into the suction."),
    "BEARING_WEAR": ("bearing", "Check the pump bearing", "This pattern can indicate bearing wear."),
    "MISALIGNMENT": ("alignment", "Check shaft alignment and coupling",
                     "This pattern can indicate shaft misalignment."),
    "IMBALANCE": ("impeller-balance", "Check the impeller for fouling or imbalance",
                  "This pattern can indicate impeller fouling or imbalance."),
    "COMMUNICATION_LOSS": ("data-link", "Check the monitoring data link",
                           "The pump's condition cannot be assessed while data are missing."),
    "UNKNOWN": ("general-inspection", "Inspect the pump",
                "The pattern does not clearly match a known cause."),
}
SENSOR_CHECKS = {
    # fault type -> (check family, title template, meaning)
    "SENSOR_DRIFT": ("verify-reading", "Verify the {c} reading",
                     "This points to the measurement rather than the machine."),
    "SENSOR_CALIBRATION_SHIFT": ("verify-reading", "Verify the {c} reading",
                                 "This points to the measurement rather than the machine."),
    "SENSOR_STUCK": ("frozen-reading", "Check the {c} sensor: reading appears frozen",
                     "This points to the sensor, its cabling or transmitter rather than the machine."),
    "SENSOR_NOISE_SPIKES": ("noisy-reading", "Check the {c} sensor signal: reading is noisy",
                            "This points to the sensor, its cabling or transmitter rather than the machine."),
}
FAULT_PLAIN = {
    "SUCTION_STRAINER_CLOGGING": "Suction strainer clogging", "DISCHARGE_RESTRICTION": "Discharge restriction",
    "IMPELLER_WEAR": "Impeller wear", "CAVITATION": "Cavitation", "AIR_ENTRAINMENT": "Air entrainment",
    "BEARING_WEAR": "Bearing wear", "MISALIGNMENT": "Misalignment", "IMBALANCE": "Imbalance",
    "SENSOR_DRIFT": "Instrument drift (gradual)", "SENSOR_CALIBRATION_SHIFT": "Instrument calibration shift (step)",
    "SENSOR_STUCK": "Instrument reading frozen", "SENSOR_NOISE_SPIKES": "Instrument noise / spikes",
    "COMMUNICATION_LOSS": "Communication loss", "UNKNOWN": "Unclassified pattern",
}

OUTCOMES = ("Concern observed", "Nothing unusual found", "Unable to verify", "Passed to maintenance")
STATUS_LABEL = {"new": "New", "acknowledged": "Acknowledged", "check_recorded": "Check recorded"}
MONITORING_CONTINUES = "Check recorded; monitoring continues."


# ------------------------------------------------------------------------------------------ evidence parsing
@dataclass
class Observation:
    kind: str           # deviation | relative | alone | frozen | scatter | absolute | crest | snapshot | gap
    key: str | None     # channel key (b4 short key) if any
    span: str           # exact substring of the B4 evidence text this observation was read from
    sentence: str       # plain-language rendering shown to the operator
    value: float | None = None
    direction: str | None = None     # "higher" / "lower"


_N = r"[a-z][a-z \-]*?"
RX_DEV = re.compile(rf"(?P<name>{_N}) (?P<mag>\d+(?:\.\d+)?) (?P<unit>\S*) ?(?P<dir>above|below) the value expected "
                    rf"for this operating point \(z=[^)]*\)")
RX_REL = re.compile(r"(?P<name>speed vs physics-required speed|head vs pump curve) (?P<pct>[+-]\d+(?:\.\d+)?)% "
                    r"\(z=[^)]*\)")
RX_ALONE = re.compile(rf"^(?P<name>{_N}) deviates alone: (?P<sgn>[+-])(?P<mag>\d+(?:\.\d+)?) (?P<unit>\S*) *from the "
                      rf"context-expected value.*?Pattern: (?P<pat>gradual ramp|step change)\.", re.S)
RX_FROZEN = re.compile(rf"^(?P<name>{_N}) \((?P<col>\w+)\) frozen at (?P<val>-?\d+(?:\.\d+)?) while the pump runs "
                       rf"and coupled channels vary\.")
RX_SCATTER = re.compile(rf"^(?P<name>{_N}) \((?P<col>\w+)\) 1-min scatter (?P<ratio>\d+(?:\.\d+)?)x its normal level; "
                        rf"coupled channels steady\.")
RX_ABS = re.compile(rf"(?P<name>{_N}) (?P<pct>[+-]\d+(?:\.\d+)?)% vs pump/system-curve physics")
RX_CREST = re.compile(r"crest factor (?P<val>\d+(?:\.\d+)?) \(healthy ~3\.2\)")
RX_SNAP_BRG = re.compile(r"Vibration snapshot: bearing outer-race \(BPFO\) envelope peak (?P<val>\d+(?:\.\d+)?)x the "
                         r"noise floor \(healthy < 4\)\.")
RX_SNAP_VEL = re.compile(r"Vibration snapshot: 1x (?P<v1>\d+(?:\.\d+)?) mm/s, 2x (?P<v2>\d+(?:\.\d+)?) mm/s "
                         r"\((?P<pat>[^)]*)\)\.")
RX_GAP = re.compile(r"No telemetry rows for (?P<min>\d+) min")
RX_CONTEXT = re.compile(r"Context: demand (?P<dem>\d+) %, supply water (?P<tsup>-?\d+(?:\.\d+)?) degC\.")


def approx(v: float, unit: str = "") -> str:
    """Friendly rounding used in operator sentences (tests check it stays within rounding of the source)."""
    a = abs(v)
    s = f"{a:.0f}" if a >= 100 else f"{a:.1f}" if a >= 0.1 else f"{a:.2f}"
    u = chn.display_unit(unit)
    return f"{s} {u}".strip() if u not in ("%",) else f"{s} %"


def _key(name: str) -> str | None:
    return chn.KEY_BY_B4_NAME.get(name.strip())


def parse_evidence(text: str) -> list[Observation]:
    """Observations stated in a B4 evidence string, in the order B4 listed them (strongest first)."""
    text = str(text or "")
    obs: list[Observation] = []
    m = RX_ALONE.search(text)
    if m:
        k = _key(m["name"])
        up = m["sgn"] == "+"
        how = "gradually" if m["pat"] == "gradual ramp" else "suddenly"
        obs.append(Observation("alone", k, m.group(0),
                               f"The {chn.lower_name(k)} reading has {'risen' if up else 'fallen'} {how} to about "
                               f"{approx(float(m['mag']), m['unit'])} {'above' if up else 'below'} what is expected "
                               f"for the current operating conditions.", float(m["mag"]),
                               "higher" if up else "lower"))
        obs.append(Observation("alone", k, "the physically coupled channels do not show the change a real machine "
                                           "fault would cause",
                               "Readings that would normally change with it if the machine itself had changed have "
                               "not shown that change."))
        return obs
    m = RX_FROZEN.search(text)
    if m:
        k = _key(m["name"])
        obs.append(Observation("frozen", k, m.group(0),
                               f"The {chn.lower_name(k)} reading has stayed fixed at {m['val']} "
                               f"{chn.unit(k)} while the pump runs and related readings vary.".replace("  ", " "),
                               float(m["val"])))
        return obs
    m = RX_SCATTER.search(text)
    if m:
        k = _key(m["name"])
        obs.append(Observation("scatter", k, m.group(0),
                               f"The {chn.lower_name(k)} reading is about {m['ratio']} times as noisy as normal, "
                               f"while related readings are steady.", float(m["ratio"])))
        return obs
    m = RX_GAP.search(text)
    if m:
        obs.append(Observation("gap", None, m.group(0),
                               f"No data were received from the pump for {m['min']} minutes.", float(m["min"])))
        return obs
    for m in RX_ABS.finditer(text):
        k = _key(m["name"])
        if k is None:
            continue
        p = float(m["pct"])
        obs.append(Observation("absolute", k, m.group(0),
                               f"{chn.name(k)} is about {abs(p):.1f} % {'above' if p > 0 else 'below'} the value "
                               f"expected from the pump's rated performance at this duty.", p,
                               "higher" if p > 0 else "lower"))
    if obs and "deviation present since the start of monitoring" in text:
        obs.append(Observation("absolute", None, "deviation present since the start of monitoring",
                               "The difference has been present since monitoring started."))
    body = text.split(". Context:")[0].split(". Vibration snapshot")[0].rstrip(".")
    for part in body.split("; "):
        d = RX_DEV.fullmatch(part.strip())
        if d:
            k = _key(d["name"])
            if k is None:
                continue
            higher = d["dir"] == "above"
            if k == "c":
                s = f"Vibration crest factor (a measure of impacts in the vibration signal) is higher than " \
                    f"expected for the current operating conditions." if higher else \
                    "Vibration crest factor is lower than expected for the current operating conditions."
            else:
                s = f"{chn.name(k)} is about {approx(float(d['mag']), d['unit'])} " \
                    f"{'higher' if higher else 'lower'} than expected for the current operating conditions."
            obs.append(Observation("deviation", k, d.group(0), s, float(d["mag"]), "higher" if higher else "lower"))
            continue
        r = RX_REL.fullmatch(part.strip())
        if r:
            p = float(r["pct"])
            if r["name"] == "head vs pump curve":
                s = f"The pressure the pump develops is about {abs(p):.1f} % {'above' if p > 0 else 'below'} " \
                    f"what its pump curve predicts at the measured flow."
                k = "hr"
            else:
                s = f"Motor speed is about {abs(p):.1f} % {'above' if p > 0 else 'below'} the speed needed for " \
                    f"the current cooling demand."
                k = "rx"
            obs.append(Observation("relative", k, r.group(0), s, p, "higher" if p > 0 else "lower"))
            continue
        c = RX_CREST.fullmatch(part.strip())
        if c:
            obs.append(Observation("crest", "c", c.group(0),
                                   f"The vibration crest factor is {c['val']} (about 3.2 is typical for a healthy "
                                   f"pump).", float(c["val"])))
    m = RX_SNAP_BRG.search(text)
    if m:
        obs.append(Observation("snapshot", "a", m.group(0),
                               f"A detailed vibration snapshot shows a bearing-defect frequency {m['val']} times "
                               f"above the noise floor (below 4 is typical when healthy).", float(m["val"])))
    m = RX_SNAP_VEL.search(text)
    if m:
        pat = m["pat"]
        if pat.startswith("2x dominant"):
            s = "A detailed vibration snapshot shows vibration at twice the running speed dominating, a pattern " \
                "typical of misalignment."
        elif pat.startswith("1x dominant"):
            s = "A detailed vibration snapshot shows vibration at the running speed dominating, a pattern typical " \
                "of imbalance."
        else:
            s = "A detailed vibration snapshot shows no clear pattern."
        obs.append(Observation("snapshot", "v", m.group(0), s))
    return obs


def parse_context(text: str) -> dict:
    m = RX_CONTEXT.search(str(text or ""))
    return {"demand_pct": float(m["dem"]), "supply_temp_c": float(m["tsup"])} if m else {}


# ------------------------------------------------------------------------------------------ check identity
@dataclass(frozen=True)
class Check:
    key: str            # grouping key within a pump
    title: str
    meaning: str
    channel: str | None


def sensor_channel(alert: dict) -> str | None:
    ev = str(alert.get("evidence", ""))
    for rx in (RX_ALONE, RX_FROZEN, RX_SCATTER):
        m = rx.search(ev)
        if m:
            k = _key(m["name"])
            return "n" if k == "rx" else k
    act = str(alert.get("recommended_action", ""))
    for k, nm in chn.B4_NAME.items():
        if f"the {nm} transmitter" in act:
            return k
    return None


def check_for(alert: dict) -> Check:
    ft = str(alert.get("fault_type", "UNKNOWN"))
    if ft in SENSOR_CHECKS:
        fam, title, meaning = SENSOR_CHECKS[ft]
        k = sensor_channel(alert)
        c = chn.compound(k) if k else "instrument"
        return Check(f"{fam}:{k or 'unknown'}", title.format(c=c), meaning, k)
    key, title, meaning = MACHINE_CHECKS.get(ft, MACHINE_CHECKS["UNKNOWN"])
    return Check(key, title, meaning, None)


def explain(alert: dict, max_obs: int = 3) -> tuple[str, list[Observation]]:
    """'Why this check?' text and the observations it is built from (all taken from the alert's evidence)."""
    obs = parse_evidence(alert.get("evidence", ""))
    chk = check_for(alert)
    used = obs[:max_obs] if obs and obs[0].kind != "alone" else obs[:2]
    parts = [o.sentence for o in used]
    if not parts:
        parts = ["The detector found a sustained difference from the expected readings."]
    parts.append(chk.meaning)
    return " ".join(parts), used


# ------------------------------------------------------------------------------------------ cards
@dataclass
class Occurrence:
    source_ref: str
    time: pd.Timestamp
    fault_type: str
    evidence: str
    recommended_action: str
    why: str
    detector: str | None = None


@dataclass
class Card:
    card_id: str
    asset_id: str
    location: str
    check: Check
    occurrences: list[Occurrence]
    status: str = "new"
    last_action: dict | None = None
    actions: list[dict] = field(default_factory=list)
    updated_after_action: bool = False
    existing: bool = False

    @property
    def title(self) -> str:
        return self.check.title

    @property
    def first(self) -> Occurrence:
        return self.occurrences[0]

    @property
    def latest(self) -> Occurrence:
        return self.occurrences[-1]

    @property
    def why(self) -> str:
        return self.latest.why

    @property
    def next_step(self) -> str:
        """B4's recommended action, verbatim (keeps its conditional wording; nothing added)."""
        return self.latest.recommended_action

    @property
    def suspected_faults(self) -> list[str]:
        seen = []
        for o in self.occurrences:
            p = FAULT_PLAIN.get(o.fault_type, o.fault_type)
            if p not in seen:
                seen.append(p)
        return seen

    @property
    def status_label(self) -> str:
        return STATUS_LABEL[self.status]

    def status_note(self) -> str:
        if self.status == "check_recorded":
            a = self.last_action or {}
            note = f"Recorded at {fmt(a.get('replay_time'))}: {a.get('outcome', '')}."
            if a.get("outcome") == "Passed to maintenance":
                note += " (Recorded in this demo only; no work order was created.)"
            return f"{note} {MONITORING_CONTINUES}"
        if self.status == "acknowledged":
            return f"Acknowledged at {fmt((self.last_action or {}).get('replay_time'))}. Monitoring continues."
        if self.updated_after_action:
            return f"Raised again at {fmt(self.latest.time)} after your last action " \
                   f"({fmt((self.last_action or {}).get('replay_time'))})."
        return ""


def card_id(asset_id: str, check: Check) -> str:
    return f"{asset_id}|{check.key}"


def build_cards(alerts: pd.DataFrame, asset: dict, actions: list[dict] | None = None,
                scene_start=None) -> list[Card]:
    """Group visible alerts by (pump, recommended check). Every alert is kept as an occurrence.

    `actions` are the operator interactions visible at the replay clock (see interactions.visible_actions).
    """
    actions = actions or []
    location = " · ".join(x for x in (asset.get("system"), asset.get("site")) if x)
    cards: dict[str, Card] = {}
    if alerts is None or not len(alerts):
        return []
    for _, a in alerts.sort_values(["time", "source_ref"]).iterrows():
        ad = a.to_dict()
        chk = check_for(ad)
        cid = card_id(ad["asset_id"], chk)
        why, _ = explain(ad)
        occ = Occurrence(ad["source_ref"], ts(ad["alert_time_utc"]), ad["fault_type"], ad["evidence"],
                         ad["recommended_action"], why, ad.get("detector"))
        if cid not in cards:
            cards[cid] = Card(cid, ad["asset_id"], location, chk, [])
        cards[cid].occurrences.append(occ)
    start = ts(scene_start) if scene_start is not None else None
    for c in cards.values():
        c.existing = start is not None and c.first.time <= start
        mine = sorted([x for x in actions if x.get("card_id") == c.card_id], key=lambda x: ts(x["replay_time"]))
        c.actions = mine
        if not mine:
            c.status = "new"
            continue
        last = mine[-1]
        c.last_action = last
        if c.latest.time > ts(last["replay_time"]):
            # new detector evidence after the operator's last action: the task is open again
            c.status, c.updated_after_action = "new", True
        else:
            c.status = "check_recorded" if last["action"] == "record_check" else "acknowledged"
    order = {"new": 0, "acknowledged": 1, "check_recorded": 2}
    return sorted(cards.values(), key=lambda c: (order[c.status], -c.latest.time.value))


def split_cards(cards: list[Card], scene_start) -> tuple[list[Card], list[Card]]:
    """(current, existing). Items first raised before the scene start go to the secondary "Existing items" area
    unless they were raised again during the scene. Nothing is dropped."""
    start = ts(scene_start)
    cur = [c for c in cards if not c.existing or any(o.time > start for o in c.occurrences)]
    ex = [c for c in cards if c not in cur]
    return cur, ex
