"""Alert record (worklist section 3) and the submission CSV writer (S01).

All parts of the assistant read and write this one record, so a part can change
without breaking the others. Pure standard library on purpose.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Iterable

EVENT_CLASSES = ("MACHINE", "SENSOR")
CERTAINTY_LEVELS = ("Low", "Medium", "High")
OPERATOR_DECISIONS = (None, "approved", "rejected", "not sure")
UNKNOWN = "UNKNOWN"

# Column order of the scorer's submission file (see submission_template/).
SUBMISSION_COLUMNS = ["asset_id", "alert_time_utc", "event_class", "fault_type",
                      "confidence", "evidence", "recommended_action"]

_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass
class Evidence:
    channel: str
    z: float
    text: str = ""


@dataclass
class AlertRecord:
    asset_id: str
    alert_time_utc: str                      # ISO 8601, UTC, e.g. 2025-09-10T06:12:00Z
    event_class: str                         # MACHINE or SENSOR
    fault_type: str = UNKNOWN                # taxonomy in BENCHMARK_RULES.md, or UNKNOWN
    probability: float = 0.0                 # part 3; written as `confidence`
    certainty: str = "Low"                   # Low, Medium or High
    top_types: list[tuple[str, float]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    channel_trust: dict[str, float] = field(default_factory=dict)   # Q01
    recommended_action: str = ""             # part 4
    what_if: dict[str, Any] | None = None    # part 4, optional
    operator_decision: str | None = None     # part 5

    def validate(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id is empty")
        if not _TS.match(self.alert_time_utc):
            raise ValueError(f"alert_time_utc must look like 2025-09-10T06:12:00Z, got {self.alert_time_utc!r}")
        datetime.strptime(self.alert_time_utc, "%Y-%m-%dT%H:%M:%SZ")
        if self.event_class not in EVENT_CLASSES:
            raise ValueError(f"event_class must be one of {EVENT_CLASSES}, got {self.event_class!r}")
        if not self.fault_type:
            raise ValueError("fault_type is empty; use UNKNOWN")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in 0..1, got {self.probability}")
        if self.certainty not in CERTAINTY_LEVELS:
            raise ValueError(f"certainty must be one of {CERTAINTY_LEVELS}, got {self.certainty!r}")
        if any(not 0.0 <= t <= 1.0 for t in self.channel_trust.values()):
            raise ValueError("channel_trust values must be in 0..1")
        if self.operator_decision not in OPERATOR_DECISIONS:
            raise ValueError(f"operator_decision must be one of {OPERATOR_DECISIONS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def evidence_text(self) -> str:
        parts = [e.text or f"{e.channel} z={e.z:+.1f}" for e in self.evidence]
        return "; ".join(parts)


def to_submission_rows(alerts: Iterable[AlertRecord]) -> list[dict[str, Any]]:
    rows = []
    for a in alerts:
        a.validate()
        rows.append({
            "asset_id": a.asset_id,
            "alert_time_utc": a.alert_time_utc,
            "event_class": a.event_class,
            "fault_type": a.fault_type,
            "confidence": round(a.probability, 4),
            "evidence": a.evidence_text(),
            "recommended_action": a.recommended_action,
        })
    return rows


def write_submission_csv(alerts: Iterable[AlertRecord], path) -> int:
    """Write alert records as the submission CSV. Returns the number of rows."""
    rows = to_submission_rows(alerts)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
