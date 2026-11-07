"""Save the operator decision (approve / reject / not sure) in results/decisions.csv. The app never talks to a pump."""
import csv
from datetime import datetime, timezone
from pathlib import Path

DECISIONS_CSV = Path(__file__).resolve().parents[1] / "results" / "decisions.csv"
CHOICES = ("approved", "rejected", "not sure")
FIELDS = ["decided_utc", "asset_id", "alert_time_utc", "fault_type", "decision"]


def save_decision(alert, decision: str, path: Path = DECISIONS_CSV) -> None:
    if decision not in CHOICES:
        raise ValueError(f"decision must be one of {CHOICES}")
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({"decided_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "asset_id": alert.asset_id,
                    "alert_time_utc": alert.alert_time_utc, "fault_type": alert.fault_type, "decision": decision})
    alert.operator_decision = decision


def load_decisions(path: Path = DECISIONS_CSV) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
