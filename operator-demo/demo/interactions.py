"""Local, single-user interaction log (JSON lines inside operator-demo/state/).

Nothing here leaves the laptop: no work orders, device commands or notifications. The log is append-only during a
session; "Restart" writes a marker that starts the chapter afresh, and "Reset demo" empties the file.

Interactions are stamped with the replay clock. A view at clock T only uses interactions with replay_time <= T, so
seeking backwards never shows an action that had not yet happened in the replay.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .adapter import OUTCOMES
from .paths import INTERACTION_LOG, inside_demo
from .replay import ts


class InteractionStore:
    def __init__(self, path: Path = INTERACTION_LOG):
        path = Path(path)
        if not inside_demo(path) and not _is_test_path(path):
            raise ValueError("interaction log must stay inside operator-demo/")
        self.path = path

    # -------------------------------------------------------------- write
    def _append(self, rec: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"id": uuid.uuid4().hex[:12], "wall_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **rec}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def acknowledge(self, scene_id: str, card_id: str, replay_time, source_refs: list[str]) -> dict:
        return self._append({"scene": scene_id, "card_id": card_id, "action": "acknowledge",
                             "replay_time": ts(replay_time).isoformat(), "source_refs": list(source_refs)})

    def record_check(self, scene_id: str, card_id: str, replay_time, outcome: str, notes: str,
                     source_refs: list[str]) -> dict:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        return self._append({"scene": scene_id, "card_id": card_id, "action": "record_check",
                             "replay_time": ts(replay_time).isoformat(), "outcome": outcome,
                             "notes": (notes or "").strip(), "source_refs": list(source_refs)})

    def restart_scene(self, scene_id: str) -> dict:
        return self._append({"scene": scene_id, "action": "restart_scene"})

    def reset(self) -> None:
        """Clear every recorded interaction (the clearly labelled 'Reset demo')."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    # -------------------------------------------------------------- read
    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue      # a torn final line must not break the demo
        return out

    def visible_actions(self, scene_id: str, clock) -> list[dict]:
        """Card actions for this chapter since its last restart, made at or before the replay clock."""
        recs = [r for r in self.all() if r.get("scene") == scene_id]
        last_restart = max((i for i, r in enumerate(recs) if r["action"] == "restart_scene"), default=-1)
        c = ts(clock)
        return [r for r in recs[last_restart + 1:] if r["action"] in ("acknowledge", "record_check")
                and ts(r["replay_time"]) <= c]


def _is_test_path(path: Path) -> bool:
    # pytest's tmp_path lives outside the repository; allow it explicitly for tests only
    return "pytest-of-" in str(path) or "pytest_" in str(path)
