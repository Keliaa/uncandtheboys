"""E04: a small decision tree for the machine fault type (depth 3 or less) that an engineer can read and check.

Trained on the same rows as E02 (z-scores of flagged minutes inside labelled train events). A channel the pump does not have
counts as "no deviation" (0). Output is the leaf's class frequencies, so probabilities are coarse (often 0 or 1).
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

CLIP = 30.0


class FaultTree:
    def __init__(self, channels: list[str], max_depth: int = 3, seed: int = 0):
        if max_depth > 3:
            raise ValueError("the worklist asks for depth 3 or less")
        self.channels = list(channels)
        self.tree = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=seed, min_samples_leaf=5)

    def _x(self, z: dict[str, float]) -> np.ndarray:
        return np.array([np.clip(z.get(c, 0.0), -CLIP, CLIP) if np.isfinite(z.get(c, 0.0)) else 0.0 for c in self.channels])

    def fit(self, rows: list[dict[str, float]], labels: list[str]) -> "FaultTree":
        self.tree.fit(np.array([self._x(r) for r in rows]), labels)
        return self

    def predict_proba(self, z: dict[str, float]) -> dict[str, float]:
        p = self.tree.predict_proba(self._x(z).reshape(1, -1))[0]
        return {c: float(v) for c, v in zip(self.tree.classes_, p)}

    def rules(self) -> str:
        return export_text(self.tree, feature_names=self.channels, show_weights=False)

    def leaves(self) -> list[dict]:
        """Every leaf as {conditions: [(channel, '<=' or '>', threshold)], channels: set, fault_type, n}."""
        t = self.tree.tree_
        out = []

        def walk(node, conds):
            if t.children_left[node] == -1:
                counts = t.value[node][0]
                out.append({"conditions": conds, "channels": {c for c, _, _ in conds},
                            "fault_type": self.tree.classes_[int(np.argmax(counts))], "n": int(t.n_node_samples[node])})
                return
            f = self.channels[t.feature[node]]
            th = float(t.threshold[node])
            walk(t.children_left[node], conds + [(f, "<=", th)])
            walk(t.children_right[node], conds + [(f, ">", th)])
        walk(0, [])
        return out
