"""Learned fault signatures (data-driven replacement for B4's physics signatures; no generator values).

A signature is the pattern of channel deviations that a fault leaves: the z-score vector at the alert minute. Three ways to learn it from
labelled TRAIN events only:
  cosine  - class prototypes (mean direction of the scaled vectors); similarity by cosine; magnitude is ignored, so a mild and a severe
            clog look alike;
  knn     - the k nearest scaled training vectors vote;
  logreg  - multinomial logistic regression on the scaled vectors.
Scaling is signed log: s = sign(z) * log(1 + |z|), so a z of 30 does not drown out a z of 5. A channel the pump lacks counts as 0.
All three return a probability per fault type like the E02 model, so they plug into the same classifier hook.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

CLIP = 30.0


def _scale(z: dict[str, float], channels: list[str]) -> np.ndarray:
    x = np.array([np.clip(z.get(c, 0.0), -CLIP, CLIP) if np.isfinite(z.get(c, 0.0)) else 0.0 for c in channels])
    return np.sign(x) * np.log1p(np.abs(x))


class LearnedSignatures:
    def __init__(self, channels: list[str], method: str = "logreg", k: int = 15, C: float = 1.0, temperature: float = 0.15, seed: int = 0):
        if method not in ("cosine", "knn", "logreg"):
            raise ValueError("method must be cosine, knn or logreg")
        self.channels, self.method, self.k, self.C, self.temperature, self.seed = list(channels), method, k, C, temperature, seed
        self.classes_: list[str] = []

    def fit(self, rows: list[dict[str, float]], labels: list[str]) -> "LearnedSignatures":
        X = np.array([_scale(r, self.channels) for r in rows])
        y = np.array(labels)
        self.classes_ = sorted(set(labels))
        if self.method == "cosine":
            P = []
            for c in self.classes_:
                v = X[y == c].mean(axis=0)
                P.append(v / max(np.linalg.norm(v), 1e-9))
            self.proto = np.array(P)
        elif self.method == "knn":
            self.model = KNeighborsClassifier(n_neighbors=min(self.k, len(X)), weights="distance").fit(X, y)
        else:
            self.model = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000, random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, z: dict[str, float]) -> dict[str, float]:
        x = _scale(z, self.channels)
        if not np.any(x):
            return {}
        if self.method == "cosine":
            sim = self.proto @ (x / np.linalg.norm(x))
            e = np.exp((sim - sim.max()) / self.temperature)
            p = e / e.sum()
            return dict(zip(self.classes_, map(float, p)))
        p = self.model.predict_proba(x.reshape(1, -1))[0]
        return {c: float(v) for c, v in zip(self.model.classes_, p)}
