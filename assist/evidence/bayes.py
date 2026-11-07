"""E02: probability with Gaussian naive Bayes.

Features are the signed z-scores of the channels at the alert minute. A missing channel is left out of the
likelihood (marginalised), so one model works for pumps with different sensors. Trained on TRAIN pumps only.
Priors are uniform: there are too few labelled events to learn class frequencies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLIP = 30.0


class GaussianNB:
    def __init__(self, channels: list[str], var_floor: float = 4.0):
        self.channels, self.var_floor = list(channels), var_floor
        self.classes: list[str] = []
        self.mean = np.zeros((0, len(channels)))
        self.var = np.ones((0, len(channels)))

    def _vec(self, z: dict[str, float]) -> np.ndarray:
        return np.array([np.clip(z[c], -CLIP, CLIP) if c in z and np.isfinite(z[c]) else np.nan for c in self.channels])

    def fit(self, rows: list[dict[str, float]], labels: list[str]) -> "GaussianNB":
        X = np.array([self._vec(r) for r in rows])
        y = np.array(labels)
        self.classes = sorted(set(labels))
        means, varis = [], []
        gm = np.nanmean(X, axis=0)
        for c in self.classes:
            Xc = X[y == c]
            with np.errstate(all="ignore"):
                m = np.where(np.isnan(Xc).all(axis=0), gm, np.nanmean(Xc, axis=0))
                v = np.where(np.isnan(Xc).all(axis=0), self.var_floor, np.nanvar(Xc, axis=0))
            means.append(np.nan_to_num(m))
            varis.append(np.maximum(np.nan_to_num(v, nan=self.var_floor), self.var_floor))
        self.mean, self.var = np.array(means), np.array(varis)
        return self

    def predict_proba(self, z: dict[str, float]) -> dict[str, float]:
        x = self._vec(z)
        ok = ~np.isnan(x)
        if not ok.any() or not self.classes:
            return {}
        ll = -0.5 * (np.log(2 * np.pi * self.var[:, ok]) + (x[ok] - self.mean[:, ok]) ** 2 / self.var[:, ok]).sum(axis=1)
        p = np.exp(ll - ll.max())
        p /= p.sum()
        return dict(zip(self.classes, map(float, p)))

    def predict_proba_tempered(self, z: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
        """Naive Bayes is overconfident (it multiplies dependent channels). Temperature T > 1 flattens it."""
        p = self.predict_proba(z)
        if not p or temperature == 1.0:
            return p
        lp = {c: np.log(max(v, 1e-300)) / temperature for c, v in p.items()}
        m = max(lp.values())
        e = {c: float(np.exp(v - m)) for c, v in lp.items()}
        tot = sum(e.values())
        return {c: v / tot for c, v in e.items()}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"channels": self.channels, "classes": self.classes, "var_floor": self.var_floor,
                                          "mean": self.mean.tolist(), "var": self.var.tolist()}, indent=1))
