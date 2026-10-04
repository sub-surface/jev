"""Post-hoc recalibrators. Always fit on a split disjoint from train AND test."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax


def _masked(logits: np.ndarray) -> np.ndarray:
    # padded options are encoded as -inf (or NaN); keep them at -inf
    return np.where(np.isfinite(logits), logits, -np.inf)


class TemperatureScaler:
    """Single global temperature fit by NLL (Guo et al., 2017). No clamping, no prior."""

    def __init__(self) -> None:
        self.T = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        z = _masked(logits)

        def nll(log_t: float) -> float:
            lp = log_softmax(z / np.exp(log_t), axis=1)
            return -lp[np.arange(len(y)), y].mean()

        self.T = float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))
        return self

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return np.exp(log_softmax(_masked(logits) / self.T, axis=1))


def pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: isotonic fit of y on x. Returns (sorted x, fitted y)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order].astype(float)
    vals, wts, ends = [], [], []
    for i, v in enumerate(ys):
        vals.append(v); wts.append(1.0); ends.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w; ends[-2] = ends[-1]
            vals.pop(); wts.pop(); ends.pop()
    fit = np.empty_like(ys)
    start = 0
    for v, e in zip(vals, ends):
        fit[start:e + 1] = v
        start = e + 1
    return xs, fit


class IsotonicConfidence:
    """Monotone map from a raw confidence score to P(correct). Good for escalation gates."""

    def fit(self, conf: np.ndarray, outcome: np.ndarray) -> "IsotonicConfidence":
        self.x, self.y = pav(conf, outcome)
        return self

    def __call__(self, conf: np.ndarray) -> np.ndarray:
        return np.interp(conf, self.x, self.y)
