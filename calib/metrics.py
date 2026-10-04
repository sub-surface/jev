"""Calibration and selective-prediction metrics (numpy, evaluation only).

Conventions: ``probs`` is (N, K) with zeros in padded columns; ``y`` is (N,)
int labels. Binary-confidence functions take ``conf`` in [0, 1] and an
``outcome`` in {0, 1} (e.g. "top-1 prediction was correct").

Binned ECE is reported because everyone reports it, but it is biased and
bin-dependent (Kumar et al. 2019; Błasiok & Nakkiran 2023). Prefer the
debiased L2 estimate or smooth ECE, always with a bootstrap interval, and
never compare ECEs computed on a few hundred examples without one.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ── basic proper scores ──────────────────────────────────────────────────────

def top_label(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-label confidence and correctness indicator."""
    pred = probs.argmax(1)
    return probs.max(1), (pred == y).astype(float)


def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], eps, None)).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


# ── calibration error ────────────────────────────────────────────────────────

def _bins(conf: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "width":
        edges = np.linspace(0, 1, n_bins + 1)
    elif scheme == "mass":
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(scheme)
    return np.clip(np.searchsorted(edges[1:-1], conf, side="right"), 0, n_bins - 1)


def ece(conf: np.ndarray, outcome: np.ndarray, n_bins: int = 15, scheme: str = "mass") -> float:
    """Binned L1 calibration error. ``scheme='mass'`` (equal-count bins) is less noisy."""
    b = _bins(conf, n_bins, scheme)
    err = 0.0
    for i in np.unique(b):
        m = b == i
        err += m.mean() * abs(outcome[m].mean() - conf[m].mean())
    return float(err)


def ece_l2_debiased(conf: np.ndarray, outcome: np.ndarray, n_bins: int = 15) -> float:
    """Debiased binned L2 calibration error (Kumar, Liang & Ma, NeurIPS 2019).

    The plug-in estimate of (acc_b - conf_b)^2 is inflated by the sampling
    variance of acc_b; subtract its unbiased estimate acc_b(1-acc_b)/(n_b-1).
    Returns sqrt(max(0, estimate)).
    """
    b = _bins(conf, n_bins, "mass")
    tot = 0.0
    for i in np.unique(b):
        m = b == i
        n = m.sum()
        if n < 2:
            continue
        a = outcome[m].mean()
        tot += m.mean() * ((a - conf[m].mean()) ** 2 - a * (1 - a) / (n - 1))
    return float(np.sqrt(max(tot, 0.0)))


def smooth_ece(conf: np.ndarray, outcome: np.ndarray, grid: int = 512) -> float:
    """Smooth ECE (Błasiok & Nakkiran, 2023): kernel-smoothed residuals with a
    reflected Gaussian kernel, bandwidth chosen as the fixed point smECE_s = s.

    Consistent (no bins to tune) and continuous in the predictor. This is a
    compact re-implementation; use the authors' ``relplot`` package for papers.
    """
    r = outcome - conf
    t = (np.arange(grid) + 0.5) / grid
    # pre-aggregate residuals on a fine grid (error <= 1/(2*grid) in location), then
    # reflect about 0 and 1 so kernel mass near the edges stays inside [0, 1]
    cell = np.clip((conf * grid).astype(int), 0, grid - 1)
    rsum = np.bincount(cell, weights=r, minlength=grid)
    pts = np.concatenate([t, -t, 2 - t])
    res = np.concatenate([rsum, rsum, rsum])

    def at(sigma: float) -> float:
        # smECE_s = ∫ |(1/n) Σ_i K_s(t, f_i) r_i| dt with K_s a normalised Gaussian density
        k = np.exp(-0.5 * ((t[:, None] - pts[None, :]) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return float(np.abs(k @ res).mean() / len(conf))

    lo, hi = 1e-4, 1.0
    for _ in range(30):  # smECE_s is decreasing in s, so bisection finds s = smECE_s
        mid = (lo + hi) / 2
        if at(mid) > mid:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def murphy_decomposition(conf: np.ndarray, outcome: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Binary Brier = reliability - resolution + uncertainty (binned approximation).

    Calibration alone is cheap (predict the base rate: reliability 0, resolution 0).
    Value comes from resolution — the "sharpness" that calibration must not trade away.
    """
    b = _bins(conf, n_bins, "mass")
    base = outcome.mean()
    rel = res = 0.0
    for i in np.unique(b):
        m = b == i
        w, a, c = m.mean(), outcome[m].mean(), conf[m].mean()
        rel += w * (c - a) ** 2
        res += w * (a - base) ** 2
    return {"brier": float(((conf - outcome) ** 2).mean()), "reliability": rel,
            "resolution": res, "uncertainty": float(base * (1 - base))}


def corp_decomposition(p: np.ndarray, y: np.ndarray, score: str = "log") -> dict[str, float]:
    """CORP decomposition of a binary proper score (Dimitriadis, Gneiting & Jordan 2021;
    Arnold, Walz, Ziegel & Gneiting 2023, arXiv:2311.14122).

        mean S(p) = MCB - DSC + UNC        (exact identity, all terms >= 0)

    q = isotonic (PAV) recalibration of p, with ties pooled; r = base rate.
    MCB = S(p) - S(q)  miscalibration: what a monotone re-labelling of p would save
    DSC = S(r) - S(q)  discrimination / resolution: what p extracted beyond the base rate
    UNC = S(r)         uncertainty: the base-rate code length

    With score='log' all three are in bits/symbol, so DSC is the information the predictor
    extracted and MCB is the cost of dishonest probabilities.
    """
    p = np.asarray(p, float); y = np.asarray(y, float)
    vals, inv = np.unique(p, return_inverse=True)            # pool ties first
    w = np.bincount(inv).astype(float)
    m = np.bincount(inv, weights=y) / w
    blocks: list[list[float]] = []                            # weighted PAV: [mean, weight, n_vals]
    for mi, wi in zip(m, w):
        blocks.append([mi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop(); a = blocks[-1]
            a[0] = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]); a[1] += b[1]; a[2] += b[2]
    qv = np.concatenate([[b[0]] * b[2] for b in blocks])
    q, r = qv[inv], np.full_like(p, y.mean())

    def S(f):
        if score == "log":
            f = np.clip(f, 1e-15, 1 - 1e-15)
            return float(-(y * np.log2(f) + (1 - y) * np.log2(1 - f)).mean())
        return float(((f - y) ** 2).mean())

    sp, sq, sr = S(p), S(q), S(r)
    return {"score": sp, "MCB": sp - sq, "DSC": sr - sq, "UNC": sr}


# ── selective prediction ─────────────────────────────────────────────────────

def risk_coverage(conf: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective accuracy when answering only the top-c most confident."""
    o = outcome[np.argsort(-conf, kind="stable")]
    n = np.arange(1, len(o) + 1)
    return n / len(o), np.cumsum(o) / n


def aurc(conf: np.ndarray, outcome: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better). Measures ranking, not calibration."""
    cov, acc = risk_coverage(conf, outcome)
    return float(np.mean(1 - acc))


# ── uncertainty about the metrics themselves ─────────────────────────────────

def bootstrap_ci(
    fn: Callable[..., float], *arrays: np.ndarray, n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval for any metric of paired arrays."""
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    stats = [fn(*(a[idx] for a in arrays)) for idx in (rng.integers(0, N, N) for _ in range(n))]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return fn(*arrays), float(lo), float(hi)


def report(probs: np.ndarray, y: np.ndarray, ci: bool = True) -> dict[str, object]:
    """One-call summary used by experiments."""
    conf, ok = top_label(probs, y)
    out: dict[str, object] = {
        "n": int(len(y)), "acc": float(ok.mean()), "nll": nll(probs, y), "brier": brier(probs, y),
        "ece_mass15": ece(conf, ok), "ece_l2_debiased": ece_l2_debiased(conf, ok),
        "smece": smooth_ece(conf, ok), "aurc": aurc(conf, ok),
    }
    if ci:
        out["acc_ci"] = bootstrap_ci(lambda o: float(o.mean()), ok)[1:]
        out["smece_ci"] = bootstrap_ci(smooth_ece, conf, ok, n=200)[1:]
    return out
