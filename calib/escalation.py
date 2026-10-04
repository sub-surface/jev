"""The quantity the program actually cares about: compute saved at matched quality.

Setting: a cheap System-1 model answers every query and reports a confidence.
A gate escalates the least-confident fraction q to an expensive System-2
(search, chain-of-thought, a bigger model, a human, a simulator). We measure
accuracy as a function of q, and compare the gate against
  * random escalation (what you get with no confidence signal), and
  * the oracle gate (escalate exactly the queries S1 gets wrong and S2 gets right).

The gap between the confidence curve and the random curve is the value of the
confidence signal; the gap to the oracle is what better calibration/resolution
could still buy.
"""
from __future__ import annotations

import numpy as np


def escalation_curve(
    conf: np.ndarray, s1_ok: np.ndarray, s2_ok: np.ndarray, grid: int = 101
) -> dict[str, np.ndarray]:
    """Accuracy vs escalation rate for confidence-gated, random, and oracle routing."""
    n = len(conf)
    q = np.linspace(0, 1, grid)
    k = np.round(q * n).astype(int)
    order = np.argsort(conf, kind="stable")          # least confident first
    gain = (s2_ok - s1_ok).astype(float)              # +1 if escalation fixes it, -1 if it breaks it
    base = s1_ok.mean()
    gated = base + np.concatenate([[0], np.cumsum(gain[order])])[k] / n
    rand = base + q * gain.mean()
    best = np.sort(gain)[::-1]
    oracle = base + np.concatenate([[0], np.cumsum(best)])[k] / n
    return {"q": q, "gated": gated, "random": rand, "oracle": oracle}


def escalation_needed(curve: dict[str, np.ndarray], target_frac: float = 0.99, key: str = "gated") -> float:
    """Smallest escalation rate whose accuracy reaches ``target_frac`` of always-escalate accuracy.

    1 - this number is the fraction of System-2 calls saved at (near-)matched quality.
    """
    target = target_frac * curve[key][-1]
    hit = np.nonzero(curve[key] >= target - 1e-12)[0]
    return float(curve["q"][hit[0]]) if len(hit) else 1.0


def expected_cost_threshold(c_s2: float, loss_wrong: float = 1.0) -> float:
    """Bayes-optimal gate for a calibrated S1 when S2 is (nearly) always right:
    escalate iff (1 - p) * loss_wrong > c_s2, i.e. p < 1 - c_s2 / loss_wrong.
    Only valid if p is calibrated — which is the whole point.
    """
    return max(0.0, 1.0 - c_s2 / loss_wrong)
