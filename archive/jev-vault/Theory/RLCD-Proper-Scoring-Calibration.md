---
tags: [theory, rlcd, brier-score, calibration, math]
created: 2026-09-18
---

# RLCD & Proper Scoring Calibration

## 1. The Geometry of Honest Self-Models
Reinforcement Learning from Human Feedback (RLHF) optimizes for human preference, notoriously inducing **sycophancy and overconfidence**. When an LLM is uncertain, it is rewarded for generating articulate justifications rather than expressing uncertainty.

**Reinforcement Learning from Contrastive Distillation (RLCD)** (Yang et al., 2023) replaces uncalibrated human reward signals with strictly convex proper scoring rules.

---

## 2. Proper Scoring Rules & Brier Loss
Let $y \in \{0, 1\}$ be the true ground-truth solvability of an instance under System 1, and let $\hat{p} = \text{Noul}(h) \in [0, 1]$ be the epistemic confidence emitted by the Jev head.

A scoring rule $S(\hat{p}, y)$ is **strictly proper** if and only if the expected score is uniquely minimized when the reported confidence equals the true conditional probability:
$$\mathbb{E}_{y \sim P}[S(\hat{p}, y)] \ge \mathbb{E}_{y \sim P}[S(P(y=1), y)]$$

In Jevformer, we utilize the **Brier proper scoring loss**:
$$\mathcal{L}_{\text{Brier}}(\hat{p}, y) = (\hat{p} - y)^2$$

### The RLCD Contrastive Pair:
To prevent mode collapse and sharpen the epistemic decision boundary, we construct contrastive prompt pairs $(p^+, p^-)$:
* $p^+$: High-solvability reflexive instance ($y = 1$).
* $p^-$: High-complexity multi-hop instance ($y = 0$).

The contrastive objective enforces:
$$\mathcal{L}_{\text{contrast}} = \mathcal{L}_{\text{Brier}}(\text{Noul}(p^+), 1) + \mathcal{L}_{\text{Brier}}(\text{Noul}(p^-), 0) + \lambda \max(0, m - (\text{Noul}(p^+) - \text{Noul}(p^-)))$$

As demonstrated in Phase 2 on Modal Cloud (**Figure 6**), this drove the contrast margin to **0.975**, with $\text{Noul}(p^+) = 1.000$ and $\text{Noul}(p^-) = 0.025$.

See also: [[Epiplexity-and-Bounded-Information]], [[Jev-Gated-Residual-JGR]].
