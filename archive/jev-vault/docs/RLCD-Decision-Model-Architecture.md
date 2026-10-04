---
title: Calibrated Multi-Task Decision Model (Jev Analogue) Architecture
date: 2026-09-19
status: active
tags: [jev, rlcd, calibration, proper-scoring-rules, tasksource, qwen2.5]
---

# Calibrated Multi-Task Decision Model Architecture

## 1. Executive Summary

Existing open reproductions of TypeSafe AI's Jev (such as Kev, openjev-lm, and jevlike) share a critical architectural omission: **they train with standard cross-entropy loss against one-hot labels and label the model "calibrated."**

As established by Guo et al. (2017) and verified empirically in our PCPR benchmarks, cross-entropy training under finite capacity optimizes purely for accuracy on observed training distributions while driving logit magnitudes towards overconfidence on ambiguous boundary conditions.

This architecture directly implements TypeSafe's actual training methodology: **RLCD (Reinforcement Learning for Calibrated Decisions)**:
1. Candidate choices are demarcated using dedicated option-marker tokens (`[OPT_A]` through `[OPT_Z]`).
2. Representations at marker tokens are pooled and projected into candidate logits $z \in \mathbb{R}^K$.
3. Gaussian exploration noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is injected into logits.
4. The policy outputs a probability distribution $\mathbf{p} = \text{Softmax}(z + \epsilon)$.
5. Reward is assigned via **strictly proper scoring rules** (Logarithmic score, Spherical score, and Ranked Probability Score for ordinal targets).
6. Post-hoc calibration is performed using **per-cardinality-bucket temperature scaling** ($K=2$, $K \in [3,5]$, $K \in [6,10]$, $K \ge 11$).

---

## 2. Decision Primitives & Tasksource Harmonization

Built upon `tasksource` (Sileo, 2023, arXiv:2301.05948) and `tasksource-instruct-v0`:

| Native Tasksource Category | Jev Primitive | Canonical Schema | Scoring Rule |
| :--- | :--- | :--- | :--- |
| **MultipleChoice** | `choice` | Distribution over $K$ candidate options $\mathbf{p} \in \Delta^K$ | Log Score + Spherical Score |
| **Binary Classification / Entailment** | `noul` | Scalar epistemic confidence $p \in [0, 1]$ | Brier / Log Score |
| **Multi-class Classification (>2 classes)** | `choice` | Distribution over $K$ classes | Log Score + Spherical Score |
| **Ordinal / Rating Classification** | `score` | Discrete probability distribution over $K$ ordered bins | Ranked Probability Score (RPS) |
| **Regression (Continuous targets)** | `score` | Discretized CDF over $K$ quantiles | Ranked Probability Score (RPS) |

---

## 3. Strictly Proper Scoring Rules in RLCD

A scoring rule $S(\mathbf{p}, y)$ is **strictly proper** if and only if:
$$\mathbb{E}_{y \sim \mathbf{q}}[S(\mathbf{q}, y)] > \mathbb{E}_{y \sim \mathbf{q}}[S(\mathbf{p}, y)] \quad \forall \mathbf{p} \neq \mathbf{q}$$
where $\mathbf{q}$ is the true data-generating distribution. Under strictly proper scoring, the unique policy that maximizes expected reward is reporting the true Bayesian posterior probability.

### 3.1 Logarithmic Score
$$S_{\text{log}}(\mathbf{p}, y) = \log(p_y)$$
Severely penalizes confident errors ($p_y \to 0 \implies S_{\text{log}} \to -\infty$).

### 3.2 Spherical Score
$$S_{\text{sph}}(\mathbf{p}, y) = \frac{p_y}{\|\mathbf{p}\|_2}$$
Provides bounded gradient stability alongside logarithmic scoring.

### 3.3 Ranked Probability Score (RPS)
$$\text{RPS}(\mathbf{p}, y) = -\frac{1}{K-1} \sum_{k=1}^{K-1} \left( \sum_{j=1}^k p_j - \mathbf{1}[y \le k] \right)^2$$
Penalizes probability mass based on ordinal distance from the true target. Predicting 4 stars when the ground truth is 5 stars incurs a small loss; predicting 1 star incurs a quadratic penalty.

---

## 4. Per-Cardinality Temperature Scaling

Rather than fitting a single global temperature scalar $T$, we partition calibration validation into cardinality buckets:
- **Bucket 2:** Binary / Noul tasks ($K=2$)
- **Bucket 3-5:** Small multiple choice ($K \in [3, 5]$)
- **Bucket 6-10:** Medium choice sets ($K \in [6, 10]$)
- **Bucket 11+:** Large categorical sets ($K \ge 11$)

Each bucket optimizes $T_b$ via L-BFGS to minimize negative log-likelihood on held-out validation logits:
$$\min_{T_b > 0} -\sum_{i \in \text{Bucket } b} \log\left( \frac{\exp(z_{i, y_i} / T_b)}{\sum_j \exp(z_{i, j} / T_b)} \right)$$

---

## 5. Dual-Axis Generalization & Epistemic Gap

To avoid false claims of general calibration, evaluation is strictly split across two orthogonal axes:
1. **In-Task Validation:** Held-out examples from task families seen during training.
2. **Zero-Shot Task Family Generalization:** Entire task families (e.g. all Fact Verification or all Discourse tasks) completely withheld from training.

The **Epistemic Gap**:
$$\Delta_{\text{ECE}} = \text{ECE}_{\text{zero-shot}} - \text{ECE}_{\text{in-task}}$$
quantifies whether the model possesses genuine cross-domain calibration or merely memorized task-specific confidence baselines.
