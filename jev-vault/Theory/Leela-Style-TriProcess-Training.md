---
type: theory
project: Leela-Style Tri-Process Training
created: 2026-09-18
tags: [leela, lc0, alphazero, triprocess, modal, self-play, value-policy]
---

# 🧠 Leela-Style Tri-Process Training on Public Master Data & Self-Play

This document outlines the theoretical synthesis of **Leela Chess Zero (Lc0) / AlphaZero dual-head training** with the **Jevformer Tri-Process Neural Engine (NNUE + Jev + Epistemic Search)**.

---

## 🏛️ 1. Theoretical Motivation: Beyond Pure Classical Priors

While classical piece-square tables (PeSTO) and MVV-LVA move ordering give Jev an immediate ~1900 Elo baseline, pure heuristic search suffers from three blindspots against skilled human bullet players:

1. **Strategic Piece Sacrifices & Exchanges:** Pure material-weighted heuristics often accept bad exchanges (e.g. trading active pieces and conceding open files/passed pawns).
2. **Passed Pawn Acceleration:** In bullet endgames, passed pawns require non-linear spatial evaluation rather than static piece counts.
3. **Over-Thinking Quiet Plies:** Without a trained policy prior $\mathbf{P}(s)$, the engine is forced to evaluate all legal moves with tree search even when 95% of candidates are obviously passive.

By training on **large-scale master games (Lichess public database)** and **deep self-play evaluations**, the network learns deep strategic heuristics without sacrificing the sub-millisecond execution speed of the CReLU accumulator.

---

## 📐 2. The Tri-Process Loss Formulation

The network optimizes three concurrent objectives with mechanistic sparsity regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{val}}(V(s), z) + \lambda_{\text{pol}} \mathcal{L}_{\text{pol}}(P(s), \pi) + \lambda_{\text{noul}} \mathcal{L}_{\text{noul}}(\text{Noul}(s), y) + \lambda_{\text{sparse}} \|\text{CReLU}(a)\|_1$$

### Head 1: Value Regression ($\mathcal{L}_{\text{val}}$)
Maps position $s$ to expected outcome $z \in [-1, +1]$:
$$\mathcal{L}_{\text{val}} = \frac{1}{B} \sum_{i=1}^B (V(s_i) - \tanh(cp_i / 400.0))^2$$

### Head 2: Policy Guidance ($\mathcal{L}_{\text{pol}}$)
Cross-entropy over candidate moves to guide search ordering toward master moves $\pi^*$:
$$\mathcal{L}_{\text{pol}} = -\sum_{a \in \mathcal{A}(s)} \pi^*(a|s) \log P(a|s)$$

### Head 3: Calibrated Epistemic $\text{Noul}$ ($\mathcal{L}_{\text{noul}}$)
Trained via Brier proper scoring to distinguish quiet positional states ($y=1$) from sharp tactical crises ($y=0$):
$$\mathcal{L}_{\text{noul}} = \frac{1}{B} \sum_{i=1}^B \left( \text{Noul}(s_i) - y_i \right)^2$$

### Mechanistic Regularization ($\mathcal{L}_{\text{sparse}}$)
Prevents dense latent corruption and maintains a 35%–50% zero-activation rate across the 128-neuron CReLU layer:
$$\mathcal{L}_{\text{sparse}} = 0.01 \cdot \frac{1}{128} \sum_{j=1}^{128} \max(0, \min(1, a_j))$$

---

## ⚡ 3. Real-Time Inference Synergy: The Epistemic Gating Loop

During live bullet play on Lichess:

```
                      Input Board Tensor (13 x 8 x 8)
                                     │
                                     ▼
                      Jevformer CReLU Accumulator
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
           Value Head V(s)     Policy P(s)         Noul Head N(s)
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     │
                         Is Position Quiet?
                         (Noul >= 0.70 & No Check)
                                  /     \
                               YES       NO
                              /           \
                 Play 1-Ply Reflex         Dynamic Negamax Search
                 (<15ms latency)           (TT + Alpha-Beta Pruning)
                 Preserves Clock!          (50ms - 250ms budget)
```

This guarantees:
* **Opening / Quiet Positions:** $\sim 0\text{ms}$ (book) or $<15\text{ms}$ (reflex).
* **Tactical Crises:** Targeted 100ms–250ms search with transposition table cache.
* **Result:** Zero clock forfeits, hyper-responsive bullet play, and master-level positional accuracy.
