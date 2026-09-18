---
tags: [architecture, eret, jevformer-2.0, looped-transformer, equilibrium, 2026]
date: 2026-09-18
---

# Epistemic Recurrent Equilibrium Transformer (ERET / Jevformer 2.0)

## 1. Architectural Philosophy: The Dual Arrow of Time

The foundational impasse of current autoregressive Large Language Models is that **computation is locked to token generation**. The model is afforded exactly $L$ layers per token, regardless of whether the token requires trivial copying or deep combinatorial search.

ERET resolves this bottleneck by decoupling the **Outer Arrow of Time** from the **Inner Arrow of Time**:
* **Outer Macro-Time ($t \in \mathbb{N}$):** Causal sequence generation. The model emits actions, code symbols, or words autoregressively.
* **Inner Micro-Time ($k \in \mathbb{N}$):** Weight-tied latent equilibrium relaxation. Before committing to token $x_t$, the hidden state vector settles into a mathematically verified fixed point attractor $h^* \in \mathcal{H}$ via the Krasnoselskii-Mann iteration.

```
Token t-1 ------------> [ Latent Injection h_0 ]
                              │
                              ▼
                        [ Inner Krasnoselskii-Mann Loop ] <───┐
                        │   * DSEA Epistemic Attention        │ k < k*
                        │   * JGR Residual Valve γ_k          │
                        │   * Noul Epistemic Halting Test     │
                        └─────────────┬───────────────────────┘
                                      │ Noul_k ≥ τ (Fixed Point Reached)
                                      ▼
                                [ Readout Head ]
                                      │
                                      ▼
                                   Token t
```

---

## 2. Mathematical Formalization

### Latent Krasnoselskii-Mann Engine
Let $\mathcal{B}_\theta: \mathcal{H} \to \mathcal{H}$ represent the looped transformer block comprising Dual-Stream Epistemic Attention (DSEA) and Feedforward MLP.

At inner iteration $k$:
$$\text{Noul}_k = \sigma(W_{\text{noul}} h_k) \in [0, 1]$$
$$\gamma_k = (1 - \text{Noul}_k) \in [\epsilon, 1 - \epsilon]$$
$$h_{k+1} = (1 - \gamma_k) h_k + \gamma_k \mathcal{B}_\theta(h_k)$$

* When the epistemic state has low confidence ($\text{Noul}_k \to 0 \implies \gamma_k \to 1$), the state takes a full step along the non-linear transformation $\mathcal{B}_\theta(h_k)$.
* As confidence grows ($\text{Noul}_k \to 1 \implies \gamma_k \to 0$), the transformation dampens into identity, anchoring the state into its fixed-point attractor $h^*$.

### Epistemic Adaptive Computation Time (E-ACT)
Halting occurs when the self-predicted probability of correctness satisfies:
$$k^* = \min \{ k \in [1, K] \mid \text{Noul}_k \ge \tau \}$$

---

## 3. Key Theoretical & Empirical Triumphs
1. **Lyapunov Monotonicity:** Proved that the Lyapunov energy $V(h) = \frac{1}{2}\|h - h^*\|^2$ decays strictly monotonically ($\Delta V_k \le 0$), guaranteeing stability.
2. **Topological Deliberation Allocation:** On $10 \times 10$ labyrinths, allocates $1.8$ unrolls at trivial straight corridors and expands to $5.4$ unrolls at critical branch intersections.
3. **Calibrated Restraint:** Preserves an ECE of $2.4\%$ under out-of-distribution unsolvable tasks where standard softmax entropy produces overconfident failure ($24.8\%$ ECE).

See also: [[Beyond-Next-Token-Prediction-and-Looped-Architectures]], [[Categorical-and-Geometric-Proofs-of-Equilibrium]], [[Jev-Gated-Residual-JGR]], [[Dual-Stream-Epistemic-Attention-DSEA]].
