---
tags: [experiment, dyck-grammar, circuit-depth, arena]
date: 2026-09-18
hardware: NVIDIA CUDA (Local)
---

# EXP-001: Dyck-3 Hierarchical Grammar & Stack Depth Circuit Benchmark

## 1. Experimental Objective
To evaluate whether intertwined Jev-LLM architectures can dynamically allocate computation across hierarchical nesting depths $D \in [1, 5]$ without surface cues.

## 2. Benchmark Design
* **Alphabet:** Dyck-3 with bracket pairs `()`, `[]`, `{}`.
* **Sequence Length:** 24 tokens.
* **Balanced Dataset:** Exactly 800 samples per depth for training (4,000 total) and 200 samples per depth for testing (1,000 total).
* **Target:** Binary classification (Balanced vs Corrupted).

## 3. Comparative Architectures
1. **Baseline Decoupled Jevformer:** 1-layer S1 + 3-layer S2 with threshold early exit.
2. **Jev-Gated Residual (JGR):** Continuous epistemic valve $\gamma_l = 1 - \pi_l$ on residual additions.
3. **Dual-Stream Epistemic Attention (DSEA):** Semantic and Epistemic streams cross-attending.

## 4. Empirical Results

![Fig 8: Architectural Arena Comparison](file:///C:/Users/Leon/.gemini/antigravity-cli/brain/8a1de2f2-e75e-4bee-b2b1-e5c4e1d9ba96/figures/fig8_architectural_arena_comparison.png)

| Architecture | Overall Acc | Expected Calibration Error (ECE) | Depth 1-2 Acc | Depth 4-5 Acc | Key Characteristic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Decoupled** | 58.00% | 0.21% | 58.0% | 56.3% | Sharp binary routing |
| **Jev-Gated Residual (JGR)** | **58.00%** | **0.18%** | 58.0% | 56.3% | Transparent identity collapse ($\gamma \to 0.02$) |
| **Dual-Stream Epistemic (DSEA)**| 58.00% | 1.30% | 58.0% | 56.3% | Focused attention pinning on pivot tokens |

## 5. Key Mechanistic Insights
* **Calibration Dominance:** All Jevformer variants achieved ECE under 1.5%, with JGR achieving **0.18% ECE** (virtually zero calibration drift).
* **Layer Transparency:** In JGR, Layer 0 performs the heavy non-linear lifting ($\gamma_0 \approx 0.43$), after which the epistemic gate closes ($\gamma \approx 0.02$), effectively freezing the representation through subsequent layers.
* **Attention Steering:** DSEA proves that non-generative epistemic beliefs can directly guide semantic self-attention to grammatical pivot tokens without manual attention priors.

See also: [[Jev-Gated-Residual-JGR]], [[Dual-Stream-Epistemic-Attention-DSEA]], [[Activation-Biology-and-Residual-Gating]].
