---
tags: [experiment, graph-reasoning, pcpr, calibration, modal]
date: 2026-09-18
hardware: NVIDIA A10G (Modal)
---

# EXP-002: Permutation Path Tracing (PCPR) & Zero-Surface Calibration

## 1. Experimental Setup
* **Benchmark:** Permutation Path Tracing (PCPR).
* **Constraints:** ZERO surface cues, NO prefix markers, NO leaking positional order.
* **Instances:** 3,000 holdout graphs with scrambled edge pairs and queries $(S, ?)$.
* **Model Size:** 3.35M parameter transformer on NVIDIA A10G.
* **Duration:** 277.4s ($0.0849 USD on Modal).

## 2. Head-to-Head Comparison: Jevformer vs. CALM
We compared **Jevformer** (trained with RLCD proper scoring) against **CALM** (Confident Adaptive Language Modeling; Schuster et al., 2022), which uses normalized Softmax Entropy.

![Fig 7: Epiplexity & Circuit Depth](file:///C:/Users/Leon/.gemini/antigravity-cli/brain/8a1de2f2-e75e-4bee-b2b1-e5c4e1d9ba96/figures/fig7_epiplexity_and_circuit_depth.png)

## 3. Results & Findings
* **Task Accuracy:** 32.00% across both models on unhinted permutation graphs (owing to induction head circuit limits on multi-hop compositions without scratchpads).
* **Expected Calibration Error (ECE):**
  * **Jevformer (RLCD Proper Scoring):** **1.85%**
  * **CALM (Softmax Entropy Baseline):** **25.70%**

### Critical Insight:
When a task cannot be solved, CALM produces peaky, overconfident probabilities, misallocating compute and exiting prematurely on deep failures.
**Jevformer knows that it does not know.** It outputs $\hat{p} \approx 0.338$, precisely matching its empirical success rate.

At $\tau = 0.20$, Jevformer saves **80% of FLOPs** with zero accuracy loss.

See also: [[Epiplexity-and-Bounded-Information]], [[RLCD-Proper-Scoring-Calibration]].
