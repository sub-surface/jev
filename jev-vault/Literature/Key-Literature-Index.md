---
tags: [literature, arxiv, rlcd, epiplexity, nnue, vq, mechinterp]
created: 2026-09-18
---

# Frontier Research Literature & Theoretical Anchors

## 1. Reinforcement Learning from Contrastive Distillation (RLCD)
* **Citation:** Yang, K., Klein, D., Celikyilmaz, A., Peng, N., & Tian, Y. (2023). *RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment.* arXiv:2307.12950.
* **Core Insight:** Eliminates human preference bias and sycophancy by prompting language models with paired contrastive instructions $(p^+, p^-)$ (positive adherence vs. negative violation). Produces crisp epistemic separation.
* **Role in Jev:** Powers TypeSafe AI's Jev primitive ($40M seed by Diogo Almeida). Optimizes decisions alongside calibrated confidence via Brier proper scoring, mapping unstructured states to typed `(Choice, Score, Noul)` triples in 1ms.

## 2. Epiplexity & Computationally Bounded Information
* **Citation:** Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., & Wilson, A. G. (2026). *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence.* arXiv:2601.03220.
* **Core Insight:** Shannon entropy assumes infinite observer compute. Epiplexity $S_{\mathcal{F}, T}(X)$ measures the structural information extractable within class $\mathcal{F}$ and time bound $T$. The Epiplexity Gap $\Delta_{\text{epi}}(x) = H_{S1}(x) - H_{S2}(x)$ determines whether System 2 deliberation actually unlocks learnable bits or wastes FLOPs.

## 3. Stockfish NNUE & Clipped ReLU Sparse Accumulators
* **Citation:** Nasu, T. (2018); Stockfish Development Team (2020-2024). *Efficiently Updatable Neural Networks (NNUE).* Chess Programming Wiki / Stockfish Documentation.
* **Core Insight:** First layer is a linear accumulator ($a = Wx + b$) activated by Clipped ReLU ($\text{CReLU}(a) = \text{clamp}(a, 0, 1)$). Incremental updates occur in $\mathcal{O}(k \cdot D)$ time ($k$ changed features). Extremely sparse activations (>80% dead/zero units) allow microsecond counterfactual evaluations in tree search.
* **The Fatal Coupling Pitfall:** Injecting continuous dense additive vectors into CReLU shifts the zero-crossing threshold across all neurons simultaneously, causing representation interference.

## 4. Vector Quantization & Discrete Bottlenecks in Reasoning
* **Citations:** Van den Oord et al. (2017) *Neural Discrete Representation Learning*; recent 2024-2026 work on discrete latent planning and communication bottlenecks.
* **Core Insight:** Vector quantization forces continuous deliberation latents through a discrete codebook $\mathcal{C} = \{e_1, \dots, e_K\}$, bounding channel capacity to $\log_2 K$ bits and filtering out continuous perturbation noise.
