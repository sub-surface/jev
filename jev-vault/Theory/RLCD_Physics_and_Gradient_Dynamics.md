---
title: The Physics of Calibrated Decisions — Gradient Dynamics, Proper Scoring Rules, and Causal Asymmetry
date: 2026-09-19
status: theoretical treatise
tags: [physics, information-theory, rlcd, proper-scoring, causal-attention, gradient-dynamics]
---

# The Physics of Calibrated Decisions: Gradient Dynamics, Proper Scoring Rules, and Causal Asymmetry

**Author:** The Research Collective  
**Context:** General Multi-Task Calibrated Decision Architecture (Jev Analogue)

---

## 1. The Information-Theoretic Failure Mode of Cross-Entropy

In modern deep learning, the ubiquitous loss for classification is cross-entropy:
$$\mathcal{L}_{\text{CE}}(\theta) = -\log p_y = -z_y + \log \sum_{j=1}^K e^{z_j}$$

The gradient with respect to logit $z_k$ is:
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_k} = p_k - \mathbf{1}[k = y]$$

### The Divergence Pathology
Notice that $\nabla_{z_k} \mathcal{L}_{\text{CE}} \to 0$ if and only if $p_y \to 1$ and $p_k \to 0$ for $k \neq y$.
Because the softmax function $\sigma(z)_y = \frac{e^{z_y}}{\sum_j e^{z_j}}$ requires infinite logit separation ($z_y - z_k \to \infty$) to reach $p_y = 1$, stochastic gradient descent under cross-entropy continually inflates the norm of the weight matrices $\|W\|_2 \to \infty$. 

When the trained network is presented with an ambiguous, out-of-distribution, or zero-shot instance $x^*$, the inflated logit scale produces **peaky, overconfident probabilities** ($p \approx 0.99$) even when the underlying epistemic uncertainty is maximal ($H(y|x) \approx \log K$). This is the mathematical root cause of the calibration crisis identified by Guo et al. (2017).

---

## 2. Strictly Proper Scoring Rules: Gradient Mechanics of RLCD

In TypeSafe's RLCD framework, cross-entropy is superseded by **strictly proper scoring rules** serving as objective reward functions.

### 2.1 The Logarithmic Scoring Rule
$$S_{\text{log}}(\mathbf{p}, y) = \log p_y$$
The gradient with respect to logit $z$ is:
$$\nabla_z S_{\text{log}} = \mathbf{e}_y - \mathbf{p}$$
In expectation over true data distribution $\mathbf{q}$:
$$\mathbb{E}_{y \sim \mathbf{q}}[\nabla_z S_{\text{log}}] = \mathbf{q} - \mathbf{p}$$
Stationarity $\mathbb{E}[\nabla_z S] = 0$ is achieved **strictly and uniquely** when $\mathbf{p} = \mathbf{q}$.

### 2.2 The Spherical Scoring Rule
$$S_{\text{sph}}(\mathbf{p}, y) = \frac{p_y}{\|\mathbf{p}\|_2}$$
Let us derive the exact logit gradient for $S_{\text{sph}}$.
By the quotient rule:
$$\frac{\partial S_{\text{sph}}}{\partial p_i} = \frac{\mathbf{1}[i = y]}{\|\mathbf{p}\|_2} - \frac{p_y p_i}{\|\mathbf{p}\|_2^3}$$
Applying the Jacobian of the softmax $\frac{\partial p_i}{\partial z_k} = p_i(\mathbf{1}[i=k] - p_k)$:
$$\frac{\partial S_{\text{sph}}}{\partial z_k} = \sum_{i=1}^K \frac{\partial S_{\text{sph}}}{\partial p_i} p_i (\mathbf{1}[i=k] - p_k)$$
$$= \frac{p_k}{\|\mathbf{p}\|_2} \left[ \mathbf{1}[k=y] - p_y \right] - \frac{p_y}{\|\mathbf{p}\|_2^3} \left[ p_k^2 - p_k \|\mathbf{p}\|_2^2 \right]$$

#### Physical Meaning:
1. **Self-Normalizing Gradient:** Unlike cross-entropy where gradients persist until $z \to \infty$, the spherical gradient scales inversely with $\|\mathbf{p}\|_2$.
2. **Overconfidence Resistance:** As $\mathbf{p}$ approaches a deterministic vertex $\mathbf{e}_y$, $\|\mathbf{p}\|_2 \to 1$, and both terms contract gracefully to 0.
3. **Symmetric Exploration Pull:** When $\mathbf{p}$ is diffuse ($\mathbf{p} \approx \frac{1}{K} \mathbf{1}$), $\|\mathbf{p}\|_2 = \frac{1}{\sqrt{K}}$, amplifying the gradient signal by $\sqrt{K}$ to pull the model out of uninformative local minima.

---

### 2.3 The Ranked Probability Score (RPS) for Ordinal Primitives
For tasks of primitive type `score` (ratings, severity scales, continuous intervals), categorical cross-entropy treats all errors equally: predicting 4 stars on a 5-star product is penalized identically to predicting 1 star.

The Ranked Probability Score penalizes according to metric distance on the ordinal lattice:
$$\text{RPS}(\mathbf{p}, y) = -\frac{1}{K-1} \sum_{m=1}^{K-1} \left( F_m - H_m \right)^2$$
where $F_m = \sum_{j=1}^m p_j$ is the cumulative distribution function (CDF), and $H_m = \mathbf{1}[m \ge y]$ is the empirical Heaviside step at target $y$.

#### The CDF Gradient:
$$\frac{\partial \text{RPS}}{\partial p_j} = -\frac{2}{K-1} \sum_{m=j}^{K-1} (F_m - H_m)$$
The gradient exerted on probability $p_j$ is proportional to the **integrated CDF discrepancy** from rank $j$ upward. Misallocating mass far from the true target $y$ compounds through multiple summation terms, creating a smooth parabolic potential well that centers the probability distribution around the ground truth.

---

## 3. The Physics of Exploration Noise: Gaussian Weierstrass Smoothing

In RLCD, Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ is added directly to logits prior to softmax:
$$\mathbf{p}_{\text{noisy}} = \text{Softmax}(z + \sigma \epsilon)$$

### Theorem 1 (Lipschitz Regularization via Gaussian Convolving)
*The expected reward under logit noise $J_\sigma(z) = \mathbb{E}_{\epsilon}[R(\text{Softmax}(z + \sigma \epsilon))]$ is the Weierstrass transform of the reward surface, and satisfies a strict Lipschitz gradient bound:*
$$\|\nabla_z J_\sigma(z)\|_2 \le \frac{\sup |R|}{\sigma}$$

#### Proof Sketch:
By the log-derivative trick:
$$\nabla_z \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)} [R(z + \epsilon)] = \frac{1}{\sigma^2} \mathbb{E}_{\epsilon}[\epsilon R(z + \epsilon)]$$
Applying the Cauchy-Schwarz inequality:
$$\|\nabla_z J_\sigma(z)\|_2 \le \frac{1}{\sigma^2} \sqrt{\mathbb{E}[\|\epsilon\|^2]} \sup |R| = \frac{\sqrt{K}}{\sigma} \sup |R|$$
As a direct consequence, **adding exploration noise prevents logit explosions**. The model cannot sharpen its decision boundaries faster than $\mathcal{O}(1/\sigma)$. Annealing $\sigma$ from $0.12 \to 0.02$ allows early coarse global alignment while preserving smooth calibration boundaries in the final policy.

---

## 4. The Causal Attention Asymmetry Paradox & The End-Marker Solution

### The Causal Asymmetry Paradox
In standard autoregressive language models (such as Qwen2.5), the attention mask $M$ is strictly lower-triangular:
$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

Consider the naive inline option formatting:
```
[Context]
[OPT_A] Candidate A text
[OPT_B] Candidate B text
[OPT_C] Candidate C text
```

Let $t_A, t_B, t_C$ denote the token positions of the option markers. Because $t_A < t_B < t_C$:
$$\text{Attention}(t_A \to \text{Text}_B) = 0, \quad \text{Attention}(t_A \to \text{Text}_C) = 0$$
$$\text{Attention}(t_C \to \text{Text}_A) > 0, \quad \text{Attention}(t_C \to \text{Text}_B) > 0$$

**The Paradox:** Candidate C has full visibility over the entirety of Candidate A and B, whereas Candidate A is evaluated in total ignorance of its competitors! This introduces severe structural position bias and causes option markers to score different contextual representations.

### The End-Marker Theorem & Solution
To enforce bidirectional semantic parity in a causal decoder without rewriting CUDA attention kernels:
1. Present all candidate options first in the prompt body.
2. Emit all option-marker tokens **at the very end of the sequence**:

```
[Instruction & Context]

Options:
A. Full text of option A
B. Full text of option B
C. Full text of option C

Decision: [OPT_A] [OPT_B] [OPT_C]
```

#### Theorem 2 (End-Marker Causal Parity)
*Let $T_{\text{opts}}$ be the token position following the completion of all option texts. For any set of marker tokens placed at $t_k \ge T_{\text{opts}}$:*
$$\forall k \in \{1, \dots, K\}, \forall j < T_{\text{opts}}: \quad M_{t_k, j} = 0$$
*Every option marker representation $h(t_k)$ attends to the full context and all candidate options with identical receptive field.*

---

## 5. Summary of Recommended Architectural Modifications

1. **Adopt the End-Marker Syntax:** Place `[OPT_A]..[OPT_Z]` at the sequence terminus so all candidate heads attend over all candidate texts.
2. **Option-Order Augmentation (Permutation Invariance):** Randomly permute the textual order of options during training to cancel residual positional priors.
3. **Cardinality Bucketing:** Scale temperatures separately per cardinality bucket ($K=2$, $K \in [3,5]$, $K \in [6,10]$) to address varying entropy floors across task types.
