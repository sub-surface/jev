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

### 2.4 Bounded Proper Scoring vs. Unbounded Logarithmic Divergence (MIT ICLR 2026 Theorem 1)
A foundational breakthrough presented by Damani et al. (*Beyond Binary Rewards: Training Language Models to Reason About Their Uncertainty*, ICLR 2026) explains why pure logarithmic RLCD can suffer from optimization instabilities and degenerate accuracy collapse in low-confidence regimes.

Consider a binary decision problem where the model predicts probability $p \in [0, 1]$ for the correct outcome:
- Under the standard logarithmic scoring rule:
  $$S_{\text{log}}(p, 1) = \log p, \quad S_{\text{log}}(p, 0) = \log(1 - p)$$
  As confidence drops toward zero ($p \to 0$), the penalty difference diverges to negative infinity:
  $$\lim_{p \to 0} [S_{\text{log}}(p, 1) - S_{\text{log}}(p, 0)] = -\infty$$
  In policy gradient training (e.g. REINFORCE or PPO), if an instance is difficult or noisy, this unbounded negative penalty creates an extreme gradient explosion that actively incentivizes the policy to collapse its predictions or produce non-committal uniform distributions across all tasks.

- Under the bounded **Brier scoring rule** (RLCR):
  $$S_{\text{brier}}(\mathbf{p}, y) = 1 - \|\mathbf{p} - \mathbf{e}_y\|_2^2 = 1 - \left[ (1 - p_y)^2 + \sum_{k \neq y} p_k^2 \right]$$
  For binary decisions:
  $$S_{\text{brier}}(p, 1) - S_{\text{brier}}(p, 0) = \left(1 - (1 - p)^2\right) - \left(1 - p^2\right) = 2p - 1$$
  Notice that:
  $$\sup_{p \in [0, 1]} |S_{\text{brier}}(p, 1) - S_{\text{brier}}(p, 0)| \le 1$$

#### Theorem 1 (Bounded Scoring Rule Monotonicity — Damani et al., 2026)
*Under any bounded proper scoring rule $S(\mathbf{p}, y)$ where $|S| \le M < \infty$, the expected reward $J(\theta) = \mathbb{E}_{y \sim \mathbf{q}}[S(\mathbf{p}_\theta, y)]$ is strictly monotonically increasing in true prediction accuracy $p_y$ everywhere on the interior of the simplex:*
$$\frac{\partial}{\partial q_y} \mathbb{E}[S(\mathbf{p}, y)] > 0 \quad \forall q_y \in (0, 1)$$
*Consequently, optimizing a bounded proper scoring rule mathematically guarantees that the policy cannot improve its expected reward by sacrificing task accuracy. Brier RLCR simultaneously preserves task accuracy while strictly regularizing calibration.*

---

### 2.5 Rewarding Doubt & Non-Zero Error Mass (TUM 2026)
Bani-Harouni et al. (*Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression*, TUM/MCML 2026) demonstrate that standard RL penalties on incorrect answers systematically push models into extreme overconfidence on their subset of correct predictions.

To counteract this, the scoring mechanism explicitly rewards expressed doubt:
$$R_{\text{doubt}}(\mathbf{p}, y) = \begin{cases} \log p_y & \text{if } \arg\max \mathbf{p} = y \\ \log(1 - \max_{k \neq y} p_k) & \text{if } \arg\max \mathbf{p} \neq y \end{cases}$$
When the model makes an incorrect prediction, instead of receiving a catastrophic flat penalty, it receives a higher reward proportional to its expressed reservation $(1 - p_{\text{pred}})$. This transforms calibration from a passive regularizer into an active survival mechanism during reinforcement learning.

---

### 2.6 Ternary Incentives & The Calibrated Abstention Frontier (Meta TruthRL 2026)
Wei et al. (*TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning*, Meta Reality Labs / UVA / UW, ICML 2026) investigate the mechanics of ternary reward structures for calibrated decision-making and selective abstention:
$$R(y, y^*) = \begin{cases} +1 & \text{if correct} \\ 0 & \text{if abstained / routed} \\ -c_{\text{wrong}} & \text{if incorrect} \end{cases}$$

In Jev's architecture, this directly governs the dual-route epistemic threshold:
$$\mathbb{E}[R(\text{commit})] = p_y (+1) + (1 - p_y)(-c_{\text{wrong}}) \ge R(\text{abstain}) = 0$$
Solving for the optimal decision boundary:
$$p_y - c_{\text{wrong}}(1 - p_y) \ge 0 \implies p_y^* = \frac{c_{\text{wrong}}}{1 + c_{\text{wrong}}}$$

For symmetric error penalties ($c_{\text{wrong}} = 1$):
$$p^* = \frac{1}{1 + 1} = 0.5$$
This provides the game-theoretic foundation for Jev's confidence metric:
$$\text{Noul} = 1 - \tau = \max_{k} p_k$$
Whenever $\text{Noul} < 0.5$, committing an answer in zero-shot or adversarial conditions yields negative expected utility compared to triggering epistemic routing or abstention.

---

### 2.7 Calibration-Aware Reinforcement Learning (CARL) & The Extraction Dilemma (USC / AWS / Oracle, ACL 2026)
Yaldiz et al. (*Balancing Classification and Calibration Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning*, Findings of ACL 2026) diagnose the fundamental root cause of overconfidence in reasoning and decision models:
1. **The Extraction Dilemma:** In chain-of-thought and autoregressive decision models, the final decision token $y_d$ functions primarily as an extraction step from the reasoning trace rather than an autonomous calibration assessment. 
2. **The RLVR Overconfidence Curse:** Because reinforcement learning with verifiable rewards (RLVR, PPO, GRPO) strongly rewards correct reasoning traces and penalizes incorrect ones without calibrated alternatives, the policy rapidly saturates the decision token's probability $p_\theta(y_d) \to 1.0$.

To resolve this, CARL introduces a hybrid objective that applies standard policy gradients to reasoning/context tokens while applying a **calibration-aware target distribution** directly to the decision token:
$$\mathcal{L}_{\text{CARL}}(y_d; \theta) = -\sum_{c \in C} q(c) \log p_\theta(c \mid x, y_{<d})$$
where the target distribution $q(c)$ is dynamically conditioned on generation correctness:
$$q(c) = \begin{cases} \mathbf{1}[c = y^*] & \text{if generation is correct } (y_d = y^*) \\ \frac{1}{|C|} & \text{if generation is incorrect } (y_d \neq y^*) \end{cases}$$

#### Mathematical Physics of the CARL Pull:
When the model makes an error, the loss gradient becomes:
$$\nabla_z \mathcal{L}_{\text{CARL}} = \mathbf{p} - \frac{1}{|C|} \mathbf{1}$$
This exerts an inward force pulling the probability vector directly toward the barycenter (the uniform centroid $\frac{1}{|C|} \mathbf{1}$) rather than allowing the model to remain peaky on an incorrect alternative. This guarantees that mistaken decisions naturally collapse toward minimum confidence ($p \to \frac{1}{|C|}$), keeping $\text{Noul} \approx 0$ and enabling downstream deferral.

---

### 2.8 Verified Scaling-Binning Recalibration & Brier Decomposition (Stanford, NeurIPS 2019)
Kumar, Liang, and Ma (*Verified Uncertainty Calibration*, NeurIPS 2019) establish that standard post-hoc scaling methods (temperature scaling, Platt scaling) improve empirical loss but lack sample-efficient verification bounds:
1. **The Brier Decomposition:**
   $$\text{MSE}(f) = \mathbb{E}[(f(X) - Y)^2] = \text{CE}(f)^2 + \text{Sharpness}(f)$$
   where $\text{CE}(f) = \left(\mathbb{E}[|f(X) - \mathbb{E}[Y \mid f(X)]|^2]\right)^{1/2}$ is the calibration error, and $\text{Sharpness}(f) = \mathbb{E}[\text{Var}(Y \mid f(X))]$ measures discriminative resolution. Optimizing the Brier score guarantees that improvements cannot occur without reducing calibration error or increasing predictive sharpness.
2. **The Scaling-Binning Calibrator:**
   To overcome the $O(B / \epsilon^2)$ sample complexity of naive histogram binning, Kumar et al. prove that first fitting a parametric scaling function $g \in \mathcal{G}$ (e.g. regularized temperature scaling) and then applying uniform-mass binning over $g(x)$ reduces sample complexity to:
   $$\mathcal{O}\left(B \log B + \frac{\log B}{\epsilon^2}\right)$$
   This provides formal theoretical justification for Jev's cardinality-bucketed temperature scaling architecture.

---

### 2.9 Risk-Constrained Planning & Value Alignment (Stanford ICML 2019 & UAMDP 2026)
Malik et al. (*Calibrated Model-Based Deep RL*, ICML 2019) and Koren et al. (*UAMDP: Uncertainty-Aware Markov Decision Process*, 2026) prove why calibrated decision models are essential when embedded inside larger autonomous systems:
In multi-step rollouts or agentic decision pipelines, the expected utility of downstream action $a$ is:
$$\hat{Q}(s, a) = \sum_{s'} \hat{P}(s' \mid s, a) \left[ R(s, a, s') + \gamma V(s') \right]$$
If the predicted probabilities $\hat{P}(s')$ are miscalibrated (even if top-1 classification accuracy is high), the estimation errors compound exponentially over planning depth $H$:
$$\|\hat{Q}_H - Q^*\|_\infty \le \mathcal{O}\left( \frac{\gamma}{(1 - \gamma)^2} \text{ECE}(P) \right)$$
By enforcing strictly proper scoring rules (Brier RLCR) and regularized temperature scaling, Jev provides calibrated probabilities that bound downstream value error and enable risk-sensitive Conditional Value-at-Risk (CVaR) gating.

---

## 3. The Physics of Exploration Noise: Gaussian Weierstrass Smoothing

In RLCD, Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ is added directly to logits prior to softmax:
$$\mathbf{p}_{\text{noisy}} = \text{Softmax}(z + \sigma \epsilon)$$

### Theorem 2 (Lipschitz Regularization via Gaussian Convolving)
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

#### Theorem 3 (End-Marker Causal Parity)
*Let $T_{\text{opts}}$ be the token position following the completion of all option texts. For any set of marker tokens placed at $t_k \ge T_{\text{opts}}$:*
$$\forall k \in \{1, \dots, K\}, \forall j < T_{\text{opts}}: \quad M_{t_k, j} = 0$$
*Every option marker representation $h(t_k)$ attends to the full context and all candidate options with identical receptive field.*

---

## 5. Summary of Recommended Architectural Modifications

1. **Adopt the End-Marker Syntax:** Place `[OPT_A]..[OPT_Z]` at the sequence terminus so all candidate heads attend over all candidate texts.
2. **Option-Order Augmentation (Permutation Invariance):** Randomly permute the textual order of options during training to cancel residual positional priors.
3. **Cardinality Bucketing:** Scale temperatures separately per cardinality bucket ($K=2$, $K \in [3,5]$, $K \in [6,10]$) to address varying entropy floors across task types.
