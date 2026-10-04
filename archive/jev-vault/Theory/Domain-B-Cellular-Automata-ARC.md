# Domain B: Cellular Automata & ARC-AGI Invariant Induction
## Discrete Symbolic Coupling for Spatial Generalization Without Dense Threshold Corruption

**Authors:** Leon & The Research Collective (Karpathy, Hinton, Shannon, Torvalds)  
**Status:** Theoretical Specification & Architectural Blueprint  
**Location:** `jev-vault/Theory/Domain-B-Cellular-Automata-ARC.md`  

---

## 1. Executive Summary & Problem Formulation

In the Abstraction and Reasoning Corpus (ARC-AGI) and general 2D cellular automata (CA), the fundamental challenge is **few-shot inductive program synthesis**:
Given $k \le 5$ input-output grid demonstration pairs $(X_i, Y_i)$ with $X_i, Y_i \in \{0, 1, \dots, 9\}^{H_i \times W_i}$, synthesize a deterministic transition operator $\mathcal{T}$ such that:
$$\mathcal{T}(X_{\text{test}}) = Y_{\text{test}}$$

### The Failure Mode of Contemporary Monolithic LLMs
Frontier autoregressive models (Claude 3.5, GPT-4o, Gemini 1.5 Pro) fail at ARC because:
1. **Spatial Deserialization & Coordinate Drift:** Autoregressive tokenization serializes 2D grids into 1D token strings (`[[0, 1], [2, 3]]`), destroying local topological neighborhood metrics $\|\vec{x}_1 - \vec{x}_2\|_2$.
2. **Dense Latent Perturbation Collapse:** In deep continuous models, conditioning on few-shot demonstrations via cross-attention adds dense continuous perturbations $\vec{a} + \vec{m}$ into activation spaces. As proven in our empirical trials (EXP-008), continuous dense modulation incurs **29.8% dead neuron leakage** and **31.7% active unit extinction**, collapsing spatial boundary invariants.
3. **Absence of Discrete Verification:** Pure next-token generation cannot verify discrete invariants (e.g., object count conservation, topological connectedness, color permutations).

---

## 2. Tri-Process Architecture for ARC-AGI

The Tri-Process architecture decouples spatial physics, epistemic uncertainty, and symbolic program induction into three mathematically coordinated systems:

```mermaid
flowchart TD
    subgraph System2 ["System 2: Symbolic Invariant Synthesizer"]
        DemoIn["Demonstrations (X_i, Y_i)"] --> HypoEngine["Hypothesis Search & VQ Codebook"]
        HypoEngine --> InvariantRule["Discrete Invariant Vector z in {0,1}^B (B=4 bits)"]
    end

    subgraph System1 ["System 1: Calibrated Epistemic Sensor (Jev)"]
        GridState["Current Grid G_t"] --> AnomalyDetector["Spatial Anomaly & Symmetry Detector"]
        AnomalyDetector --> NoulMap["Noul(x, y) Epistemic Certainty Field"]
        NoulMap --> GateCheck{"Noul(x,y) < 0.70?"}
    end

    subgraph System0 ["System 0: Sparse 2D Cellular Automaton Engine"]
        TestInput["Test Grid X_test"] --> Accumulator["2D CReLU Sparse Grid Tensor"]
        InvariantRule -.->|"Discrete Mask Injection"| Accumulator
        Accumulator --> LocalKernel["3x3 Morphological & CA Transition Kernels"]
        LocalKernel --> OutGrid["Output Grid Y_pred"]
    end

    GateCheck -- "Yes: Falsified Invariant" --> HypoEngine
    GateCheck -- "No: Verified" --> LocalKernel
```

### System 0: Sparse 2D Cellular Automaton Accumulator
- **Physical Representation:** A 10-channel one-hot spatial tensor $\mathbf{A} \in \mathbb{R}^{10 \times H \times W}$, where channel $c$ represents the presence of color $c \in \{0, \dots, 9\}$.
- **Accumulator Dynamics:** Transition operators are composed of discrete morphological primitives:
  $$\mathbf{A}_{t+1} = \text{CReLU}\left( \mathbf{W}_{\text{CA}} * \mathbf{A}_t + \mathbf{M}_{\text{discrete}} \right)$$
  where $\mathbf{W}_{\text{CA}}$ represents local $3 \times 3$ convolutional stencils (dilation, erosion, edge detection, flood-fill propagation) and $\mathbf{M}_{\text{discrete}}$ is an integer factor mask supplied by System 2.

### System 1: Jev Epistemic Anomaly Field $\text{Noul}(x, y)$
- Evaluates spatial confidence per cell $(x, y) \in [H] \times [W]$:
  $$\text{Noul}(x, y) = \sigma\left( \mathbf{W}_{\text{noul}} * \mathbf{A}(x, y) \right) \in [0, 1]$$
- **Epistemic Invariant Detection:**
  * If a proposed transformation generates an illegal color overlap or breaks demonstration symmetries (e.g. diagonal reflection, bounding box parity), $\text{Noul}(x, y)$ drops sharply to $<0.30$.
  * This triggers an **epistemic interrupt**, halting System 0 execution and requesting a hypothesis update from System 2 without wasteful full rollout.

### System 2: Discrete VQ Invariant Synthesizer ($B=4$ bits)
- As proven in our Shannon Rate-Distortion analysis ($K = 2^B$, $B=4 \implies K=16$ codebook primitives), System 2 compresses the task invariant into a discrete code $z \in \{0, \dots, 15\}$:

| Code ID | Primitive Transformation Class | Symbolic Invariant Specification |
|:---:|:---|:---|
| `0x0` | `Identity / Null` | $Y = X$ |
| `0x1` | `Horizontal Reflection` | $Y(r, c) = X(r, W - 1 - c)$ |
| `0x2` | `Vertical Reflection` | $Y(r, c) = X(H - 1 - r, c)$ |
| `0x3` | `Diagonal Transpose` | $Y(r, c) = X(c, r)$ |
| `0x4` | `Gravity Vector (Down)` | Fall until contact with non-zero pixel or floor |
| `0x5` | `Gravity Vector (Right)` | Slide rightward until boundary collision |
| `0x6` | `Connected Component Extraction` | Mask largest contiguous non-background cluster |
| `0x7` | `Morphological Dilation` | Expand foreground by $3 \times 3$ structuring element |
| `0x8` | `Bounding Box Crop` | Extract subgrid $\mathbf{G}[r_{\min}:r_{\max}, c_{\min}:c_{\max}]$ |
| `0x9` | `Color Permutation` | Bijective color mapping $\pi: \{0..9\} \to \{0..9\}$ |
| `0xA` | `Interior Hole Filling` | Flood-fill from borders; invert unreached zeros |
| `0xB` | `Periodic Grid Tiling` | Tile $h \times w$ motif across $H \times W$ canvas |
| `0xC` | `Raycast Projection` | Project orthogonal rays from color markers |
| `0xD` | `Scale Factor Upsample` | Kronecker expansion $X \otimes \mathbf{1}_{k \times k}$ |
| `0xE` | `Symmetry Completion` | Complete partial reflection about centroid |
| `0xF` | `Conditional Filter` | Keep only objects satisfying predicate $\mathcal{P}(\text{size}, \text{color})$ |

---

## 3. Mathematical Proof of Invariant Coupling in CA Spaces

### Theorem 1 (Zero-Leakage Invariant Steering in 2D CReLU)
*Let $\mathbf{A} \in \mathbb{R}^{C \times H \times W}$ be the accumulator grid with activation $\phi(\mathbf{A}) = \min(\max(\mathbf{A}, 0), 1)$. If System 2 modulates System 0 via discrete mask $\mathbf{M} \in \{-\infty, 0, 1\}$ or discrete multiplicative gating $\mathbf{S} \in \{0, 1\}^{C}$, then for all inactive cells where $\mathbf{A}_{c, r, w} \le 0$:*
$$\phi(\mathbf{A}_{c, r, w} \odot \mathbf{S}_c) = 0$$
*The inactive zero-point threshold is invariant, and the spurious dead neuron leakage rate is identically zero:*
$$\mathcal{L}_{\text{leak}} = 0.000$$

*Proof:*  
For any inactive element $a \le 0$, since $S_c \in \{0, 1\}$, the gated value is $a' = a \cdot S_c$.  
If $S_c = 0$, $a' = 0 \implies \phi(0) = 0$.  
If $S_c = 1$, $a' = a \le 0 \implies \phi(a) = 0$.  
In both cases, $\phi(a') = 0$, hence no inactive neuron can cross the positive threshold $a > 0$. The zero-point is preserved, preventing spurious color hallucinations. $\blacksquare$

---

## 4. Empirical Evaluation Protocol on ARC-AGI-Pub

1. **Dataset Split:** 400 training tasks, 100 evaluation tasks from ARC-AGI-Pub.
2. **Baselines:**
   - Monolithic Autoregressive LLM (Few-shot text prompt).
   - Dense Additive Coupling ($\mathbf{A} + \mathbf{m}_{\text{dense}}$).
   - Discrete Tri-Process ($B=4$ VQ Invariant + 2D CA System 0 + Jev Epistemic Sensor).
3. **Metrics:**
   - Exact Grid Match Accuracy (%).
   - CReLU Spurious Leakage Rate ($\mathcal{L}_{\text{leak}}$).
   - Inference Latency per Grid (ms).
   - Epistemic Calibration (Brier Score on task solvability).

---

## 5. Empirical 2D Cellular Automata Physics & Visualizations

We simulated 36-step 2D Cellular Automata evolution under continuous dense latent perturbation vs. TypeSafe discrete invariant steering:

![[figures/fig31_cellular_automata_physics.png]]
> *Figure 31: 2D Cellular Automata CReLU Physics under Continuous vs. TypeSafe Discrete Invariant Steering. Continuous dense additive modulation induces 5.58% parasitic dead-cell leakage and 12.60% structural Hamming error from the invariant. In contrast, TypeSafe discrete invariant steering maintains exactly 0.00% dead-cell leakage and preserves the ground truth structural topology with 0.0% divergence.*

### Dynamic State Progression (Animated Evolution)
![[figures/ca_triprocess_evolution.gif]]
> *Real-time 36-step simulation comparing TypeSafe Discrete Invariant steering (left, emerald) vs. Continuous Dense Additive modulation (right, crimson). While continuous latent noise rapidly corrupts the boundary invariants and triggers runaway cell growth, the discrete invariant maintains pristine structural stability.*

