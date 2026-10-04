# Contract 02: TypeSafe Jev Decision Foundation Model (RLCD)

> **Status:** Fused Kernels Machine-Verified | Multi-Task Pipeline Scaled on Modal L40S | Drop-in Wire Engine Ready  
> **Core Primitive:** High-throughput, sub-100ms typed decisions emitting `(Choice, Score, Noul)` triples without fragile JSON parsing.

---

## 🏛️ Executive Overview

Contract 02 implements the **TypeSafe Jev Decision Foundation Model**, inspired by the TypeSafe AI primitive and anchored in **Reinforcement Learning from Contrastive Distillation (RLCD)** (Yang et al., arXiv:2307.12950) and **Strictly Proper Scoring Rules** (Brier, Ranked Probability Score, Logarithmic).

In modern agentic architectures, delegating micro-decisions (tool dispatch, action gating, context pruning) to heavy System 2 models incurs 1,500ms–5,000ms latency, context drift, and high token costs. Jev provides a **calibrated System 1 reflex engine**:

```
Input Context + Candidates ───> [ Jev Decision Model ] ───> {
                                                              Choice: Candidate ID (int)
                                                              Score:  Calibrated Probability [0, 1]
                                                              Noul:   Epistemic Confidence [0, 1]
                                                            }
                                                            (Latency < 100ms)
```

---

## 🔬 Theoretical Foundations

### 1. RLCD & Proper Scoring Calibration
Standard RLHF induces sycophancy and overconfidence. Contract 02 optimizes strictly convex proper scoring rules where expected loss is uniquely minimized when the reported credence matches true conditional probability:
$$\mathbb{E}_{y \sim P}[S(\hat{p}, y)] \ge \mathbb{E}_{y \sim P}[S(P(y=1), y)]$$

### 2. High-Performance Software Stack
* **Fused Brier Kernel (`fused_brier_kernel.py`):**
  - Computes masked softmax, Gaussian exploration, Brier loss, and Noul in one pass.
  - Closed-form analytic backward pass:
    $$\frac{\partial \mathcal{L}_b}{\partial z_{b, i}} = \frac{2}{\tau \cdot K_b} p_{b, i} \left[ (p_{b, i} - y_{b, i}) - \sum_{k=0}^{K_b-1} p_{b, k} (p_{b, k} - y_{b, k}) \right]$$
  - Machine-precision verified ($6.94 \times 10^{-18}$ error vs. PyTorch autograd). Zero autograd tape allocations.
* **Batched Option Tensor Gather (`model/test_batched_gather.py`):**
  - Replaces sequential Python batch loops with a single GPU `torch.gather` on `last_hidden`. 6.35x throughput increase.
* **Fused Proper Scoring Rules (`fused_proper_scoring_kernel.py`):**
  - Vectorized Ranked Probability Score (RPS) for ordinal outcomes and Spherical scoring.
* **Cardinality-Calibrated Temperature Scaling (`temperature_scaling.py`):**
  - Per-cardinality temperature parameter $T(K)$ for option counts $K \in [2, 10]$.

---

## 📊 Multi-Task Ingestion & Zero-Shot Split Strategy

Ingests `tasksource-instruct-v0` (5.3M examples across 485 tasks), stratified into 18 task families:
- **Training Families (80%):** NLI, Multiple-Choice QA, Topic Classification, Sentiment, Grammar Analysis (~80,000 to 300,000 examples).
- **Held-Out Zero-Shot Families (Strictly Excluded from Gradient Updates):**
  1. `TaskFamily.PARAPHRASE`: `mrpc`, `paws`, `qqp`, `stsb`.
  2. `TaskFamily.COMMONSENSE`: `hellaswag`, `piqa`, `winogrande`, `copa`.
  3. `TaskFamily.FACT_VERIFICATION`: `fever`, `vitaminc`, `climate_fever`.

---

## 🌐 Ecosystem Compatibility

Compatible with the broader Jev ecosystem as cataloged on [Awesome Jev Projects](https://github.com/logicrw/awesome-jev-projects) (691+ verified projects):
- **Browser & Desktop Control:** Cua, Jev-ultrafast, typesafe-computer-use, Jev-cu, Aside-jev.
- **Model Routing & Guardrails:** Fast classifier gates before expensive reasoning LLMs.
- **Drop-in Wire Engine:** `openjev_engine.py` provides full wire compatibility with `POST /v1/systemone` requests.

---

## 📂 Directory Layout

```
contracts/02-rlcd-decision/
├── README.md                          # This contract document
├── config.py                          # Hyperparameters & path specifications
├── reproducibility.py                 # Deterministic seed locking & env guards
├── model/                             # Neural architecture & kernels
│   ├── decision_heads.py              # Choice, Score, Noul multi-heads
│   ├── fused_brier_kernel.py          # Analytic Brier gradient kernel
│   ├── fused_proper_scoring_kernel.py # Analytic RPS & Spherical kernels
│   ├── fused_epistemic_router.py      # Microsecond decision routing
│   ├── option_marker.py               # Option tensor pooling
│   ├── temperature_scaling.py         # Platt / Temperature scaling
│   ├── openjev_engine.py              # Drop-in TypeSafe wire engine
│   ├── openjev_engine_v2.py           # Engine v2 runtime
│   └── test_*.py                      # Unit tests & machine precision checks
├── training/                          # Optimization pipelines
│   ├── rlcd_trainer.py                # Contrastive distillation & proper scoring
│   ├── scoring_rules.py               # Objective functions
│   └── cross_entropy_baseline.py      # Standard CE benchmark
├── data/                              # Data loaders & splits
│   ├── tasksource_loader.py           # Streaming tasksource ingestion
│   ├── task_registry.py               # 18 task families & schemas
│   └── splits.py                      # Train, val, and held-out splits
├── evaluation/                        # Calibration & verification
│   ├── calibration_metrics.py         # ECE, MCE, Brier score, Reliability curves
│   ├── eval_harness.py                # Multi-task evaluation harness
│   └── jev_api_comparison.py          # Benchmark vs live TypeSafe Jev API
├── modal_runs/                        # Cloud GPU execution (Modal)
│   ├── train_a10.py                   # L40S/A10G multi-task training
│   └── eval_diffusion_gemma.py        # Backbone comparative evaluation
└── scripts/                           # Driver scripts & figure generators
```

---

## 🚀 Quickstart

Run kernel verification suite:
```bash
python contracts/02-rlcd-decision/model/test_kernel_integration_suite.py
```

Run local prototype smoke test:
```bash
python contracts/02-rlcd-decision/scripts/run_local_prototype.py
```

Run multi-task calibration evaluation:
```bash
python contracts/02-rlcd-decision/scripts/run_evaluation.py
```
