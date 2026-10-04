# Contract 03: Typed Self-Play Pretraining & Epiplexity (Zero Data)

> **Status:** Active Frontier | Toy 1 & Toy 2 Validated | SPEC.md Pinned  
> **Theoretical Anchors:** Cowsik et al. (arXiv:2609.30063) · Finzi et al. (arXiv:2601.03220)  
> **Core Hypothesis:** Pretraining a Jev-style System 1 typed-decision model on type-valid computable structure alone—curated by epiplexity rather than raw difficulty—produces transferable predictive priors without natural or human-curated data.

---

## 🏛️ Executive Overview

Contract 03 investigates the **pretraining foundation** of Jev decision models. While Contract 02 trains Jev on curated tasks via RLCD, and Contract 01 studies discrete coupling at inference time, Contract 03 asks:

> *Can a System 1 decision model acquire a reusable prior over structured decision processes tabula rasa—from synthetic, computable structure alone—before it ever sees human or task data?*

In *Self-Play Pretraining with Zero Data* (Cowsik et al., Sept 2026), two models co-evolve: a generator proposes programs for an untyped universal Turing machine (Brainf\*ck), and a learner predicts byte sequences via next-token prediction, with the generator rewarded via preconditioned gradient alignment:
$$r_i = |\langle \nabla_\theta \mathcal{L}(y_i; \theta_e), P_e \odot \delta\theta_e \rangle|$$

### The Typed Jev Bet
1. **Typed Validity over Untyped Search:** Random untyped Brainf\*ck generation crashes 99.3% of the time. Type-directed program generation (e.g., typed stack VM) crashes 0.0% of the time, guaranteeing zero compute is wasted on syntax/runtime invalidity.
2. **Epiplexity over Raw Difficulty:** Raw difficulty can be trivialized by injecting pseudorandom entropy. We evaluate data through **Epiplexity** ($S_{\mathcal{F}, T}(X)$)—the structural, learnable information extractable by computationally bounded observer class $\mathcal{F}$ within budget $T$ (Finzi et al., 2026).

---

## 📊 Empirical Findings So Far

### Toy 1: Typed Stack VM + Curriculum Controller
- **Validity Precondition:** Typed stack VM program generation crashed **0/5,000** times (0.0%), whereas untyped generation crashed **99.3%** of the time.
- **Curriculum Dynamics:** The learner's loss fell below uniform baseline and the curriculum controller correctly stepped difficulty until learner capacity became the bottleneck.

### Toy 2: Epiplexity Zoo & Transfer Matrix (`epiplexity_zoo_experiment.py`)
Tested across 7 token-stream families (`iid`, `periodic`, `modadd`, `modmul`, `modadd_shuf`, `fsm`, `stack`):

| Family | Epiplexity Proxy | Final Loss (Nats) | Uniform Baseline |
| :--- | :---: | :---: | :---: |
| `fsm` | $0.520 \pm 0.057$ | $2.043 \pm 0.053$ | 2.890 |
| `periodic` | $0.348 \pm 0.055$ | $2.399 \pm 0.055$ | 2.890 |
| `modmul` | $0.343 \pm 0.045$ | $2.320 \pm 0.029$ | 2.890 |
| `stack` | $0.119 \pm 0.066$ | $2.061 \pm 0.172$ | 2.890 |
| `iid` (control) | $0.014 \pm 0.010$ | $2.892 \pm 0.011$ | 2.890 |
| `modadd_shuf` | $0.005 \pm 0.005$ | $2.908 \pm 0.010$ | 2.890 |
| `modadd` | $0.004 \pm 0.001$ | $2.910 \pm 0.003$ | 2.890 |

#### Crux Discoveries:
1. **Negative Transfer ($R = -0.151 \pm 0.009$):**
   Pretraining on `modmul` actively impaired performance on `modadd` (~15x above noise), despite both being "modular arithmetic". Reason: mod-prime multiplication generates short orbital cycles, acting as periodic memorization rather than general arithmetic recurrence.
2. **Observer Relativity & Structural Invisibility:**
   `modadd` was never learned by the tiny order-$K$ MLP; its loss remained pegged at the uniform baseline ($2.910$ vs $2.890$). Structure was completely invisible to a bounded, capacity-constrained observer.
3. **Epiplexity $\approx$ Difficulty at Toy Scale:**
   With a single weak observer, ranking by epiplexity proxy and ranking by final loss were nearly identical. Separating them requires a multi-scale observer spread ($M_0 \to M_3$).

---

## 🔬 Open Research Cruxes

- **Crux 1:** Does epiplexity cleanly separate from raw final-loss difficulty once evaluated across an observer capacity hierarchy ($M_0$ n-gram through $M_3$ Jev-scale)?
- **Crux 2:** Can "shared computational primitive" be verified a priori before pairing families for transfer?
- **Crux 3:** Does negative transfer (interference) require an explicit penalty in the generator's RL policy objective?

---

## 📂 Directory Layout

```
contracts/03-typed-selfplay/
├── README.md                      # This contract document
├── SPEC.md                        # The working specification & research bet
├── epiplexity_zoo_experiment.py   # Toy 2 benchmark: 7 families, transfer matrix, epiplexity proxy
└── ...                            # Next: multi-observer hierarchy & RL generator
```

---

## 🚀 Quickstart

Run the Toy 2 Epiplexity Zoo experiment:
```bash
python contracts/03-typed-selfplay/epiplexity_zoo_experiment.py
```
*(Runs completely in vanilla Python + NumPy with deterministic seed averaging).*
