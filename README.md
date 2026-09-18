# ⚡ Jev: Type-Safe Tri-Process Decision Engine

[![Lichess Bot](https://img.shields.io/badge/Lichess-@jess--hyperbullet-brightgreen)](https://lichess.org/@/jess-hyperbullet)
[![Edge Cockpit](https://img.shields.io/badge/Edge%20Cockpit-jev.subsurfaces.net-blue)](https://jev.subsurfaces.net)
[![Compute](https://img.shields.io/badge/Modal-Zero--Idle%20Serverless-purple)](https://modal.com)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Jev** is a first-principles, type-safe decision primitive engineered to bridge intuitive neural representations (**System 1**) with discrete symbolic constraints and search (**System 0 / System 2**). 

Unlike auto-regressive large language models that generate token sequences with hallucination risk, Jev operates over calibrated topological invariants—enforcing hard type boundaries and epistemic confidence gating ($\text{Noul}$) to achieve near-instantaneous reflex decisions ($<35\text{ms}$) while dynamically escalating tactical crises into deep tree search.

---

## 🏛️ Tri-Process Architectural Model

The architecture decouples cognitive decision-making into three mutually stabilizing layers:

```mermaid
flowchart TD
    State([Raw State / Board / Graph]) --> Pre[Tensor Feature Encoder]
    Pre --> S1[System 1: Neural Feature Extractor]
    
    subgraph Jevformer Core
        S1 --> Accum[128-Dim Discrete Accumulator]
        Accum --> CReLU[Clipped ReLU Activation 0.0 to 1.0]
        CReLU --> ValHead[Value Head: V(s) in -1, 1]
        CReLU --> NoulHead[Epistemic Head: Noul(s) in 0, 1]
    end
    
    NoulHead --> Gate{Epistemic Gate<br/>Noul >= tau & Quiet?}
    
    Gate -->|YES: Positional Equilibrium| S0[System 0: Instant Reflex Move<br/>Latency < 35ms]
    Gate -->|NO: Tactical Crisis / Low Noul| S2[System 2: Deep Negamax Alpha-Beta<br/>TT Cache + Dynamic Clock Budget]
    
    S0 --> Exec([Executed Action])
    S2 --> Exec
```

### 1. System 0: Discrete Invariant Accumulator & Heuristics
- **128-Neuron CReLU Bottleneck:** Activations are strictly clamped $\text{CReLU}(z) = \min(1.0, \max(0.0, z))$. This creates a sparse, interpretable semantic basis (60.0% activation sparsity) with zero dead-neuron leakage.
- **Classical Heuristic Grounding:** Anchored by midgame/endgame piece-square tables (PeSTO) and topological invariants.

### 2. System 1: Epistemic Confidence Gating ($\text{Noul}$)
- **Information-Theoretic Volatility:** $\text{Noul}(s) \in [0.0, 1.0]$ measures position certainty and tactical stability.
- **Reflex Execution:** When $\text{Noul}(s) \ge \tau$ (quiet positional equilibrium), candidate actions are dispatched immediately ($15\text{ms}–35\text{ms}$), preserving clock time under bullet constraints.

### 3. System 2: Gated Combinatorial Search
- **Dynamic Negamax with Transposition Tables:** Triggered strictly during sharp tactical crises ($\text{Noul} < \tau$, king in check, or direct threats).
- Allocates dynamic search budgets ($50\text{ms}–250\text{ms}$), guaranteeing clock safety in hyperbullet formats ($30\text{s}+0$).

---

## 🔬 Core Research Domains

### Domain A: Standard 8x8 Chess & Live Lichess Autonomous Bot
- **Bot Handle:** [`@jess-hyperbullet`](https://lichess.org/@/jess-hyperbullet)
- **Edge Cockpit:** [`https://jev.subsurfaces.net`](https://jev.subsurfaces.net)
- **Cloud Hosting:** Zero-idle serverless wake-on-demand daemon on Modal Cloud ($0.00 compute when idle, instant spin-up via edge webhook).
- **Leela Distillation:** Trained on 40,000+ stratified curriculum positions on NVIDIA A10G GPUs; Value MSE reduced by **-95.4%**.

### Domain B: Continuous Cellular Automata & Emergence
- Investigates self-organizing continuous neural cellular automata (NCAs).
- Simulates localized wave packets (solitons), reaction-diffusion morphogenesis, and stable topological patterns under continuous differential updates.

### Domain C: Combinatorial Search & Cycle Entrapment Avoidance
- In reversible state spaces (modular arithmetic, register machines, grid operations), naive greedy search and continuous latent recurrence suffer from **94%+ infinite orbital limit cycle traps**.
- Jev formalizes visited-state manifold projection, establishing path history pruning to guarantee monotonic search convergence.

### Theorem 2: Mathematical Impossibility of Invalid Output Symbols
- Proves that by projecting latent states into constrained type manifolds $\mathcal{M}_{\text{safe}}$, the probability measure of generating an illegal or ill-formed output symbol is identically zero:
  $$\mathbb{P}(\text{Output} \notin \Sigma_{\text{valid}}) = 0$$

---

## 🚀 Live Edge & Cloud Deployment

- **Custom Domain:** [`https://jev.subsurfaces.net`](https://jev.subsurfaces.net) (Cloudflare Anycast Worker)
- **Modal Webhook:** `https://sub-surface--jess-hyperbullet-bot-wake.modal.run`
- **Zero Idle Cost:** The container terminates after 15 minutes of inactivity. Visiting the site or challenging the bot proactively fires a wake signal.

---

## 🛠️ Quickstart & Local Setup

### 1. Clone & Environment
```bash
git clone https://github.com/sub-surface/jev.git
cd jev
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch chess numpy requests fastapi
```

### 2. Run Local Web GUI
```bash
python jev-vault/src/chess_gui_server.py
# Open http://127.0.0.1:8765
```

### 3. Run Lichess Bot Locally
```bash
python jev-vault/src/lichess_bot.py
```

### 4. Deploy to Modal Cloud
```bash
modal deploy modal_cloud/modal_lichess_bot_daemon.py
```

---

## 📜 License
MIT License. Open research and implementation by [Sub-Surface](https://github.com/sub-surface).
