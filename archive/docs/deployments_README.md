# JEV Live Deployments & Showcases

> **Status:** Production Services Operational | Edge Latency < 50ms | 24/7 Lichess Bot

This directory consolidates the live, working deployment artifacts and user-facing applications of the JEV project.

---

## 🚀 Live Services

### 1. Edge Cockpit (`deployments/edge-cockpit/`)
- **Live URL:** [jev.subsurfaces.net](https://jev.subsurfaces.net)
- **Platform:** Cloudflare Workers (V8 Edge Compute)
- **Features:**
  - Real-time interactive neural forward-pass visualizer.
  - 13-bitplane board state slicer.
  - Channel excitation heatmap for 64-channel Squeeze-and-Excitation attention.
  - 128-neuron CReLU sparse accumulator inspector with semantic cluster breakdown.
  - Dual epistemic readouts (Value centipawns, Noul gate vs $\tau=0.35$, Top-5 Policy moves).
- **Deployment:**
  ```bash
  cd deployments/edge-cockpit
  npx wrangler deploy
  ```

---

### 2. Lichess Bot Daemon (`deployments/lichess-bot/`)
- **Live Account:** [@jess-hyperbullet](https://lichess.org/@/jess-hyperbullet)
- **Platform:** Docker container / Serverless daemon / Hugging Face Spaces
- **Features:**
  - Autonomous 24/7 hyperbullet chess play (0.5+0 and 1+0 time controls).
  - Sub-15ms move selection latency powered by Jev epistemic gating:
    - Positional plies: Instant reflex move ($\text{Noul} \ge \tau$, $<1\text{ms}$).
    - Tactical crises: Quiescence capture search ($\text{Noul} < \tau$).
  - Full FIDE rulebook enforcement (`claim_draw=True`, 3-fold repetition, 50-move rule, checkmate).
- **Deployment:**
  ```bash
  cd deployments/lichess-bot
  docker build -t jev-lichess-bot .
  docker run -d --env-file .env jev-lichess-bot
  ```
