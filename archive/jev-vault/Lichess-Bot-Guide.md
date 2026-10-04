---
type: documentation
project: Jess-Hyperbullet Lichess Bot
created: 2026-09-18
tags: [lichess, bot, hyperbullet, jevformer, triprocess, deployment]
---

# ⚡ Jess-Hyperbullet: Autonomous Lichess Bot

**Account Profile:** [https://lichess.org/@/jess-hyperbullet](https://lichess.org/@/jess-hyperbullet)  
**Status:** Verified Official Lichess Bot (`BOT` title active)  
**Architecture:** Jevformer Tri-Process Neural Engine (128-neuron CReLU + Epistemic $\text{Noul}$ Search Gating)  
**Source Module:** `jev-vault/src/lichess_bot.py`

---

## 🔐 1. Credential Security & Storage

The bot API token is stored securely in `.env` in the repository root:

```ini
# .env (Protected & Ignored by .gitignore)
LICHESS_BOT_TOKEN=lip_your_bot_token_here
LICHESS_BOT_USERNAME=jess-hyperbullet
```

`.gitignore` has been updated with explicit rules preventing any accidental commitment of credentials:
```gitignore
# Environment variables and secrets
.env
.env*
*.token
secrets/
tokens/
```

---

## 🚀 2. Bot Engine Capabilities

The bot operates on the **Jevformer Tri-Process Engine**:
* **System 0:** Classical PeSTO piece-square priors and material invariants.
* **System 1:** 128-neuron CReLU fast reflex accumulator evaluating in $<1.0\text{ms}$ per ply.
* **System 2:** Negamax Alpha-Beta with MVV-LVA move ordering and tactical check priority.
* **Epistemic Gating ($\tau = 0.70$):** In quiet positions ($\text{Noul} \ge 0.70$), the bot moves within $1\text{ms}$, banking clock time. In tactical crises ($\text{Noul} < 0.70$), it dynamically deploys search depth (2–3 plies).

---

## 🎮 3. How to Play Against the Bot on Lichess

1. Log into your personal account on [Lichess.org](https://lichess.org).
2. Visit the bot profile: **[https://lichess.org/@/jess-hyperbullet](https://lichess.org/@/jess-hyperbullet)**.
3. Click the ⚔️ **Challenge** icon.
4. Select your preferred time control:
   * **Hyperbullet:** `1/2+0` (30 seconds) or `15s+0`
   * **Bullet:** `1+0` or `2+1`
   * **Blitz:** `3+0` or `5+0`
5. Click **Challenge** (Rated or Casual).
6. The bot automatically accepts standard chess challenges, sends a welcome greeting in game chat, and plays moves autonomously.
7. Upon game completion, the match transcript and Jev activations are automatically archived in:
   * `data/lichess_games/{game_id}.json`
   * `jev-vault/analysis/lichess_games/{game_id}.json`

---

## 🛠️ 4. Running and Managing the Bot Daemon

### Starting the Daemon (Listening Mode)
```bash
python -u "jev-vault/src/lichess_bot.py"
```

### Challenging Another Player / Bot Directly
You can also issue challenges outward to test the bot against other engines (e.g. `maia1` or friends):
```bash
# Challenge a player to 30s hyperbullet
python -u "jev-vault/src/lichess_bot.py" --challenge <username> --time 30 --inc 0

# Challenge to 1+0 rated bullet
python -u "jev-vault/src/lichess_bot.py" --challenge <username> --time 60 --inc 0 --rated
```

---

## 📊 5. Local Web GUI Enhancements

In the local Web GUI ([http://127.0.0.1:8765](http://127.0.0.1:8765)):
* **🏳️ Resign Button:** Added to the controls row in Tab 1 (Bullet Arena). Allows the human player to resign gracefully when down material, immediately ending the game, stopping clocks, recording the match in history, and updating the review tab.
