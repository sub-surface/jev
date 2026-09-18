---
type: documentation
project: Jess-Hyperbullet Permanent 24/7 Hosting
created: 2026-09-18
tags: [hosting, cloudflare, huggingface, permanent, docker, lichess]
---

# 🌐 Permanent 24/7 Free Hosting Architecture & Cloudflare Setup

This guide explains how to host **`jess-hyperbullet`** permanently online 24/7 for **$0.00**, and how to set up **`jev.subsurfaces.net`** on Cloudflare.

---

## 🏛️ 1. The Core Architecture: Split Frontend & Daemon

```
┌──────────────────────────────────────┐       ┌───────────────────────────────────────┐
│       Cloudflare Pages / Worker      │       │     Hugging Face Spaces / Free VM     │
│         (jev.subsurfaces.net)        │       │             (Docker Daemon)           │
├──────────────────────────────────────┤       ├───────────────────────────────────────┤
│ • Minimal Elegant Chessboard UI      │       │ • 24/7 Persistent Lichess Stream Loop │
│ • Live Game Review & Stockfish       │       │ • Jevformer Tri-Process Neural Engine │
│ • Static Assets & Telemetry Displays │       │ • Auto-Accepts Challenges & Plays     │
│ • Free Global CDN Edge (0ms latency) │       │ • 2 vCPU, 16GB RAM, $0 Forever        │
└──────────────────────────────────────┘       └───────────────────────────────────────┘
                                ▲                                  ▲
                                │                                  │
                                └─────────────────┬────────────────┘
                                                  │
                                          ┌───────┴───────┐
                                          │  Lichess.org  │
                                          │  Bot API 2.0  │
                                          └───────────────┘
```

### Why a Standard Cloudflare Worker Alone Cannot Run the Bot Stream:
* **Protocol Requirement:** The Lichess Bot API relies on a long-lived, continuous HTTP chunked stream (`GET https://lichess.org/api/stream/event`).
* **Worker Limit:** Free Cloudflare Workers have a **30-second execution cutoff** and a **50ms CPU limit**. They cannot keep an idle connection open listening for games indefinitely without paid Durable Objects.
* **The Solution:** Use **Cloudflare Pages** for the public domain `jev.subsurfaces.net` (100% free), and use **Hugging Face Spaces** for the persistent Python bot daemon (100% free, 2 vCPUs, 16GB RAM, always-on).

---

## 🚀 2. Setting Up the Bot Daemon on Hugging Face Spaces (100% Free Forever)

Hugging Face provides free persistent containers (Docker or Python) with 2 vCPUs and 16GB RAM that run 24/7 without shutting down.

### Step 1: Create a Space on Hugging Face
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set **Space Name:** `jess-hyperbullet`.
3. Set **License:** `mit` or `apache-2.0`.
4. Set **SDK:** **Docker** (Blank).
5. Set **Space Hardware:** **CPU Basic (2 vCPU · 16 GB RAM) — FREE**.
6. Set **Visibility:** Public or Private.

### Step 2: Add Secret Token in Space Settings
1. Go to your Space **Settings** $\rightarrow$ **Variables and secrets**.
2. Click **New secret**:
   * **Key:** `LICHESS_BOT_TOKEN`
   * **Value:** `<your-lichess-bot-token>`
3. Click **Save**.

### Step 3: Push Code to Space
Clone the space repository and push the files from `deployment/`:
```bash
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/jess-hyperbullet
git push space main
```
The Docker container will build in ~60 seconds and run `python -u jev-vault/src/lichess_bot.py`. The bot will stay permanently connected to Lichess 24 hours a day, 365 days a year at zero cost!

---

## ⚡ 3. Setting Up `jev.subsurfaces.net` on Cloudflare

To host the minimal Web GUI and review surface at your custom subdomain:

### Step 1: Cloudflare Pages Deployment
1. Log into the [Cloudflare Dashboard](https://dash.cloudflare.com).
2. Navigate to **Workers & Pages** $\rightarrow$ **Create Application** $\rightarrow$ **Pages**.
3. Select **Connect to Git** (or upload directory directly).
4. Point to the web root folder containing `index.html` (or our static bundle).

### Step 2: Custom Subdomain Binding
1. In the Cloudflare Pages project, go to **Custom domains**.
2. Click **Set up a custom domain**.
3. Enter: `jev.subsurfaces.net`.
4. Since `subsurfaces.net` is already on Cloudflare, Cloudflare will automatically add the DNS `CNAME` record:
   ```
   CNAME  jev  ->  <your-project>.pages.dev  (Proxied)
   ```
5. SSL/TLS is automatically provisioned and active within 60 seconds!
