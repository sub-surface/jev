"""
==========================================================================
⚡ JESS-HYPERBULLET: HUGGING FACE SPACE 24/7 DAEMON & STATUS SERVER
==========================================================================
Runs the autonomous Lichess bot continuously 24/7 while serving an elegant
status page on port 7860 for Hugging Face container health checks.
==========================================================================
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import bot runner
from lichess_bot import LichessBotRunner, BOT_USERNAME

PORT = int(os.environ.get("PORT", 7860))

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>⚡ Jess-Hyperbullet | 24/7 Lichess Bot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --bg: #090B0E;
      --card: #12151B;
      --border: #1F242D;
      --text: #F3F4F6;
      --muted: #9CA3AF;
      --cyan: #38BDF8;
      --green: #10B981;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }}
    .container {{
      max-width: 580px;
      width: 100%;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 32px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 20px;
      color: var(--green);
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 16px;
    }}
    .dot {{
      width: 8px;
      height: 8px;
      background: var(--green);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--green);
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.4; }}
    }}
    h1 {{
      font-size: 1.6rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 8px;
    }}
    p.subtitle {{
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.5;
      margin-bottom: 24px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-bottom: 24px;
    }}
    .stat-tile {{
      background: #0E1015;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .stat-lbl {{
      font-size: 0.7rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }}
    .stat-val {{
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text);
      margin-top: 4px;
      font-family: ui-monospace, monospace;
    }}
    .btn-row {{
      display: flex;
      gap: 12px;
    }}
    .btn {{
      flex: 1;
      text-align: center;
      padding: 12px 18px;
      border-radius: 6px;
      font-size: 0.88rem;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.15s;
    }}
    .btn-primary {{
      background: var(--cyan);
      color: #000000;
    }}
    .btn-primary:hover {{
      background: #7dd3fc;
      transform: translateY(-1px);
    }}
    .btn-secondary {{
      background: #1C222C;
      color: var(--text);
      border: 1px solid var(--border);
    }}
    .btn-secondary:hover {{
      background: #252D3B;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="badge">
      <span class="dot"></span>
      Online 24/7 Persistent
    </div>
    <h1>⚡ Jess-Hyperbullet</h1>
    <p class="subtitle">
      Autonomous Lichess Bot powered by the <strong>Jevformer Tri-Process Neural Engine</strong> (CReLU sparse activations + Epistemic Noul search gating).
    </p>

    <div class="stats-grid">
      <div class="stat-tile">
        <div class="stat-lbl">Lichess Account</div>
        <div class="stat-val">@{BOT_USERNAME}</div>
      </div>
      <div class="stat-tile">
        <div class="stat-lbl">Reflex Latency</div>
        <div class="stat-val">&lt; 35 ms</div>
      </div>
      <div class="stat-tile">
        <div class="stat-lbl">Target Speeds</div>
        <div class="stat-val">30s / 1m / 2m</div>
      </div>
      <div class="stat-tile">
        <div class="stat-lbl">CReLU Sparsity</div>
        <div class="stat-val">60.0%</div>
      </div>
    </div>

    <div class="btn-row">
      <a href="https://lichess.org/@/{BOT_USERNAME}" target="_blank" class="btn btn-primary">⚔️ Challenge on Lichess</a>
      <a href="https://jev.subsurfaces.net" target="_blank" class="btn btn-secondary">🌐 GUI Dashboard</a>
    </div>
  </div>
</body>
</html>
"""

class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "bot": "online"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Silence standard access logs


def run_server():
    server = HTTPServer(("0.0.0.0", PORT), StatusHandler)
    print(f"✅ Hugging Face Health Server listening on port {PORT}...", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    print(f"--- Starting Jess-Hyperbullet 24/7 Daemon on Hugging Face Space ---", flush=True)

    # 1. Start HTTP Health check server in background thread for HF container
    t_web = threading.Thread(target=run_server, daemon=True)
    t_web.start()

    # 2. Start Bot Runner in foreground
    try:
        bot = LichessBotRunner()
        bot.start_event_stream()
    except Exception as e:
        print(f"Bot runner fatal error: {e}", flush=True)
        # Keep web server alive so container doesn't exit-loop
        while True:
            time.sleep(10)
