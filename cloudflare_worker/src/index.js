/**
 * ==========================================================================
 * ⚡ JEVFORMER EDGE WORKER: jev.subsurfaces.net
 * ==========================================================================
 * Edge-hosted minimal monochromatic chess cockpit, live Lichess bot telemetry,
 * and PGN review surface powered by Cloudflare Workers.
 * ==========================================================================
 */

const HTML_CONTENT = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Jevformer 8x8 | Subsurfaces Chess</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #080A0D;
      --bg-card: #11141B;
      --bg-elevated: #181C26;
      --border: #1E2330;
      --text-primary: #F3F4F6;
      --text-muted: #8892B0;
      --text-dim: #4B5568;
      --accent-cyan: #38BDF8;
      --accent-green: #10B981;
      --accent-red: #EF4444;
      --sq-light: #C4C9D4;
      --sq-dark: #687282;
      --sq-last: #60A5FA;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-base);
      color: var(--text-primary);
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: var(--bg-card);
      border-bottom: 1px solid var(--border);
      padding: 14px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand h1 {
      font-size: 1.15rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .brand span.tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 3px 8px;
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent-cyan);
      border-radius: 4px;
    }

    .header-links {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .btn-cta {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--accent-cyan);
      color: #000;
      font-weight: 700;
      font-size: 0.8rem;
      padding: 8px 16px;
      border-radius: 6px;
      text-decoration: none;
      transition: transform 0.15s, background 0.15s;
    }
    .btn-cta:hover {
      background: #7dd3fc;
      transform: translateY(-1px);
    }

    .container {
      max-width: 1240px;
      width: 100%;
      margin: 0 auto;
      padding: 24px;
      flex: 1;
      display: grid;
      grid-template-columns: 560px 1fr;
      gap: 28px;
    }

    @media (max-width: 1024px) {
      .container { grid-template-columns: 1fr; }
    }

    /* Left: Board & Match Controls */
    .left-col {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .board-frame {
      width: 560px;
      height: 560px;
      background: var(--bg-card);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
      display: grid;
      grid-template-columns: repeat(8, 70px);
      grid-template-rows: repeat(8, 70px);
    }

    .square {
      width: 70px;
      height: 70px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 2.2rem;
      user-select: none;
      transition: background 0.1s;
    }
    .square.light { background: var(--sq-light); color: #1E2330; }
    .square.dark { background: var(--sq-dark); color: #0E1116; }
    .square.last { background: #93C5FD !important; }

    /* Right: Cockpit & Live Telemetry */
    .cockpit {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px 20px;
    }

    .card-title {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .stats-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }

    .stat-tile {
      background: var(--bg-elevated);
      border-radius: 6px;
      padding: 12px;
    }
    .stat-lbl {
      font-size: 0.68rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
    }
    .stat-val {
      font-size: 1.3rem;
      font-weight: 800;
      color: var(--text-primary);
      margin-top: 4px;
      font-family: 'JetBrains Mono', monospace;
    }

    .bot-status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      text-transform: uppercase;
    }
    .bot-status-pill.online {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
    }
    .bot-status-pill .dot {
      width: 6px;
      height: 6px;
      background: currentColor;
      border-radius: 50%;
    }

    .recent-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 260px;
      overflow-y: auto;
    }

    .game-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-elevated);
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 0.82rem;
      text-decoration: none;
      color: var(--text-primary);
      transition: background 0.15s;
    }
    .game-item:hover {
      background: #232938;
    }

    .pgn-preview {
      width: 100%;
      height: 120px;
      background: var(--bg-base);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      padding: 10px;
      resize: none;
      outline: none;
    }

    footer {
      border-top: 1px solid var(--border);
      padding: 16px 28px;
      font-size: 0.75rem;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>⚡ Jevformer 8x8</h1>
      <span class="tag">Tri-Process Engine</span>
    </div>
    <div class="header-links">
      <button id="wakeBtn" onclick="triggerWake()" style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border); padding: 8px 14px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; font-family: 'JetBrains Mono', monospace;">
        ⚡ Modal: Zero-Idle Ready
      </button>
      <a href="https://lichess.org/@/jess-hyperbullet" target="_blank" onclick="challengeBot(event)" class="btn-cta">
        ⚔️ Challenge @jess-hyperbullet
      </a>
    </div>
  </header>

  <div class="container">
    <!-- Left Column: Board -->
    <div class="left-col">
      <div class="board-frame" id="boardGrid">
        <!-- Rendered via JS -->
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">
        <span>FEN: <span id="curFen">rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1</span></span>
      </div>
    </div>

    <!-- Right Column: Cockpit -->
    <div class="cockpit">
      <div class="card">
        <div class="card-title">
          <span>Lichess Bot Status</span>
          <span class="bot-status-pill online"><span class="dot"></span> Online 24/7</span>
        </div>
        <div class="stats-row">
          <div class="stat-tile">
            <div class="stat-lbl">Rating</div>
            <div class="stat-val" id="botRating">3000?</div>
          </div>
          <div class="stat-tile">
            <div class="stat-lbl">Reflex Latency</div>
            <div class="stat-val">&lt; 35ms</div>
          </div>
          <div class="stat-tile">
            <div class="stat-lbl">CReLU Sparsity</div>
            <div class="stat-val">60.0%</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Recent Lichess Games</span>
          <a href="https://lichess.org/@/jess-hyperbullet" target="_blank" style="color: var(--accent-cyan); font-size: 0.72rem; text-decoration: none;">View All &rarr;</a>
        </div>
        <div class="recent-list" id="gamesList">
          <div style="color: var(--text-dim); font-size: 0.8rem; text-align: center; padding: 12px;">Loading games from Lichess...</div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Latest Game PGN Archive</span>
          <button onclick="copyPgn()" style="background: none; border: none; color: var(--accent-cyan); cursor: pointer; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">Copy</button>
        </div>
        <textarea id="pgnBox" class="pgn-preview" readonly placeholder="PGN will appear here..."></textarea>
      </div>
    </div>
  </div>

  <footer>
    <div>Hosted globally on Cloudflare Edge &bull; Domain: <code>jev.subsurfaces.net</code></div>
    <div>Engine: Jevformer Tri-Process (128-neuron CReLU + Epistemic Noul Gating)</div>
  </footer>

  <script>
    const INITIAL_BOARD = [
      ['♜','♞','♝','♛','♚','♝','♞','♜'],
      ['♟','♟','♟','♟','♟','♟','♟','♟'],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['♙','♙','♙','♙','♙','♙','♙','♙'],
      ['♖','♘','♗','♕','♔','♗','♘','♖'],
    ];

    function renderBoard(grid) {
      const el = document.getElementById("boardGrid");
      el.innerHTML = "";
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const sq = document.createElement("div");
          sq.className = "square " + ((r + c) % 2 === 0 ? "light" : "dark");
          sq.textContent = grid[r][c] || "";
          el.appendChild(sq);
        }
      }
    }

    renderBoard(INITIAL_BOARD);

    async function loadBotInfo() {
      try {
        const res = await fetch("/api/bot_info");
        const data = await res.json();
        if (data.perfs && data.perfs.bullet) {
          const r = data.perfs.bullet.rating;
          const prov = data.perfs.bullet.prov ? "?" : "";
          document.getElementById("botRating").textContent = r + prov;
        }
      } catch(e) {
        console.error("Bot info fetch error:", e);
      }
    }

    async function loadRecentGames() {
      try {
        const res = await fetch("/api/recent_games");
        const data = await res.json();
        const list = document.getElementById("gamesList");
        list.innerHTML = "";

        if (!data || data.length === 0) {
          list.innerHTML = '<div style="color: var(--text-dim); font-size: 0.8rem; text-align: center; padding: 12px;">No games recorded yet.</div>';
          return;
        }

        data.forEach(g => {
          const item = document.createElement("a");
          item.className = "game-item";
          item.href = "https://lichess.org/" + g.id;
          item.target = "_blank";

          const white = g.players?.white?.user?.name || "White";
          const black = g.players?.black?.user?.name || "Black";
          const result = g.winner ? (g.winner === "white" ? "1-0" : "0-1") : "1/2-1/2";

          item.innerHTML = \`
            <div><strong>\${white}</strong> vs <strong>\${black}</strong> (\${g.speed})</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--accent-cyan);">\${result}</div>
          \`;
          list.appendChild(item);
        });

        if (data[0] && data[0].pgn) {
          document.getElementById("pgnBox").value = data[0].pgn;
        }
      } catch(e) {
        console.error("Recent games error:", e);
      }
    }

    function copyPgn() {
      const box = document.getElementById("pgnBox");
      box.select();
      navigator.clipboard.writeText(box.value);
      alert("PGN copied to clipboard!");
    }

    async function triggerWake() {
      const btn = document.getElementById("wakeBtn");
      btn.textContent = "⏳ Waking Modal Cloud...";
      btn.style.color = "var(--accent-cyan)";
      try {
        const res = await fetch("/api/wake_bot");
        const data = await res.json();
        btn.textContent = "✅ Awakened (15m Active)";
        btn.style.color = "var(--accent-green)";
        setTimeout(() => {
          btn.textContent = "⚡ Modal: Zero-Idle Ready";
          btn.style.color = "var(--text-primary)";
        }, 8000);
      } catch(e) {
        btn.textContent = "⚠️ Error waking";
      }
    }

    function challengeBot(e) {
      // Fire wake ping silently in background
      fetch('/api/wake_bot').catch(() => {});
    }

    loadBotInfo();
    loadRecentGames();
  </script>
</body>
</html>
`;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // API: Bot account status with edge cache
    if (url.pathname === "/api/bot_info") {
      const cache = caches.default;
      let cachedRes = await cache.match(request);
      if (cachedRes) return cachedRes;

      const lichessRes = await fetch("https://lichess.org/api/user/jess-hyperbullet", {
        headers: { "Accept": "application/json" }
      });
      const data = await lichessRes.text();
      const response = new Response(data, {
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "public, max-age=30"
        }
      });
      ctx.waitUntil(cache.put(request, response.clone()));
      return response;
    }

    // API: Recent games played by the bot
    if (url.pathname === "/api/recent_games") {
      const lichessRes = await fetch("https://lichess.org/api/games/user/jess-hyperbullet?max=5&pgnInJson=true", {
        headers: { "Accept": "application/x-ndjson" }
      });
      const ndjson = await lichessRes.text();
      const games = ndjson.trim().split("\\n").filter(Boolean).map(line => {
        try { return JSON.parse(line); } catch(e) { return null; }
      }).filter(Boolean);

      return new Response(JSON.stringify(games), {
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "public, max-age=15"
        }
      });
    }

    // API: Wake bot on Modal Cloud (Zero-Idle Wake-on-Demand)
    if (url.pathname === "/api/wake_bot") {
      try {
        const modalRes = await fetch("https://sub-surface--jess-hyperbullet-bot-wake.modal.run", {
          headers: { "User-Agent": "Jevformer-Edge-Worker/1.0" }
        });
        const modalData = await modalRes.json();
        return new Response(JSON.stringify(modalData), {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: err.toString() }), {
          status: 500,
          headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }
        });
      }
    }

    // Healthcheck
    if (url.pathname === "/health" || url.pathname === "/healthz") {
      return new Response(JSON.stringify({ status: "healthy", domain: "jev.subsurfaces.net" }), {
        headers: { "Content-Type": "application/json" }
      });
    }

    // Default: Serve minimal elegant Web Cockpit
    return new Response(HTML_CONTENT, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "public, max-age=300"
      }
    });
  }
};
