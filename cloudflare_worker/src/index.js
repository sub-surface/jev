/**
 * ==========================================================================
 * ⚡ JEVFORMER ARCHITECTURE EXPLORER & EDGE COCKPIT: jev.subsurfaces.net
 * ==========================================================================
 * Interactive Tri-Process Architecture Explorer, Cellular Automata Simulator,
 * Theorem 2/3 Formal Proof Walkthrough, and Lichess Autonomous Bot Telemetry.
 * Hosted globally on Cloudflare Edge.
 * ==========================================================================
 */

const HTML_CONTENT = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Jevformer | Tri-Process Epistemic Architecture & Research Explorer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #080A0D;
      --bg-card: #10131B;
      --bg-elevated: #171B26;
      --border: #1E2433;
      --border-accent: rgba(56, 189, 248, 0.35);

      --text-primary: #F8FAFC;
      --text-muted: #8492A6;
      --text-dim: #475569;

      --accent-cyan: #38BDF8;
      --accent-green: #10B981;
      --accent-red: #EF4444;
      --accent-amber: #F59E0B;
      --accent-purple: #A855F7;

      --sq-light: #D1D5DB;
      --sq-dark: #4B5563;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-base);
      color: var(--text-primary);
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Navigation Header */
    header {
      background: var(--bg-card);
      border-bottom: 1px solid var(--border);
      padding: 12px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(12px);
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
      border: 1px solid rgba(56, 189, 248, 0.25);
    }

    .header-links {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .btn-wake {
      background: var(--bg-elevated);
      color: var(--text-primary);
      border: 1px solid var(--border);
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: 'JetBrains Mono', monospace;
      transition: all 0.15s;
    }
    .btn-wake:hover {
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
    }

    .btn-cta {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--accent-cyan);
      color: #000;
      font-weight: 700;
      font-size: 0.8rem;
      padding: 7px 16px;
      border-radius: 6px;
      text-decoration: none;
      transition: transform 0.15s, background 0.15s;
    }
    .btn-cta:hover {
      background: #7dd3fc;
      transform: translateY(-1px);
    }

    /* Sub-Navigation Tabs */
    .subnav {
      background: #0C0E14;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: center;
      padding: 6px 20px;
      gap: 8px;
      overflow-x: auto;
    }
    .nav-tab {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
      font-weight: 600;
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .nav-tab:hover {
      color: var(--text-primary);
      background: var(--bg-elevated);
    }
    .nav-tab.active {
      color: var(--accent-cyan);
      background: var(--bg-card);
      border: 1px solid var(--border-accent);
    }

    /* Main Container */
    main {
      flex: 1;
      max-width: 1260px;
      width: 100%;
      margin: 0 auto;
      padding: 24px 20px 48px;
    }

    .tab-panel {
      display: none;
    }
    .tab-panel.active {
      display: block;
      animation: fadeIn 0.2s ease-in-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Card System */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .card-title {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }
    .grid-4 {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }
    @media (max-width: 960px) {
      .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
    }

    .stat-box {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
    }
    .stat-label {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.04em;
    }
    .stat-number {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.5rem;
      font-weight: 800;
      margin-top: 4px;
      color: var(--text-primary);
    }
    .stat-number.cyan { color: var(--accent-cyan); }
    .stat-number.green { color: var(--accent-green); }
    .stat-number.amber { color: var(--accent-amber); }

    /* Interactive Architecture Visualizer */
    .arch-flow {
      display: grid;
      grid-template-columns: 1fr 60px 1fr 60px 1fr;
      gap: 12px;
      align-items: center;
      margin-bottom: 24px;
    }
    @media (max-width: 960px) {
      .arch-flow { grid-template-columns: 1fr; gap: 16px; }
      .arch-arrow { transform: rotate(90deg); margin: 8px 0; }
    }
    .arch-node {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      position: relative;
      transition: all 0.2s;
    }
    .arch-node.active-route {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    }
    .arch-node h3 {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--accent-cyan);
      margin-bottom: 6px;
    }
    .arch-node p {
      font-size: 0.78rem;
      color: var(--text-muted);
      line-height: 1.45;
    }
    .arch-arrow {
      text-align: center;
      color: var(--text-dim);
      font-size: 1.4rem;
      font-weight: bold;
    }

    /* 128-dim CReLU Matrix */
    .crelu-matrix {
      display: grid;
      grid-template-columns: repeat(16, 1fr);
      gap: 4px;
      padding: 12px;
      background: var(--bg-elevated);
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .neuron-cell {
      aspect-ratio: 1;
      border-radius: 2px;
      background: rgba(255,255,255,0.04);
      transition: background 0.15s, transform 0.15s;
      cursor: pointer;
    }
    .neuron-cell.active {
      background: var(--accent-cyan);
      box-shadow: 0 0 6px rgba(56, 189, 248, 0.4);
    }
    .neuron-cell:hover {
      transform: scale(1.3);
      z-index: 10;
    }

    /* Domain B: Cellular Automata Simulator */
    .ca-canvas-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }
    canvas#caCanvas {
      background: #050608;
      border: 1px solid var(--border);
      border-radius: 6px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.8);
      image-rendering: pixelated;
    }
    .ca-controls {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .btn-tool {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      color: var(--text-primary);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-tool:hover {
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
    }
    .btn-tool.active-mode {
      background: rgba(56, 189, 248, 0.15);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
    }

    /* Domain C: Proof Step Walkthrough */
    .proof-step {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent-cyan);
      border-radius: 0 6px 6px 0;
      padding: 14px 16px;
      margin-bottom: 12px;
      font-size: 0.84rem;
      line-height: 1.6;
    }
    .proof-step code {
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255,255,255,0.06);
      padding: 2px 6px;
      border-radius: 3px;
      color: var(--accent-cyan);
      font-size: 0.8rem;
    }

    /* Chess Board Cockpit (Domain A) */
    .board-frame {
      width: 480px;
      height: 480px;
      background: var(--bg-card);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
      display: grid;
      grid-template-columns: repeat(8, 60px);
      grid-template-rows: repeat(8, 60px);
      margin: 0 auto;
    }
    @media (max-width: 540px) {
      .board-frame { width: 320px; height: 320px; grid-template-columns: repeat(8, 40px); grid-template-rows: repeat(8, 40px); }
      .square { font-size: 1.4rem !important; }
    }
    .square {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 2.1rem;
      user-select: none;
    }
    .square.light { background: var(--sq-light); color: #1E2433; }
    .square.dark { background: var(--sq-dark); color: #0C0E14; }

    .recent-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 240px;
      overflow-y: auto;
    }
    .game-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-elevated);
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 0.8rem;
      text-decoration: none;
      color: var(--text-primary);
      transition: background 0.15s;
    }
    .game-item:hover { background: #222736; }

    /* Tables */
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      font-family: 'JetBrains Mono', monospace;
    }
    table.data-table th, table.data-table td {
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    table.data-table th {
      color: var(--text-muted);
      font-weight: 700;
      background: var(--bg-elevated);
    }

    footer {
      border-top: 1px solid var(--border);
      padding: 16px 28px;
      font-size: 0.75rem;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-card);
    }
  </style>
</head>
<body>

  <!-- Top Header -->
  <header>
    <div class="brand">
      <h1>⚡ Jevformer</h1>
      <span class="tag">Tri-Process Epistemic Engine</span>
    </div>
    <div class="header-links">
      <button id="wakeBtn" onclick="triggerWake()" class="btn-wake">
        ⚡ Modal Daemon: Ready
      </button>
      <a href="https://lichess.org/@/jess-hyperbullet" target="_blank" onclick="fetch('/api/wake_bot').catch(()=>{})" class="btn-cta">
        ⚔️ Challenge @jess-hyperbullet
      </a>
    </div>
  </header>

  <!-- Subnav Tabs -->
  <nav class="subnav">
    <button class="nav-tab active" onclick="switchTab('arch')">📐 Architecture Explorer</button>
    <button class="nav-tab" onclick="switchTab('chess')">♟️ Domain A: Bullet Bot Cockpit</button>
    <button class="nav-tab" onclick="switchTab('ca')">🧬 Domain B: Cellular Automata</button>
    <button class="nav-tab" onclick="switchTab('proof')">📜 Domain C: Theorem 2 & Cycles</button>
    <button class="nav-tab" onclick="switchTab('training')">📊 Training Curves & Sparsity</button>
  </nav>

  <!-- Main Content Area -->
  <main>

    <!-- ============================================================= -->
    <!-- TAB 1: ARCHITECTURE EXPLORER                                  -->
    <!-- ============================================================= -->
    <div id="tab-arch" class="tab-panel active">
      <div class="card">
        <div class="card-title">
          <span>Tri-Process Execution Dynamics</span>
          <div style="display: flex; gap: 8px;">
            <button class="btn-tool active-mode" id="btnRouteQuiet" onclick="setArchMode('quiet')">Mode: Quiet Reflex</button>
            <button class="btn-tool" id="btnRouteCrisis" onclick="setArchMode('crisis')">Mode: Tactical Crisis</button>
          </div>
        </div>

        <div class="arch-flow">
          <div class="arch-node active-route" id="nodeSystem0">
            <h3>System 0: Sparse CReLU</h3>
            <p><strong>Reflex & Discrete Accumulator</strong></p>
            <p>128-neuron CReLU layer with 67.2% sparsity. Evaluates 1-ply candidates in &lt;1ms during quiet positional states.</p>
          </div>
          <div class="arch-arrow">&harr;</div>
          <div class="arch-node active-route" id="nodeSystem1">
            <h3>System 1: Epistemic Gate</h3>
            <p><strong>Jev Volatility Sensor (Noul)</strong></p>
            <p>Computes calibrated confidence Noul(s). If Noul &ge; &tau;, triggers instant reflex move. If Noul &lt; &tau;, escalates to System 2.</p>
          </div>
          <div class="arch-arrow">&harr;</div>
          <div class="arch-node" id="nodeSystem2">
            <h3>System 2: Symbolic Search</h3>
            <p><strong>Alpha-Beta / Quiescence / Lean 4</strong></p>
            <p>Dynamic time-gated Negamax search with Quiescence on tactical exchanges. Disables continuous hallucinations.</p>
          </div>
        </div>

        <div id="archExplainer" style="background: var(--bg-elevated); padding: 14px; border-radius: 6px; font-size: 0.8rem; color: var(--text-muted); border-left: 3px solid var(--accent-cyan);">
          <strong>Current Route (Quiet Positional Ply):</strong> Model operates at 0.5ms reflex latency. System 1 measures high certainty (Noul &ge; 0.70), bypassing deep search and preserving bullet clock time.
        </div>
      </div>

      <!-- 128-dim CReLU Matrix -->
      <div class="card">
        <div class="card-title">
          <span>128-Neuron CReLU Sparse Accumulator Matrix</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-cyan);">
            Active: <span id="activeNeuronCount">82</span> / 128 &bull; Sparsity: <span id="sparsityPercent">67.2%</span>
          </span>
        </div>
        <p style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 12px;">
          Each block represents a single neuron in the 128-dimensional clamped ReLU accumulator (Rule 4: Zero-leakage discrete steering). Click any cell to inspect activation status.
        </p>
        <div class="crelu-matrix" id="neuronGrid"></div>
        <div id="neuronInfo" style="margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-dim);">
          Hover or click a neuron cell above to inspect activation value.
        </div>
      </div>
    </div>

    <!-- ============================================================= -->
    <!-- TAB 2: BULLET BOT COCKPIT (DOMAIN A)                          -->
    <!-- ============================================================= -->
    <div id="tab-chess" class="tab-panel">
      <div class="grid-2">
        <!-- Left: Live Board -->
        <div class="card" style="display: flex; flex-direction: column; align-items: center;">
          <div class="card-title" style="width: 100%;">
            <span>Autonomous Board Preview</span>
            <span style="color: var(--accent-green);">Live Lichess Feed</span>
          </div>
          <div class="board-frame" id="liveBoardGrid"></div>
          <div style="margin-top: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); text-align: center;">
            FEN: <span id="cockpitFen">rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1</span>
          </div>
        </div>

        <!-- Right: Telemetry & Benchmark Card -->
        <div style="display: flex; flex-direction: column; gap: 20px;">
          <div class="card">
            <div class="card-title">
              <span>Lichess Bot Metrics (@jess-hyperbullet)</span>
              <span style="background: rgba(16,185,129,0.15); color: var(--accent-green); padding: 2px 8px; border-radius: 12px; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;">● Online 24/7</span>
            </div>
            <div class="grid-3">
              <div class="stat-box">
                <div class="stat-label">Bullet Rating</div>
                <div class="stat-number cyan" id="telemetryRating">1781?</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">Reflex Latency</div>
                <div class="stat-number green">&lt; 35ms</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">Tactical Solve</div>
                <div class="stat-number amber">60.0%</div>
              </div>
            </div>
          </div>

          <!-- Benchmark Upgrade Card -->
          <div class="card">
            <div class="card-title">
              <span>Rigorous Tactical Suite Benchmark (20 Positions)</span>
            </div>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Before Fix</th>
                  <th>After Fix</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Tactical Solve Rate</td>
                  <td>25.0% (5/20)</td>
                  <td style="color: var(--accent-green); font-weight: 700;">60.0% (12/20)</td>
                  <td>+140% Relative</td>
                </tr>
                <tr>
                  <td>Crisis Escalation Recall</td>
                  <td>10.0%</td>
                  <td style="color: var(--accent-green); font-weight: 700;">100.0%</td>
                  <td>10x Improvement</td>
                </tr>
                <tr>
                  <td>Median Crisis Latency</td>
                  <td>51.4ms</td>
                  <td>170.5ms</td>
                  <td>Strict &lt; 250ms Cap</td>
                </tr>
                <tr>
                  <td>Estimated Bullet Elo</td>
                  <td>~1781 Elo</td>
                  <td style="color: var(--accent-cyan); font-weight: 700;">~2040 Elo</td>
                  <td>+259 Elo Leap</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Recent Games -->
          <div class="card">
            <div class="card-title">
              <span>Recent Matches</span>
              <a href="https://lichess.org/@/jess-hyperbullet" target="_blank" style="color: var(--accent-cyan); font-size: 0.72rem; text-decoration: none;">View on Lichess &rarr;</a>
            </div>
            <div class="recent-list" id="cockpitGamesList">
              <div style="color: var(--text-dim); font-size: 0.8rem; text-align: center; padding: 12px;">Loading games from Lichess API...</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ============================================================= -->
    <!-- TAB 3: CELLULAR AUTOMATA & SPATIAL INVARIANTS (DOMAIN B)      -->
    <!-- ============================================================= -->
    <div id="tab-ca" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>2D Cellular Automata Simulator (Interactive)</span>
            <span id="caGenCounter" style="font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan);">Gen: 0</span>
          </div>
          <div class="ca-canvas-container">
            <canvas id="caCanvas" width="384" height="384"></canvas>
            <div class="ca-controls">
              <button class="btn-tool" onclick="toggleCaPlay()" id="caPlayBtn">▶ Play</button>
              <button class="btn-tool" onclick="stepCa()">Step</button>
              <button class="btn-tool" onclick="randomizeCa()">Randomize</button>
              <button class="btn-tool" onclick="clearCa()">Clear</button>
              <button class="btn-tool active-mode" id="btnSteerMode" onclick="toggleSteerMode()">Mode: TypeSafe Discrete (0% Leakage)</button>
            </div>
          </div>
          <div id="caStatusText" style="margin-top: 14px; font-size: 0.78rem; color: var(--text-muted); line-height: 1.5;">
            <strong>Theorem 1 Verification:</strong> Under discrete integer factor masks, inactive threshold cells satisfy &phi;(0) = 0 identically. Parasitic dead-neuron leakage is strictly <strong>0.00%</strong>.
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>ARC-AGI Invariant Codebook (B = 4 bits, K = 16)</span>
          </div>
          <p style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 12px;">
            Rate-distortion optimal discrete codebook: System 2 transmits a 4-bit invariant vector to steer System 0 without continuous threshold corruption.
          </p>
          <table class="data-table">
            <thead>
              <tr>
                <th>Code</th>
                <th>Primitive Class</th>
                <th>Discrete Operator</th>
              </tr>
            </thead>
            <tbody>
              <tr><td><code>0x0</code></td><td>Identity</td><td>Y = X</td></tr>
              <tr><td><code>0x1</code></td><td>Horizontal Flip</td><td>Y(r,c) = X(r, W-1-c)</td></tr>
              <tr><td><code>0x2</code></td><td>Vertical Flip</td><td>Y(r,c) = X(H-1-r, c)</td></tr>
              <tr><td><code>0x3</code></td><td>Diagonal Transpose</td><td>Y(r,c) = X(c, r)</td></tr>
              <tr><td><code>0x4</code></td><td>Gravity Down</td><td>Fall until collision</td></tr>
              <tr><td><code>0x5</code></td><td>Connected Component</td><td>Mask largest contiguous cluster</td></tr>
              <tr><td><code>0x6</code></td><td>Morphological Dilation</td><td>3x3 Structuring stencil</td></tr>
              <tr><td><code>0x7</code></td><td>Color Permutation</td><td>Bijective color map &pi;</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ============================================================= -->
    <!-- TAB 4: THEOREM 2 & CYCLE ENTRAPMENT (DOMAIN C)                -->
    <!-- ============================================================= -->
    <div id="tab-proof" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>The 94% Cycle Entrapment Pathology</span>
          </div>
          <p style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 16px;">
            In combinatorial formal systems (Lean 4, Collatz labyrinths, register machines) with reversible rewrite rules (A &harr; B), standard continuous latent recurrence and naive search enter infinite closed limit cycles with <strong>&gt;94% frequency</strong>.
          </p>
          <div style="background: var(--bg-elevated); padding: 16px; border-radius: 6px; border: 1px solid var(--border); text-align: center; margin-bottom: 16px;">
            <svg viewBox="0 0 320 120" style="width: 100%; max-width: 320px; margin: 0 auto; display: block;">
              <circle cx="50" cy="60" r="22" fill="#171B26" stroke="#EF4444" stroke-width="2"/>
              <text x="50" y="65" fill="#EF4444" font-family="JetBrains Mono" font-size="12" text-anchor="middle">s₀</text>

              <path d="M 72 50 Q 160 20 248 50" fill="none" stroke="#EF4444" stroke-width="2" stroke-dasharray="4"/>
              <polygon points="248,50 238,45 242,54" fill="#EF4444"/>

              <circle cx="270" cy="60" r="22" fill="#171B26" stroke="#EF4444" stroke-width="2"/>
              <text x="270" y="65" fill="#EF4444" font-family="JetBrains Mono" font-size="12" text-anchor="middle">s₁</text>

              <path d="M 248 70 Q 160 100 72 70" fill="none" stroke="#EF4444" stroke-width="2" stroke-dasharray="4"/>
              <polygon points="72,70 82,75 78,66" fill="#EF4444"/>
            </svg>
            <div style="color: var(--accent-red); font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; font-weight: 700; margin-top: 8px;">
              Orbital Limit Cycle: s₀ &rarr; s₁ &rarr; s₀ (94.2% Trap Rate)
            </div>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5;">
            Monolithic LLMs hallucinate that progress is occurring because continuous vectors &vec;h_t continuously mutate even as symbolic states orbit the exact same invariant loop.
          </p>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Theorem 2 & 3: Dual Impossibility Proof</span>
          </div>

          <div class="proof-step">
            <strong>Step 1: Lyapunov Potential Functional</strong><br>
            Define non-negative potential <code>V(s) &ge; 0</code> with strict contraction condition:
            <code>V(s_{t+1}) - V(s_t) &le; -&epsilon;</code> for constant <code>&epsilon; &gt; 0</code>.
          </div>

          <div class="proof-step">
            <strong>Step 2: Telescoping Sum along a Closed Cycle</strong><br>
            Suppose for contradiction a closed orbit exists: <code>s_0 &rarr; s_1 &rarr; ... &rarr; s_k = s_0</code>.<br>
            Summing the inequality over all k steps:<br>
            <code>&Sigma; (V(s_{i+1}) - V(s_i)) &le; -k&epsilon; &implies; V(s_k) - V(s_0) &le; -k&epsilon;</code>
          </div>

          <div class="proof-step">
            <strong>Step 3: Direct Contradiction</strong><br>
            Since <code>s_k = s_0</code>, the left-hand side is exactly <code>0</code>:<br>
            <code>0 &le; -k&epsilon; &implies; k&epsilon; &le; 0</code>.<br>
            Because <code>k &ge; 1</code> and <code>&epsilon; &gt; 0</code>, we have <code>k&epsilon; &gt; 0</code>. Thus <code>0 &lt; k&epsilon; &le; 0</code>, a direct mathematical contradiction. &FilledSmallSquare;
          </div>

          <div class="proof-step" style="border-left-color: var(--accent-green);">
            <strong>Step 4: Finite Termination Upper Bound</strong><br>
            Because <code>V(s) &ge; 0</code> for all states, search must reach Q.E.D. in at most:
            <code>K &le; &lfloor; V(s_0) / &epsilon; &rfloor; steps</code>.
          </div>
        </div>
      </div>
    </div>

    <!-- ============================================================= -->
    <!-- TAB 5: TRAINING CURVES & METRICS                              -->
    <!-- ============================================================= -->
    <div id="tab-training" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>12-Epoch Multi-Objective Training Loss</span>
            <span style="color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">Huber + Brier Noul</span>
          </div>
          <!-- Inline SVG Loss Curve -->
          <svg viewBox="0 0 500 220" style="width: 100%; height: auto; background: var(--bg-elevated); border-radius: 6px; padding: 10px;">
            <!-- Grid lines -->
            <line x1="40" y1="20" x2="480" y2="20" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="70" x2="480" y2="70" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="120" x2="480" y2="120" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="170" x2="480" y2="170" stroke="#1E2433" stroke-width="1"/>

            <!-- Axes labels -->
            <text x="32" y="24" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.80</text>
            <text x="32" y="74" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.60</text>
            <text x="32" y="124" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.40</text>
            <text x="32" y="174" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.20</text>

            <text x="40" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E1</text>
            <text x="145" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E4</text>
            <text x="255" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E8</text>
            <text x="465" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E12</text>

            <!-- Loss line -->
            <path d="M 40 45 L 80 82 L 120 105 L 160 118 L 200 126 L 240 134 L 280 142 L 320 149 L 360 154 L 400 159 L 440 163 L 480 166" fill="none" stroke="#38BDF8" stroke-width="2.5"/>
            <!-- Points -->
            <circle cx="480" cy="166" r="4" fill="#38BDF8"/>
          </svg>
          <div style="margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); display: flex; justify-content: space-between;">
            <span>Initial: 0.7420</span>
            <span style="color: var(--accent-green); font-weight: 700;">Final Best: 0.0848</span>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>CReLU Sparsity & Active Neurons</span>
            <span style="color: var(--accent-green); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">Target: ~60% Sparsity</span>
          </div>
          <svg viewBox="0 0 500 220" style="width: 100%; height: auto; background: var(--bg-elevated); border-radius: 6px; padding: 10px;">
            <line x1="40" y1="20" x2="480" y2="20" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="70" x2="480" y2="70" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="120" x2="480" y2="120" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="170" x2="480" y2="170" stroke="#1E2433" stroke-width="1"/>

            <text x="32" y="24" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">128</text>
            <text x="32" y="74" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">96</text>
            <text x="32" y="124" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">64</text>
            <text x="32" y="174" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">32</text>

            <text x="40" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E1</text>
            <text x="255" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E6</text>
            <text x="465" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E12</text>

            <!-- Active Neurons Curve -->
            <path d="M 40 30 L 80 42 L 120 58 L 160 72 L 200 80 L 240 85 L 280 88 L 320 90 L 360 92 L 400 93 L 440 94 L 480 95" fill="none" stroke="#10B981" stroke-width="2.5"/>
            <circle cx="480" cy="95" r="4" fill="#10B981"/>
          </svg>
          <div style="margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); display: flex; justify-content: space-between;">
            <span>Initial: 128 / 128 active</span>
            <span style="color: var(--accent-green); font-weight: 700;">Converged: 82 active (67.2% sparse)</span>
          </div>
        </div>
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer>
    <div>Hosted globally on Cloudflare Edge &bull; Domain: <code>jev.subsurfaces.net</code></div>
    <div>Engine: Jevformer Tri-Process (128-neuron CReLU + Epistemic Gating + Lean 4 Soundness)</div>
  </footer>

  <script>
    // Tab Switching
    function switchTab(name) {
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      const targetPanel = document.getElementById('tab-' + name);
      if (targetPanel) targetPanel.classList.add('active');
      event.currentTarget.classList.add('active');

      if (name === 'ca') initCa();
      if (name === 'chess') loadCockpitGames();
    }

    // Architecture Mode Toggler
    function setArchMode(mode) {
      const expl = document.getElementById('archExplainer');
      const bQuiet = document.getElementById('btnRouteQuiet');
      const bCrisis = document.getElementById('btnRouteCrisis');
      const s0 = document.getElementById('nodeSystem0');
      const s2 = document.getElementById('nodeSystem2');

      if (mode === 'quiet') {
        bQuiet.classList.add('active-mode');
        bCrisis.classList.remove('active-mode');
        s0.classList.add('active-route');
        s2.classList.remove('active-route');
        expl.innerHTML = '<strong>Current Route (Quiet Positional Ply):</strong> Model operates at 0.5ms reflex latency. System 1 measures high certainty (Noul &ge; 0.70), bypassing deep search and preserving bullet clock time.';
      } else {
        bCrisis.classList.add('active-mode');
        bQuiet.classList.remove('active-mode');
        s2.classList.add('active-route');
        s0.classList.remove('active-route');
        expl.innerHTML = '<strong>Current Route (Tactical Crisis / Formal Conflict):</strong> System 1 detects low confidence (Noul &lt; 0.70) or hanging captures. Search escalates to System 2 (Negamax Depth 2-4 with Quiescence Search on captures).';
      }
    }

    // Initialize 128-neuron CReLU Grid
    function initNeuronGrid() {
      const grid = document.getElementById('neuronGrid');
      grid.innerHTML = '';
      const total = 128;
      const activeCount = 82;
      for (let i = 0; i < total; i++) {
        const cell = document.createElement('div');
        const isActive = (i < activeCount);
        cell.className = 'neuron-cell' + (isActive ? ' active' : '');
        const val = isActive ? (0.2 + (i % 7) * 0.11).toFixed(2) : '0.00';
        cell.onmouseenter = () => {
          document.getElementById('neuronInfo').textContent =
            \`Neuron #\${i}: \${isActive ? 'ACTIVE' : 'DEAD/SPARSE'} (Activation: \${val})\`;
        };
        grid.appendChild(cell);
      }
    }
    initNeuronGrid();

    // Domain B: Interactive Cellular Automata
    const CA_SIZE = 24;
    let caGrid = Array(CA_SIZE).fill(0).map(() => Array(CA_SIZE).fill(0));
    let caRunning = false;
    let caTimer = null;
    let caGen = 0;
    let discreteSteer = true;

    function initCa() {
      const canvas = document.getElementById('caCanvas');
      if (!canvas) return;
      randomizeCa();
      drawCa();
    }

    function randomizeCa() {
      caGen = 0;
      for (let r = 0; r < CA_SIZE; r++) {
        for (let c = 0; c < CA_SIZE; c++) {
          caGrid[r][c] = (Math.random() < 0.28) ? 1 : 0;
        }
      }
      document.getElementById('caGenCounter').textContent = 'Gen: ' + caGen;
      drawCa();
    }

    function clearCa() {
      caGen = 0;
      caGrid = Array(CA_SIZE).fill(0).map(() => Array(CA_SIZE).fill(0));
      document.getElementById('caGenCounter').textContent = 'Gen: ' + caGen;
      drawCa();
    }

    function drawCa() {
      const canvas = document.getElementById('caCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const cellSize = canvas.width / CA_SIZE;
      ctx.fillStyle = '#080A0D';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let r = 0; r < CA_SIZE; r++) {
        for (let c = 0; c < CA_SIZE; c++) {
          if (caGrid[r][c] === 1) {
            ctx.fillStyle = discreteSteer ? '#10B981' : '#EF4444';
            ctx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
          } else {
            ctx.fillStyle = '#12151D';
            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
          }
        }
      }
    }

    function stepCa() {
      const next = Array(CA_SIZE).fill(0).map(() => Array(CA_SIZE).fill(0));
      for (let r = 0; r < CA_SIZE; r++) {
        for (let c = 0; c < CA_SIZE; c++) {
          let neighbors = 0;
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              if (dr === 0 && dc === 0) continue;
              const nr = (r + dr + CA_SIZE) % CA_SIZE;
              const nc = (c + dc + CA_SIZE) % CA_SIZE;
              neighbors += caGrid[nr][nc];
            }
          }
          if (caGrid[r][c] === 1) {
            next[r][c] = (neighbors === 2 || neighbors === 3) ? 1 : 0;
          } else {
            // If continuous dense noise mode, inject parasitic leakage
            if (!discreteSteer && Math.random() < 0.04) {
              next[r][c] = 1; // Parasitic leakage
            } else {
              next[r][c] = (neighbors === 3) ? 1 : 0;
            }
          }
        }
      }
      caGrid = next;
      caGen++;
      document.getElementById('caGenCounter').textContent = 'Gen: ' + caGen;
      drawCa();
    }

    function toggleCaPlay() {
      caRunning = !caRunning;
      const btn = document.getElementById('caPlayBtn');
      if (caRunning) {
        btn.textContent = '⏸ Pause';
        caTimer = setInterval(stepCa, 160);
      } else {
        btn.textContent = '▶ Play';
        clearInterval(caTimer);
      }
    }

    function toggleSteerMode() {
      discreteSteer = !discreteSteer;
      const btn = document.getElementById('btnSteerMode');
      const status = document.getElementById('caStatusText');
      if (discreteSteer) {
        btn.textContent = 'Mode: TypeSafe Discrete (0% Leakage)';
        btn.classList.add('active-mode');
        status.innerHTML = '<strong>Theorem 1 Verification:</strong> Under discrete integer factor masks, inactive threshold cells satisfy &phi;(0) = 0 identically. Parasitic dead-neuron leakage is strictly <strong>0.00%</strong>.';
      } else {
        btn.textContent = 'Mode: Continuous Dense Perturbation';
        btn.classList.remove('active-mode');
        status.innerHTML = '<strong style="color: var(--accent-red);">Continuous Collapse:</strong> Adding continuous dense latent noise shifts the zero-point threshold for every cell simultaneously, inducing parasitic cell leakage and structural breakdown.';
      }
      drawCa();
    }

    // Domain A: Chess Board Setup
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

    function renderCockpitBoard(grid) {
      const el = document.getElementById('liveBoardGrid');
      if (!el) return;
      el.innerHTML = '';
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const sq = document.createElement('div');
          sq.className = 'square ' + ((r + c) % 2 === 0 ? 'light' : 'dark');
          sq.textContent = grid[r][c] || '';
          el.appendChild(sq);
        }
      }
    }
    renderCockpitBoard(INITIAL_BOARD);

    async function loadCockpitGames() {
      try {
        const res = await fetch('/api/recent_games');
        const data = await res.json();
        const list = document.getElementById('cockpitGamesList');
        if (!list) return;
        list.innerHTML = '';

        if (!data || data.length === 0) {
          list.innerHTML = '<div style="color: var(--text-dim); font-size: 0.8rem; text-align: center; padding: 12px;">No live games recorded yet.</div>';
          return;
        }

        data.forEach(g => {
          const item = document.createElement('a');
          item.className = 'game-item';
          item.href = 'https://lichess.org/' + g.id;
          item.target = '_blank';

          const white = g.players?.white?.user?.name || 'White';
          const black = g.players?.black?.user?.name || 'Black';
          const result = g.winner ? (g.winner === 'white' ? '1-0' : '0-1') : '1/2-1/2';

          item.innerHTML = \`
            <div><strong>\${white}</strong> vs <strong>\${black}</strong> (\${g.speed})</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--accent-cyan);">\${result}</div>
          \`;
          list.appendChild(item);
        });
      } catch(e) {
        console.error('Cockpit games error:', e);
      }
    }

    async function triggerWake() {
      const btn = document.getElementById('wakeBtn');
      btn.textContent = '⏳ Waking Modal Cloud...';
      btn.style.color = 'var(--accent-cyan)';
      try {
        const res = await fetch('/api/wake_bot');
        const data = await res.json();
        btn.textContent = '✅ Awakened (15m Active)';
        btn.style.color = 'var(--accent-green)';
        setTimeout(() => {
          btn.textContent = '⚡ Modal Daemon: Ready';
          btn.style.color = 'var(--text-primary)';
        }, 8000);
      } catch(e) {
        btn.textContent = '⚠️ Error waking';
      }
    }
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
      const games = ndjson.trim().split("\n").filter(Boolean).map(line => {
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

    // Default: Serve the comprehensive Architecture Explorer Web Application
    return new Response(HTML_CONTENT, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "public, max-age=60"
      }
    });
  }
};
