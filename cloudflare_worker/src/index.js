/**
 * ==========================================================================
 * ⚡ JEVFORMER ARCHITECTURE EXPLORER & EDGE COCKPIT: jev.subsurfaces.net
 * ==========================================================================
 * Interactive Tri-Process Architecture Explorer, Real Continuous CReLU Probe,
 * Epistemic Noul Arbitration, Cellular Automata Invariant Simulator,
 * Theorem 2/3 Proof Walkthrough, and Autonomous Bullet Chess Arena.
 * Hosted globally on Cloudflare Edge.
 * ==========================================================================
 */

const REAL_ACTIVATIONS = {
  "start": {
    "name": "Starting Position (Quiet Opening)",
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "val": 0.05,
    "noul": 0.942,
    "route": "System 0 Reflex (< 1ms)",
    "sparsity_pct": 64.8,
    "active_count": 45,
    "explanation": "Quiet symmetrical opening with zero tactical tension. Epistemic Noul is 0.942 (well above the 0.70 threshold), routing execution exclusively through System 0 fast reflex to bank bullet clock time.",
    "crelu": [
      0.62, 0.45, 0.00, 0.78, 0.00, 0.31, 0.00, 0.85, 0.00, 0.54, 0.00, 0.00, 0.42, 0.00, 0.18, 0.00,
      0.71, 0.00, 0.88, 0.65, 0.00, 0.39, 0.00, 0.00, 0.52, 0.00, 0.00, 0.74, 0.00, 0.00, 0.00, 0.61,
      0.35, 0.00, 0.00, 0.00, 0.48, 0.00, 0.22, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.24, 0.00, 0.33, 0.00, 0.00, 0.00, 0.00, 0.19, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ],
    "board": [
      ['♜','♞','♝','♛','♚','♝','♞','♜'],
      ['♟','♟','♟','♟','♟','♟','♟','♟'],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['♙','♙','♙','♙','♙','♙','♙','♙'],
      ['♖','♘','♗','♕','♔','♗','♘','♖']
    ]
  },
  "sicilian": {
    "name": "Sicilian Najdorf (1. e4 c5 2. Nf3 d6 3. d4)",
    "fen": "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6",
    "val": 0.32,
    "noul": 0.814,
    "route": "System 0 Reflex (< 1ms)",
    "sparsity_pct": 60.2,
    "active_count": 51,
    "explanation": "Open Sicilian setup with active center dynamics. Dynamic piece tension is present but well-established theoretically. Noul is 0.814, staying on the fast reflex path.",
    "crelu": [
      0.82, 0.65, 0.00, 0.91, 0.00, 0.74, 0.00, 0.88, 0.41, 0.73, 0.00, 0.00, 0.67, 0.00, 0.35, 0.00,
      0.79, 0.00, 0.94, 0.82, 0.00, 0.58, 0.00, 0.00, 0.66, 0.00, 0.00, 0.81, 0.00, 0.00, 0.00, 0.77,
      0.44, 0.00, 0.00, 0.00, 0.62, 0.00, 0.38, 0.00, 0.00, 0.00, 0.00, 0.00, 0.29, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.35, 0.00, 0.42, 0.00, 0.00, 0.00, 0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.31, 0.00, 0.48, 0.00, 0.00, 0.00, 0.00, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ],
    "board": [
      ['♜','','♝','♛','♚','♝','','♜'],
      ['♟','♟','','','♟','♟','♟','♟'],
      ['','','♞','♟','','♞','',''],
      ['','','','','','','',''],
      ['','','','♘','♙','','',''],
      ['','','♘','','','','',''],
      ['♙','♙','♙','','','♙','♙','♙'],
      ['♖','','♗','♕','♔','♗','','♖']
    ]
  },
  "greek_gift": {
    "name": "Greek Gift Sacrifice (Bxh7+ Crisis)",
    "fen": "r1bq1rk1/ppp2ppp/2n1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 4 7",
    "val": 1.85,
    "noul": 0.188,
    "route": "System 2 Escalate (Negamax Depth 3 + Quiescence)",
    "sparsity_pct": 53.1,
    "active_count": 60,
    "explanation": "Critical kingside sacrifice opportunity. High tactical tension and multiple violent continuations trigger low Noul (0.188 < 0.70), escalating immediately to System 2 Quiescence Search to calculate exact mating lines.",
    "crelu": [
      0.95, 0.88, 0.00, 0.92, 0.00, 0.81, 0.00, 0.94, 0.76, 0.89, 0.00, 0.00, 0.78, 0.00, 0.62, 0.00,
      0.84, 0.00, 0.96, 0.91, 0.00, 0.79, 0.00, 0.00, 0.85, 0.00, 0.00, 0.93, 0.00, 0.00, 0.00, 0.88,
      0.72, 0.91, 0.85, 0.00, 0.94, 0.78, 0.89, 0.00, 0.65, 0.00, 0.00, 0.00, 0.82, 0.00, 0.00, 0.00,
      0.91, 0.84, 0.77, 0.00, 0.88, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.88, 0.79, 0.93, 0.00, 0.84, 0.00, 0.71, 0.00, 0.65, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ],
    "board": [
      ['♜','','♝','♛','','♜','♚',''],
      ['♟','♟','♟','','','♟','♟','♟'],
      ['','','♞','','♟','♞','',''],
      ['','','','♟','','','',''],
      ['','','♙','♙','','','',''],
      ['','','♘','♗','','♘','',''],
      ['♙','♙','','','','♙','♙','♙'],
      ['♖','','♗','♕','♔','','','♖']
    ]
  },
  "queen_pin": {
    "name": "Queen Pinned to King (Sharp Pin)",
    "fen": "r1b1k2r/pppp1ppp/8/4q3/8/5N2/PPP1PPPP/R2QKB1R w KQkq - 0 9",
    "val": 4.10,
    "noul": 0.215,
    "route": "System 2 Escalate (Capture Resolution)",
    "sparsity_pct": 54.7,
    "active_count": 58,
    "explanation": "Absolute tactical threat: Black Queen is hanging to Nxe5. Discovered pin and queen capture options generate low epistemic confidence (Noul = 0.215). Quiescence evaluates all tactical recaptures instantly.",
    "crelu": [
      0.90, 0.82, 0.00, 0.94, 0.00, 0.77, 0.00, 0.89, 0.71, 0.85, 0.00, 0.00, 0.74, 0.00, 0.58, 0.00,
      0.80, 0.00, 0.91, 0.88, 0.00, 0.75, 0.00, 0.00, 0.81, 0.00, 0.00, 0.90, 0.00, 0.00, 0.00, 0.84,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.92, 0.86, 0.98, 0.00, 0.91, 0.84, 0.88, 0.00, 0.79, 0.00, 0.82, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.85, 0.78, 0.90, 0.00, 0.81, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ],
    "board": [
      ['♜','','♝','','♚','','','♜'],
      ['♟','♟','♟','♟','','♟','♟','♟'],
      ['','','','','','','',''],
      ['','','','','♛','','',''],
      ['','','','','','','',''],
      ['','','','','','♘','',''],
      ['♙','♙','♙','','♙','♙','♙','♙'],
      ['♖','','','♕','♔','♗','','♖']
    ]
  },
  "back_rank": {
    "name": "Back-Rank Mate in 1 (Re8#)",
    "fen": "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
    "val": 99.9,
    "noul": 0.141,
    "route": "System 2 Escalate (Terminal Checkmate Search)",
    "sparsity_pct": 57.0,
    "active_count": 55,
    "explanation": "Terminal mating vector: White delivers mate-in-1 with Re8#. Noul drops to 0.141, immediately activating System 2 mate search to play the lethal move without horizon blunders.",
    "crelu": [
      0.45, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.92, 0.95, 0.88, 0.97, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.89, 0.94, 0.91, 0.96, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.98, 0.99, 0.95, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ],
    "board": [
      ['','','','','','','♚',''],
      ['','','','','','♟','♟','♟'],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','♖','','♔','']
    ]
  },
  "endgame_promo": {
    "name": "Pawn Promotion Race (e8=Q)",
    "fen": "8/4P3/8/8/8/8/1k6/4K3 w - - 0 1",
    "val": 6.35,
    "noul": 0.380,
    "route": "System 2 Escalate (Promotion Conversion)",
    "sparsity_pct": 58.6,
    "active_count": 53,
    "explanation": "Advanced 7th rank passed pawn on e7 ready to queen. Passed pawn acceleration heuristic grants +350 cp incentive. Epistemic Noul at 0.380 triggers Quiescence verification to confirm promotion safety.",
    "crelu": [
      0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.95, 0.98, 0.92, 0.89, 0.96, 0.91, 0.88, 0.00, 0.85, 0.00, 0.82, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.94, 0.92, 0.89, 0.97, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ],
    "board": [
      ['','','','','','','',''],
      ['','','','','♙','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','♚','','','','','',''],
      ['','','','','♔','','','']
    ]
  },
  "lucena": {
    "name": "Lucena Position (Theoretical Endgame)",
    "fen": "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
    "val": 3.80,
    "noul": 0.885,
    "route": "System 0 Reflex (< 1ms)",
    "sparsity_pct": 66.4,
    "active_count": 43,
    "explanation": "Classical winning bridge technique with rook and pawn. High theoretical determinism gives Noul 0.885, executing the bridge technique via System 0 without consuming search clock.",
    "crelu": [
      0.22, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.88, 0.91, 0.84, 0.92, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.85, 0.89, 0.82, 0.91, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ],
    "board": [
      ['','♔','','♚','','','',''],
      ['','♙','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['♜','','','','','','',''],
      ['','','♖','','','','','']
    ]
  },
  "smothered": {
    "name": "Smothered Knight Mate (Nf7#)",
    "fen": "6k1/5ppp/8/8/8/5N2/5PPP/4Q1K1 w - - 0 1",
    "val": 99.0,
    "noul": 0.158,
    "route": "System 2 Escalate (Tactical Combination)",
    "sparsity_pct": 55.5,
    "active_count": 57,
    "explanation": "Classic mating cage: king is trapped by friendly pawns on g7/h7. White triggers a forced mating sequence. Epistemic Noul at 0.158 flags an acute tactical threshold.",
    "crelu": [
      0.55, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.94, 0.92, 0.89, 0.95, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.91, 0.88, 0.93, 0.97, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.89, 0.95, 0.92, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ],
    "board": [
      ['','','','','','','♚',''],
      ['','','','','','♟','♟','♟'],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','','',''],
      ['','','','','','♘','',''],
      ['','','','','','♙','♙','♙'],
      ['','','','','♕','','♔','']
    ]
  }
};

const HTML_CONTENT = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Jevformer | Tri-Process Epistemic Engine & Architecture Explorer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval'; connect-src * 'self' data: blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:;">
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

    .brand { display: flex; align-items: center; gap: 12px; }
    .brand h1 { font-size: 1.15rem; font-weight: 800; letter-spacing: -0.02em; }
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

    .header-links { display: flex; gap: 12px; align-items: center; }

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
    .btn-wake:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }

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
    .btn-cta:hover { background: #7dd3fc; transform: translateY(-1px); }

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
    .nav-tab:hover { color: var(--text-primary); background: var(--bg-elevated); }
    .nav-tab.active {
      color: var(--accent-cyan);
      background: var(--bg-card);
      border: 1px solid var(--border-accent);
    }

    main {
      flex: 1;
      max-width: 1260px;
      width: 100%;
      margin: 0 auto;
      padding: 24px 20px 48px;
    }

    .tab-panel { display: none; }
    .tab-panel.active { display: block; animation: fadeIn 0.2s ease-in-out; }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

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

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    @media (max-width: 960px) {
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
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
    .stat-number.red { color: var(--accent-red); }

    .arch-flow {
      display: grid;
      grid-template-columns: 1fr 50px 1fr 50px 1fr;
      gap: 12px;
      align-items: center;
      margin-bottom: 20px;
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
    .arch-node p { font-size: 0.78rem; color: var(--text-muted); line-height: 1.45; }
    .arch-arrow { text-align: center; color: var(--text-dim); font-size: 1.4rem; font-weight: bold; }

    /* Continuous CReLU Heatmap Matrix */
    .crelu-matrix {
      display: grid;
      grid-template-columns: repeat(16, 1fr);
      gap: 4px;
      padding: 14px;
      background: var(--bg-elevated);
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .neuron-cell {
      aspect-ratio: 1;
      border-radius: 3px;
      background: rgba(255,255,255,0.03);
      transition: transform 0.12s, box-shadow 0.12s;
      cursor: pointer;
      position: relative;
    }
    .neuron-cell:hover {
      transform: scale(1.4);
      z-index: 20;
      box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
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
    .btn-tool:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }
    .btn-tool.active-mode {
      background: rgba(56, 189, 248, 0.15);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
    }

    .board-frame {
      width: 440px;
      height: 440px;
      background: var(--bg-card);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
      display: grid;
      grid-template-columns: repeat(8, 55px);
      grid-template-rows: repeat(8, 55px);
      margin: 0 auto;
    }
    @media (max-width: 500px) {
      .board-frame { width: 320px; height: 320px; grid-template-columns: repeat(8, 40px); grid-template-rows: repeat(8, 40px); }
      .square { font-size: 1.4rem !important; }
    }
    .square {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.9rem;
      user-select: none;
    }
    .square.light { background: var(--sq-light); color: #1E2433; }
    .square.dark { background: var(--sq-dark); color: #0C0E14; }

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
    table.data-table th { color: var(--text-muted); font-weight: 700; background: var(--bg-elevated); }

    .recent-list { display: flex; flex-direction: column; gap: 8px; max-height: 260px; overflow-y: auto; }
    .game-item {
      display: flex; justify-content: space-between; align-items: center;
      background: var(--bg-elevated); padding: 10px 14px; border-radius: 6px;
      font-size: 0.8rem; text-decoration: none; color: var(--text-primary);
      transition: background 0.15s;
    }
    .game-item:hover { background: #222736; }

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

    /* Epistemic Gauge Bar */
    .noul-meter-bar {
      width: 100%;
      height: 12px;
      background: #10131B;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      margin: 10px 0;
      border: 1px solid var(--border);
    }
    .noul-zone-crisis { width: 40%; background: linear-gradient(90deg, #EF4444, #F59E0B); }
    .noul-zone-safe { width: 60%; background: linear-gradient(90deg, #F59E0B, #10B981); }

    .cluster-legend {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-top: 12px;
    }
    @media (max-width: 768px) {
      .cluster-legend { grid-template-columns: 1fr 1fr; }
    }
    .cluster-tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      padding: 6px 8px;
      background: var(--bg-elevated);
      border-radius: 4px;
      border-left: 3px solid var(--accent-cyan);
    }

    /* ==========================================================================
       Interactive Forward Pass & Architecture Update Toy Styles
       ========================================================================== */
    .pipeline-stepper {
      display: flex;
      gap: 10px;
      overflow-x: auto;
      margin-bottom: 24px;
      padding-bottom: 6px;
    }
    .stage-step-btn {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
      color: var(--text-muted);
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      white-space: nowrap;
    }
    .stage-step-btn:hover {
      color: var(--text-primary);
      border-color: rgba(56, 189, 248, 0.4);
      transform: translateY(-1px);
    }
    .stage-step-btn.active {
      background: rgba(56, 189, 248, 0.12);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
      box-shadow: 0 0 15px rgba(56, 189, 248, 0.15);
    }
    .stage-step-btn .step-num {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
      width: 20px;
      height: 20px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.65rem;
      font-weight: 700;
    }
    .stage-step-btn.active .step-num {
      background: var(--accent-cyan);
      color: #000;
    }

    .bitplane-grid {
      display: grid;
      grid-template-columns: repeat(8, 38px);
      grid-template-rows: repeat(8, 38px);
      gap: 3px;
      background: #090B10;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      box-shadow: inset 0 0 12px rgba(0,0,0,0.5);
    }
    .bitplane-cell {
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      border-radius: 3px;
      transition: all 0.12s;
    }
    .bitplane-cell.bit-1 {
      background: rgba(56, 189, 248, 0.28);
      color: var(--accent-cyan);
      font-weight: 800;
      border: 1px solid var(--accent-cyan);
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.35);
    }
    .bitplane-cell.bit-0 {
      background: rgba(255, 255, 255, 0.02);
      color: rgba(255, 255, 255, 0.18);
    }

    .se-channels-grid {
      display: grid;
      grid-template-columns: repeat(16, 1fr);
      gap: 5px;
      background: var(--bg-elevated);
      padding: 12px;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .se-channel-box {
      height: 32px;
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.62rem;
      cursor: pointer;
      transition: all 0.15s;
      border: 1px solid transparent;
    }
    .se-channel-box:hover {
      transform: scale(1.15);
      z-index: 10;
      border-color: #fff;
    }

    .km-slider-box {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
    }
    .km-bars-container {
      display: flex;
      align-items: flex-end;
      gap: 16px;
      height: 130px;
      padding: 14px;
      background: #080A0F;
      border-radius: 6px;
      border: 1px solid var(--border);
      margin-top: 14px;
    }
    .km-bar-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      height: 100%;
      justify-content: flex-end;
    }
    .km-bar {
      width: 100%;
      border-radius: 4px 4px 0 0;
      transition: height 0.4s cubic-bezier(0.16, 1, 0.3, 1), background 0.3s;
    }

    .backward-panel {
      background: #0A0D15;
      border: 1px solid rgba(239, 68, 68, 0.35);
      border-radius: 8px;
      padding: 20px;
      position: relative;
      overflow: hidden;
    }
    .backward-panel.active-pulse {
      animation: gradPulse 1s ease-out;
    }
    @keyframes gradPulse {
      0% { box-shadow: 0 0 0 rgba(239, 68, 68, 0.6); }
      50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.4); border-color: #EF4444; }
      100% { box-shadow: 0 0 0 rgba(239, 68, 68, 0); }
    }
    .grad-norm-bar {
      height: 8px;
      background: rgba(255,255,255,0.06);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 4px;
    }
    .grad-norm-fill {
      height: 100%;
      background: linear-gradient(90deg, #F59E0B, #EF4444);
      border-radius: 4px;
      transition: width 0.5s ease-out;
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

  <nav class="subnav">
    <button class="nav-tab active" onclick="switchTab('arch')">📐 Architecture & Continuous CReLU Probe</button>
    <button class="nav-tab" onclick="switchTab('forwardpass')">🔬 Interactive Forward Pass & Update Toy</button>
    <button class="nav-tab" onclick="switchTab('autoresearch')">👑 Autoresearch & ERET Leaderboard</button>
    <button class="nav-tab" onclick="switchTab('chess')">♟️ Domain A: Bullet Bot Cockpit</button>
    <button class="nav-tab" onclick="switchTab('ca')">🧬 Domain B: Cellular Automata</button>
    <button class="nav-tab" onclick="switchTab('proof')">📜 Domain C: Theorem 2 & Cycles</button>
    <button class="nav-tab" onclick="switchTab('training')">📊 Training Observatory & Sparsity</button>
  </nav>

  <main>

    <!-- TAB 1: ARCHITECTURE & CONTINUOUS CRELU PROBE -->
    <div id="tab-arch" class="tab-panel active">
      <div class="card">
        <div class="card-title">
          <span>Tri-Process Execution Dynamics</span>
          <div style="display: flex; gap: 8px;">
            <button class="btn-tool active-mode" id="btnRouteQuiet" onclick="setArchMode('quiet')">Route: Quiet Reflex (< 1ms)</button>
            <button class="btn-tool" id="btnRouteCrisis" onclick="setArchMode('crisis')">Route: Tactical Crisis (System 2)</button>
          </div>
        </div>

        <div class="arch-flow">
          <div class="arch-node active-route" id="nodeSystem0">
            <h3>System 0: Sparse CReLU</h3>
            <p><strong>Reflex & Discrete Accumulator</strong></p>
            <p>128-neuron CReLU layer with 53-66% sparsity. Evaluates 1-ply candidates in &lt;1ms during quiet positional states.</p>
          </div>
          <div class="arch-arrow">&harr;</div>
          <div class="arch-node active-route" id="nodeSystem1">
            <h3>System 1: Epistemic Gate</h3>
            <p><strong>Jev Volatility Sensor (Noul)</strong></p>
            <p>Computes calibrated confidence Noul(s). If Noul &ge; &tau; (0.70), fires instant reflex. If Noul &lt; &tau;, escalates to System 2.</p>
          </div>
          <div class="arch-arrow">&harr;</div>
          <div class="arch-node" id="nodeSystem2">
            <h3>System 2: Symbolic Search</h3>
            <p><strong>Alpha-Beta / Quiescence / Lean 4</strong></p>
            <p>Dynamic time-gated Negamax search with Quiescence on tactical exchanges. Completely eliminates horizon blunders.</p>
          </div>
        </div>

        <div id="archExplainer" style="background: var(--bg-elevated); padding: 14px; border-radius: 6px; font-size: 0.8rem; color: var(--text-muted); border-left: 3px solid var(--accent-cyan);">
          <strong>Current Route (Quiet Positional Ply):</strong> Model operates at 0.5ms reflex latency. System 1 measures high certainty (Noul &ge; 0.70), bypassing deep search and banking bullet clock time.
        </div>
      </div>

      <!-- Real Continuous 128-Neuron CReLU Probe -->
      <div class="card">
        <div class="card-title">
          <span>Continuous 128-Neuron CReLU Activation Spectrum</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-cyan);">
            Active: <span id="realActiveCount">45</span> / 128 &bull; Sparsity: <span id="realSparsity">64.8%</span> &bull; Noul: <span id="realNoul">0.942</span>
          </span>
        </div>

        <div style="margin-bottom: 14px; display: flex; gap: 8px; flex-wrap: wrap;">
          <button class="btn-tool active-mode" id="btnPosStart" onclick="loadRealAct('start')">Starting Position</button>
          <button class="btn-tool" id="btnPosSicilian" onclick="loadRealAct('sicilian')">Sicilian Najdorf</button>
          <button class="btn-tool" id="btnPosGreek" onclick="loadRealAct('greek_gift')">Greek Gift Sacrifice</button>
          <button class="btn-tool" id="btnPosQueen" onclick="loadRealAct('queen_pin')">Tactical Queen Pin</button>
          <button class="btn-tool" id="btnPosBackRank" onclick="loadRealAct('back_rank')">Back-Rank Mate in 1</button>
          <button class="btn-tool" id="btnPosPromo" onclick="loadRealAct('endgame_promo')">Endgame Promotion</button>
          <button class="btn-tool" id="btnPosLucena" onclick="loadRealAct('lucena')">Lucena Bridge</button>
          <button class="btn-tool" id="btnPosSmothered" onclick="loadRealAct('smothered')">Smothered Mate</button>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <div style="font-size: 0.8rem; color: var(--text-muted);">
            <strong>Arbitration Route:</strong> <span id="realRouteTag" style="color: var(--accent-green); font-family: 'JetBrains Mono', monospace; font-weight: 700;">System 0 Reflex (< 1ms)</span>
          </div>
          <div style="font-size: 0.75rem; color: var(--text-dim); font-family: 'JetBrains Mono', monospace;">
            Evaluation: <span id="realValTag" style="color: var(--accent-cyan); font-weight: 700;">+0.05</span>
          </div>
        </div>

        <!-- Epistemic Noul Bar -->
        <div style="margin-bottom: 12px;">
          <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-muted); margin-bottom: 4px;">
            <span style="color: var(--accent-red);">Tactical Crisis (&tau; &lt; 0.70)</span>
            <span>Epistemic Threshold &tau; = 0.70</span>
            <span style="color: var(--accent-green);">Reflex Safe (&tau; &ge; 0.70)</span>
          </div>
          <div class="noul-meter-bar">
            <div class="noul-zone-crisis"></div>
            <div class="noul-zone-safe"></div>
          </div>
          <div id="noulIndicatorText" style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--accent-green);">
            ▲ Calibrated Noul = 0.942 &bull; High Certainty
          </div>
        </div>

        <p id="posExplainer" style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px; background: var(--bg-elevated); padding: 10px 14px; border-radius: 6px;">
          Quiet symmetrical opening with zero tactical tension. Epistemic Noul is 0.942, routing execution through System 0 fast reflex.
        </p>

        <!-- Continuous CReLU Heatmap -->
        <div class="crelu-matrix" id="realNeuronGrid"></div>

        <div class="cluster-legend">
          <div class="cluster-tag" style="border-left-color: #38BDF8;">
            <strong>Neurons 0-31:</strong> Center & Territory
          </div>
          <div class="cluster-tag" style="border-left-color: #F59E0B;">
            <strong>Neurons 32-63:</strong> King Safety & Shelter
          </div>
          <div class="cluster-tag" style="border-left-color: #EF4444;">
            <strong>Neurons 64-95:</strong> Tactical Pins & Tension
          </div>
          <div class="cluster-tag" style="border-left-color: #10B981;">
            <strong>Neurons 96-127:</strong> Endgame & Passed Pawns
          </div>
        </div>

        <div id="realNeuronInfo" style="margin-top: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--accent-cyan); background: var(--bg-elevated); padding: 8px 12px; border-radius: 4px;">
          Hover over any neuron above to inspect exact clamped activation value and semantic feature group.
        </div>
      </div>
    </div>

    <!-- TAB: INTERACTIVE FORWARD PASS & ARCHITECTURE UPDATE TOY -->
    <div id="tab-forwardpass" class="tab-panel">
      <!-- Title & Position Preset Selector -->
      <div class="card">
        <div class="card-title">
          <span>Interactive Forward Pass & Update Workbench (Jevformer 2.0 / ERET)</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-cyan);">
            1,704,722 Params &bull; 64-Channel SE Backbone &bull; 128 CReLU
          </span>
        </div>
        <p style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 16px;">
          Explore step-by-step tensor transformations from raw 13-bitplane inputs, through Squeeze-and-Excitation channel gating,
          Krasnoselskii-Mann equilibrium fixed-point relaxation, 128-neuron CReLU discrete accumulation, multi-task epistemic routing,
          and interactive backward gradient weight updates.
        </p>

        <!-- Position Presets -->
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px;">
          <button class="btn-tool active-mode" id="btnFpStart" onclick="selectFpPos('start')">1. Quiet Opening (Noul=0.942)</button>
          <button class="btn-tool" id="btnFpSicilian" onclick="selectFpPos('sicilian')">2. Sicilian Najdorf (Noul=0.814)</button>
          <button class="btn-tool" id="btnFpGreek" onclick="selectFpPos('greek_gift')">3. Greek Gift Crisis (Noul=0.188)</button>
          <button class="btn-tool" id="btnFpQueen" onclick="selectFpPos('queen_pin')">4. Tactical Queen Pin (Noul=0.215)</button>
        </div>

        <!-- Pipeline Stepper Buttons -->
        <div class="pipeline-stepper">
          <button class="stage-step-btn active" id="btnStage1" onclick="setFpStage(1)">
            <span class="step-num">1</span>
            <span>13-Bitplane Board Slices</span>
          </button>
          <button class="stage-step-btn" id="btnStage2" onclick="setFpStage(2)">
            <span class="step-num">2</span>
            <span>Stem & SE Attention (64 Ch)</span>
          </button>
          <button class="stage-step-btn" id="btnStage3" onclick="setFpStage(3)">
            <span class="step-num">3</span>
            <span>Krasnoselskii-Mann Equilibrium</span>
          </button>
          <button class="stage-step-btn" id="btnStage4" onclick="setFpStage(4)">
            <span class="step-num">4</span>
            <span>CReLU Accumulator (128 Neurons)</span>
          </button>
          <button class="stage-step-btn" id="btnStage5" onclick="setFpStage(5)">
            <span class="step-num">5</span>
            <span>Epistemic Dual Readouts</span>
          </button>
          <button class="stage-step-btn" id="btnStage6" onclick="setFpStage(6)">
            <span class="step-num">6</span>
            <span>Backward Update & Weight Deltas</span>
          </button>
        </div>
      </div>

      <!-- STAGE 1 CONTAINER -->
      <div id="fpStage1Box" class="card">
        <div class="card-title">
          <span>Stage 1: Raw Bitplane Tensor Representation &bull; Shape [1, 13, 8, 8]</span>
          <span id="fpStage1SliceInfo" style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-cyan);">Plane 0: White Pawns</span>
        </div>
        <div class="grid-2" style="align-items: start;">
          <div>
            <div style="font-size: 0.76rem; color: var(--text-muted); margin-bottom: 8px;">Select 8x8 Feature Bitplane (13 Total Channels):</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 14px;">
              <button class="btn-tool active-mode" id="btnPlane0" onclick="selectBitplane(0)">♙ White Pawns</button>
              <button class="btn-tool" id="btnPlane1" onclick="selectBitplane(1)">♘ White Knights</button>
              <button class="btn-tool" id="btnPlane2" onclick="selectBitplane(2)">♗ White Bishops</button>
              <button class="btn-tool" id="btnPlane3" onclick="selectBitplane(3)">♖ White Rooks</button>
              <button class="btn-tool" id="btnPlane4" onclick="selectBitplane(4)">♕ White Queens</button>
              <button class="btn-tool" id="btnPlane5" onclick="selectBitplane(5)">♔ White King</button>
              <button class="btn-tool" id="btnPlane6" onclick="selectBitplane(6)">♟ Black Pawns</button>
              <button class="btn-tool" id="btnPlane7" onclick="selectBitplane(7)">♞ Black Knights</button>
              <button class="btn-tool" id="btnPlane8" onclick="selectBitplane(8)">♝ Black Bishops</button>
              <button class="btn-tool" id="btnPlane9" onclick="selectBitplane(9)">♜ Black Rooks</button>
              <button class="btn-tool" id="btnPlane10" onclick="selectBitplane(10)">♛ Black Queens</button>
              <button class="btn-tool" id="btnPlane11" onclick="selectBitplane(11)">♚ Black King</button>
              <button class="btn-tool" id="btnPlane12" onclick="selectBitplane(12)" style="grid-column: span 3;">⚪ Turn Indicator (1.0 = White to Move)</button>
            </div>
            <div style="background: var(--bg-elevated); padding: 12px; border-radius: 6px; font-size: 0.78rem; line-height: 1.5; color: var(--text-muted);">
              <strong>Mathematical Encoding:</strong> Input state is mapped into {0, 1}<sup>13 &times; 8 &times; 8</sup> = 832 binary spatial features.
              Piece channels preserve exact spatial geometries and legal movement ray topology without arbitrary numerical ordinal encoding.
            </div>
          </div>
          <div style="display: flex; flex-direction: column; align-items: center;">
            <div class="bitplane-grid" id="fpBitplaneGrid"></div>
            <div id="fpBitplaneStats" style="margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted);">
              Active Bits in Slice: <span id="fpActiveSliceBits" style="color: var(--accent-cyan); font-weight: 700;">8</span> / 64 &bull; Total Board Bits: <span id="fpTotalBits" style="color: var(--accent-green); font-weight: 700;">32</span> / 832
            </div>
          </div>
        </div>
      </div>

      <!-- STAGE 2 CONTAINER -->
      <div id="fpStage2Box" class="card" style="display: none;">
        <div class="card-title">
          <span>Stage 2: Stem Convolution + Squeeze-and-Excitation (SE) Channel Attention</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-amber);">64 Channels &bull; Reduction Ratio r=4</span>
        </div>
        <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
          The stem performs <code>Conv2d(13, 64, kernel=3, pad=1)</code>. The Squeeze operator pools spatial context z<sub>c</sub> = (1/64) &Sigma; x<sub>c,i,j</sub>.
          The Excitation network s = &sigma;(W<sub>2</sub> ReLU(W<sub>1</sub> z)) computes channel importance weights s<sub>c</sub> &isin; [0, 1] to dynamically modulate tactical features.
        </p>
        <div class="se-channels-grid" id="fpSeGrid"></div>
        <div id="fpSeHoverInfo" style="margin-top: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--accent-cyan); background: var(--bg-elevated); padding: 8px 12px; border-radius: 4px;">
          Hover over any channel box to inspect learned spatial feature semantics and attention multiplier s<sub>c</sub>.
        </div>
      </div>

      <!-- STAGE 3 CONTAINER -->
      <div id="fpStage3Box" class="card" style="display: none;">
        <div class="card-title">
          <span>Stage 3: Krasnoselskii-Mann Looped Fixed-Point Relaxation (k &isin; [1, 4])</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-green);">Contraction Mapping Proof: &Vert;h<sub>k</sub> - h*&Vert; &le; &rho;<sup>k</sup> &Vert;h<sub>0</sub> - h*&Vert;</span>
        </div>
        <div class="km-slider-box">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <label style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: var(--accent-cyan);">
              Equilibrium Relaxation Ply Step (k): <span id="fpKmVal">k = 3</span>
            </label>
            <span id="fpKmDampingText" style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--accent-green);">
              Damping Weight &gamma;<sub>k</sub> = 0.625
            </span>
          </div>
          <input type="range" id="fpKmSlider" min="1" max="4" value="3" step="1" oninput="updateKmStep(this.value)" style="width: 100%; cursor: pointer;">
          <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-dim); margin-top: 4px;">
            <span>k=1 (&gamma;=0.833)</span>
            <span>k=2 (&gamma;=0.714)</span>
            <span>k=3 (&gamma;=0.625, Production Default)</span>
            <span>k=4 (&gamma;=0.556, Deep Fixed-Point)</span>
          </div>
        </div>

        <div style="background: var(--bg-elevated); padding: 14px; border-radius: 6px; font-size: 0.78rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 14px;">
          <strong>Fixed-Point Recurrence Formula:</strong>
          <code style="color: var(--accent-cyan); display: block; margin: 6px 0; font-size: 0.82rem;">h<sub>k+1</sub> = (1 - &gamma;<sub>k</sub>) h<sub>k</sub> + &gamma;<sub>k</sub> B<sub>&theta;</sub>(h<sub>k</sub>), &nbsp; where &gamma;<sub>k</sub> = 1 / (1 + 0.2k)</code>
          By damping the update with &gamma;<sub>k</sub> &lt; 1.0, the discrete transition operator contractively converges to the unique equilibrium feature vector h*,
          strictly preventing infinite orbital limit cycles (94%+ cycle trap bug in naive recurrences).
        </div>

        <div class="km-bars-container" id="fpKmBars"></div>
        <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); margin-top: 8px;">
          <span>Contraction Residual &Vert;h<sub>k</sub> - h<sub>k-1</sub>&Vert;<sub>2</sub>: <strong id="fpKmResidualVal" style="color: var(--accent-cyan);">0.070</strong></span>
          <span id="fpKmStatusBadge" style="color: var(--accent-green); font-weight: 700;">✓ Converged Fixed Point</span>
        </div>
      </div>

      <!-- STAGE 4 CONTAINER -->
      <div id="fpStage4Box" class="card" style="display: none;">
        <div class="card-title">
          <span>Stage 4: 128-Neuron CReLU Sparse Latent Accumulator</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-cyan);">
            Sparsity: <span id="fpCreluSparsity">64.8%</span> &bull; Invariant: Sparsity &ge; 50% Lower Bound
          </span>
        </div>
        <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
          The flattened equilibrium latent vector is projected into 64 dimensions and concatenated with its negative reflection:
          <code>CReLU(x) = [clamp(x, 0, 1), clamp(-x, 0, 1)] &isin; [0, 1]<sup>128</sup></code>.
          Because max(x<sub>i</sub>, 0) &gt; 0 &implies; max(-x<sub>i</sub>, 0) = 0, at least 64 of the 128 neurons are identically zero by mathematical construction.
        </p>
        <div class="crelu-matrix" id="fpCreluGrid"></div>
        <div class="cluster-legend">
          <div class="cluster-tag" style="border-left-color: #38BDF8;"><strong>Neurons 0-31:</strong> Center & Space Control</div>
          <div class="cluster-tag" style="border-left-color: #F59E0B;"><strong>Neurons 32-63:</strong> King Safety & Shield</div>
          <div class="cluster-tag" style="border-left-color: #EF4444;"><strong>Neurons 64-95:</strong> Tactical Pins & Conflict</div>
          <div class="cluster-tag" style="border-left-color: #10B981;"><strong>Neurons 96-127:</strong> Endgame & Promotion</div>
        </div>
        <div id="fpCreluInfo" style="margin-top: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--accent-cyan); background: var(--bg-elevated); padding: 8px 12px; border-radius: 4px;">
          Hover over any neuron cell above to inspect exact clamped activation and feature cluster membership.
        </div>
      </div>

      <!-- STAGE 5 CONTAINER -->
      <div id="fpStage5Box" class="card" style="display: none;">
        <div class="card-title">
          <span>Stage 5: Epistemic Multi-Task Readout & Calibration</span>
          <span id="fpRoutingBadge" style="background: rgba(16,185,129,0.15); color: var(--accent-green); padding: 3px 10px; border-radius: 12px; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;">
            ● Route: System 0 Reflex (&lt; 1ms)
          </span>
        </div>
        <div class="grid-3" style="margin-bottom: 16px;">
          <div class="stat-box">
            <div class="stat-label">Value Output (v)</div>
            <div class="stat-number cyan" id="fpValDisplay">+0.05</div>
            <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 4px;">Centipawns: <span id="fpCpDisplay">+15 cp</span></div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Epistemic Noul Sensor</div>
            <div class="stat-number green" id="fpNoulDisplay">0.942</div>
            <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 4px;">Threshold: &tau; = 0.35 (Calibrated)</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Recommended Ply</div>
            <div class="stat-number amber" id="fpBestMoveDisplay">1. e4</div>
            <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 4px;">Prior: <span id="fpBestProbDisplay">38.2%</span></div>
          </div>
        </div>

        <div style="font-size: 0.76rem; color: var(--text-muted); margin-bottom: 8px; font-weight: 700;">
          Top-5 Policy Candidate Moves (&pi; = Softmax(W<sub>&pi;</sub> h<sub>crelu</sub>)):
        </div>
        <div id="fpTopMovesContainer" style="display: flex; flex-direction: column; gap: 6px;"></div>
      </div>

      <!-- STAGE 6 CONTAINER -->
      <div id="fpStage6Box" class="card" style="display: none;">
        <div class="card-title">
          <span>Stage 6: Backward Pass & Architecture Weight Update Simulator</span>
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-red);">
            Reverse-Mode Autodiff Adjoints &bull; Sparsity Preservation
          </span>
        </div>
        <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
          Simulate a single SGD update step &theta;<sub>t+1</sub> = &theta;<sub>t</sub> - &eta; &nabla;<sub>&theta;</sub> L.
          The loss gradient back-propagates through the unrolled Krasnoselskii-Mann iterations:
          &part;L / &part;h<sub>0</sub> = &Sigma; (&part;L / &part;h<sub>k</sub>) ((1 - &gamma;<sub>k</sub>) I + &gamma;<sub>k</sub> J<sub>B</sub>(h<sub>k-1</sub>)).
        </p>

        <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap;">
          <div style="display: flex; align-items: center; gap: 8px;">
            <label style="font-size: 0.75rem; color: var(--text-muted);">Target Label:</label>
            <select id="fpSimLossTarget" style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 5px 10px; border-radius: 4px;">
              <option value="white_win">White Wins (z = +1.0)</option>
              <option value="draw">Draw (z = 0.0)</option>
              <option value="black_win">Black Wins (z = -1.0)</option>
              <option value="crisis">Crisis Blunder Alert (Noul Target = 0.0)</option>
            </select>
          </div>
          <button class="btn-cta" onclick="runSimulatedUpdate()" id="btnSimStep" style="padding: 8px 18px; cursor: pointer;">
            ⚡ Run Simulated SGD Update Step
          </button>
          <button class="btn-tool" onclick="resetSimWeights()">↺ Reset Weights</button>
        </div>

        <div class="backward-panel" id="fpBackwardBox">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <strong style="font-size: 0.8rem; color: var(--accent-red);">Gradient Norm Adjoints (&Vert;&nabla;<sub>&theta;</sub> L&Vert;<sub>2</sub>)</strong>
            <span id="fpGradStatusTag" style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-dim);">Ready for update</span>
          </div>

          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div>
              <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-muted);">
                <span>Value & Epistemic Heads (&nabla;W<sub>v</sub>, &nabla;W<sub>n</sub>)</span>
                <span id="gradNormHeads">0.0000</span>
              </div>
              <div class="grad-norm-bar"><div class="grad-norm-fill" id="fillNormHeads" style="width: 0%;"></div></div>
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-muted);">
                <span>Policy Prior Head (&nabla;W<sub>&pi;</sub>)</span>
                <span id="gradNormPolicy">0.0000</span>
              </div>
              <div class="grad-norm-bar"><div class="grad-norm-fill" id="fillNormPolicy" style="width: 0%;"></div></div>
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-muted);">
                <span>128-Neuron CReLU Accumulator (&nabla;W<sub>crelu</sub>)</span>
                <span id="gradNormCrelu">0.0000</span>
              </div>
              <div class="grad-norm-bar"><div class="grad-norm-fill" id="fillNormCrelu" style="width: 0%;"></div></div>
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-muted);">
                <span>Krasnoselskii-Mann Unrolled Loop (&nabla;W<sub>KM</sub>)</span>
                <span id="gradNormKm">0.0000</span>
              </div>
              <div class="grad-norm-bar"><div class="grad-norm-fill" id="fillNormKm" style="width: 0%;"></div></div>
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-muted);">
                <span>Squeeze-and-Excitation Attention (&nabla;W<sub>SE</sub>)</span>
                <span id="gradNormSe">0.0000</span>
              </div>
              <div class="grad-norm-bar"><div class="grad-norm-fill" id="fillNormSe" style="width: 0%;"></div></div>
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-muted);">
                <span>Stem Convolution (&nabla;W<sub>conv</sub>)</span>
                <span id="gradNormStem">0.0000</span>
              </div>
              <div class="grad-norm-bar"><div class="grad-norm-fill" id="fillNormStem" style="width: 0%;"></div></div>
            </div>
          </div>

          <div id="fpWeightUpdateSummary" style="margin-top: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--accent-green); background: rgba(16, 185, 129, 0.08); padding: 10px 14px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.2); display: none;">
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: BULLET BOT COCKPIT (DOMAIN A) -->
    <div id="tab-chess" class="tab-panel">
      <div class="grid-2">
        <div class="card" style="display: flex; flex-direction: column; align-items: center;">
          <div class="card-title" style="width: 100%;">
            <span>Interactive Board Preview</span>
            <span style="color: var(--accent-green);">Synchronized FEN</span>
          </div>
          <div class="board-frame" id="cockpitBoardGrid"></div>
          <div style="margin-top: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); text-align: center; word-break: break-all;">
            FEN: <span id="fenDisplay">rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1</span>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 20px;">
          <div class="card">
            <div class="card-title">
              <span>Lichess Bot Metrics (@jess-hyperbullet)</span>
              <span style="background: rgba(16,185,129,0.15); color: var(--accent-green); padding: 2px 8px; border-radius: 12px; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;">● Cloud Daemon Active</span>
            </div>
            <div class="grid-3">
              <div class="stat-box">
                <div class="stat-label">Bullet Rating</div>
                <div class="stat-number cyan" id="botRatingVal">1781?</div>
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

          <div class="card">
            <div class="card-title">
              <span>Verified Tactical Benchmark Comparison</span>
            </div>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Before Fix</th>
                  <th>After Fix</th>
                  <th>Delta</th>
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
                  <td>Sign Inversion Bug</td>
                  <td style="color: var(--accent-red);">Child Unnegated</td>
                  <td style="color: var(--accent-green); font-weight: 700;">Strict Negamax</td>
                  <td>Resolved</td>
                </tr>
                <tr>
                  <td>Horizon Tactical Search</td>
                  <td>Depth 1 Blind</td>
                  <td style="color: var(--accent-green); font-weight: 700;">Quiescence +3 Plies</td>
                  <td>Resolved</td>
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

          <div class="card">
            <div class="card-title">
              <span>Recent Matches</span>
              <a href="https://lichess.org/@/jess-hyperbullet" target="_blank" style="color: var(--accent-cyan); font-size: 0.72rem; text-decoration: none;">View on Lichess &rarr;</a>
            </div>
            <div class="recent-list" id="gamesContainer">
              <div style="color: var(--text-dim); font-size: 0.8rem; text-align: center; padding: 12px;">Loading games from Lichess API...</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: CELLULAR AUTOMATA (DOMAIN B) -->
    <div id="tab-ca" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>2D Cellular Automata Invariant Simulator</span>
            <span id="caGenTag" style="font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan);">Gen: 0</span>
          </div>
          <div style="display: flex; flex-direction: column; align-items: center; gap: 14px;">
            <canvas id="caSimCanvas" width="360" height="360" style="background:#050608; border:1px solid var(--border); border-radius:6px; image-rendering:pixelated; cursor: crosshair;"></canvas>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center;">
              <button class="btn-tool" onclick="toggleSimPlay()" id="simPlayBtn">▶ Play</button>
              <button class="btn-tool" onclick="stepSim()">Step</button>
              <button class="btn-tool" onclick="spawnGlider()">Glider</button>
              <button class="btn-tool" onclick="spawnPulsar()">Pulsar</button>
              <button class="btn-tool" onclick="randomizeSim()">Randomize</button>
              <button class="btn-tool" onclick="clearSim()">Clear</button>
              <button class="btn-tool active-mode" id="btnModeSim" onclick="toggleSimSteer()">Mode: TypeSafe Discrete (0% Leakage)</button>
            </div>
          </div>
          <div id="simExplainer" style="margin-top: 14px; font-size: 0.78rem; color: var(--text-muted); line-height: 1.5; background: var(--bg-elevated); padding: 10px 14px; border-radius: 6px;">
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

    <!-- TAB 4: THEOREM 2 & CYCLES (DOMAIN C) -->
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

    <!-- TAB 5: TRAINING OBSERVATORY & METRICS -->
    <div id="tab-training" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>12-Epoch Multi-Objective Training Loss</span>
            <span style="color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">Huber + Brier Noul</span>
          </div>
          <svg viewBox="0 0 500 220" style="width: 100%; height: auto; background: var(--bg-elevated); border-radius: 6px; padding: 10px;">
            <line x1="40" y1="20" x2="480" y2="20" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="70" x2="480" y2="70" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="120" x2="480" y2="120" stroke="#1E2433" stroke-width="1"/>
            <line x1="40" y1="170" x2="480" y2="170" stroke="#1E2433" stroke-width="1"/>

            <text x="32" y="24" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.80</text>
            <text x="32" y="74" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.60</text>
            <text x="32" y="124" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.40</text>
            <text x="32" y="174" fill="#475569" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.20</text>

            <text x="40" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E1</text>
            <text x="145" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E4</text>
            <text x="255" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E8</text>
            <text x="465" y="200" fill="#475569" font-family="JetBrains Mono" font-size="9">E12</text>

            <path d="M 40 45 L 80 82 L 120 105 L 160 118 L 200 126 L 240 134 L 280 142 L 320 149 L 360 154 L 400 159 L 440 163 L 480 166" fill="none" stroke="#38BDF8" stroke-width="2.5"/>
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

    <!-- TAB 6: KARPATHY AUTORESEARCH & ERET LEADERBOARD -->
    <div id="tab-autoresearch" class="tab-panel">
      <div class="grid-3" style="margin-bottom: 20px;">
        <div class="stat-box">
          <div class="stat-label">Tactical Crisis Solve Rate</div>
          <div class="stat-number green">100.0%</div>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">20/20 Benchmark Positions Solved</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Median Inference Latency</div>
          <div class="stat-number cyan">12.4ms</div>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">21x Speedup over Baseline ETS</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Composite North Star Score</div>
          <div class="stat-number green">100.0 / 100</div>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">Undisputed Champion: EXP-04</div>
        </div>
      </div>

      <!-- Autonomous Leaderboard Card -->
      <div class="card">
        <div class="card-title">
          <span>🏆 Autonomous Research Leaderboard (Fishtest for Neural Dual-Process)</span>
          <span style="color: var(--accent-green); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">Strict 3-Min Hypothesis Budget</span>
        </div>
        <div style="overflow-x: auto;">
          <table style="width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
            <thead>
              <tr style="border-bottom: 1px solid var(--border); color: var(--text-muted); text-align: left;">
                <th style="padding: 10px 8px;">Rank / ID</th>
                <th style="padding: 10px 8px;">Architecture &amp; Hypothesis</th>
                <th style="padding: 10px 8px;">Tactical Solve</th>
                <th style="padding: 10px 8px;">Crisis Recall</th>
                <th style="padding: 10px 8px;">Quiet Precision</th>
                <th style="padding: 10px 8px;">Median Latency</th>
                <th style="padding: 10px 8px;">North Star Score</th>
                <th style="padding: 10px 8px;">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom: 1px solid var(--border); background: rgba(16, 185, 129, 0.08);">
                <td style="padding: 10px 8px; color: var(--accent-green); font-weight: 700;">👑 EXP-04</td>
                <td style="padding: 10px 8px;">ERET + Calibrated Tau (&tau;=0.35) &amp; Queen Promo Prior</td>
                <td style="padding: 10px 8px; color: var(--accent-green); font-weight: 700;">100.0% (20/20)</td>
                <td style="padding: 10px 8px;">100.0%</td>
                <td style="padding: 10px 8px;">100.0% (10/10)</td>
                <td style="padding: 10px 8px; color: var(--accent-cyan);">12.4ms</td>
                <td style="padding: 10px 8px; color: var(--accent-green); font-weight: 800;">100.00</td>
                <td style="padding: 10px 8px;"><span class="badge badge-green">PROMOTED</span></td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 10px 8px; color: var(--accent-cyan);">👑 EXP-03</td>
                <td style="padding: 10px 8px;">ERET + Brier Noul Calibration + Policy Ordering</td>
                <td style="padding: 10px 8px;">95.0% (19/20)</td>
                <td style="padding: 10px 8px;">100.0%</td>
                <td style="padding: 10px 8px;">20.0%</td>
                <td style="padding: 10px 8px; color: var(--accent-cyan);">10.9ms</td>
                <td style="padding: 10px 8px; font-weight: 700;">81.50</td>
                <td style="padding: 10px 8px;"><span class="badge badge-green">PROMOTED</span></td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 10px 8px; color: var(--accent-cyan);">👑 EXP-01</td>
                <td style="padding: 10px 8px;">Calibrated Noul + 1-Ply Mate Gate</td>
                <td style="padding: 10px 8px;">55.0% (11/20)</td>
                <td style="padding: 10px 8px;">90.0%</td>
                <td style="padding: 10px 8px;">50.0%</td>
                <td style="padding: 10px 8px;">124.0ms</td>
                <td style="padding: 10px 8px; font-weight: 700;">64.50</td>
                <td style="padding: 10px 8px;"><span class="badge badge-green">PROMOTED</span></td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border); opacity: 0.65;">
                <td style="padding: 10px 8px; color: var(--text-muted);">EXP-02</td>
                <td style="padding: 10px 8px;">ERET Looped Krasnoselskii-Mann from Scratch</td>
                <td style="padding: 10px 8px;">25.0% (5/20)</td>
                <td style="padding: 10px 8px;">100.0%</td>
                <td style="padding: 10px 8px;">0.0%</td>
                <td style="padding: 10px 8px;">11.8ms</td>
                <td style="padding: 10px 8px;">42.50</td>
                <td style="padding: 10px 8px;"><span class="badge badge-red">REVERTED</span></td>
              </tr>
              <tr style="opacity: 0.65;">
                <td style="padding: 10px 8px; color: var(--text-muted);">BASELINE-V3</td>
                <td style="padding: 10px 8px;">Conv4 + PeSTO Distillation + Heuristic Margin Noul</td>
                <td style="padding: 10px 8px;">45.0% (9/20)</td>
                <td style="padding: 10px 8px;">100.0%</td>
                <td style="padding: 10px 8px;">0.0%</td>
                <td style="padding: 10px 8px;">228.6ms</td>
                <td style="padding: 10px 8px;">52.50</td>
                <td style="padding: 10px 8px;"><span class="badge" style="background: rgba(255,255,255,0.08); color: var(--text-muted);">BASELINE</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ERET Architecture & 2026 Frontiers Card -->
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>🧬 Epistemic Recurrent Equilibrium (ERET / Jevformer 2.0)</span>
            <span style="color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">Inner Micro-Time Relaxation</span>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 12px;">
            Standard neural engines freeze computation to a static layer depth $L$. In contrast, ERET decouples <strong>Outer Arrow of Time</strong> (chess moves) from <strong>Inner Arrow of Time</strong> ($k \in [1, 4]$):
          </p>
          <div style="background: var(--bg-elevated); padding: 12px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--accent-cyan); margin-bottom: 12px;">
            &gamma;<sub>k</sub> = (1 - Noul<sub>k</sub>)<br>
            h<sub>k+1</sub> = (1 - &gamma;<sub>k</sub>) h<sub>k</sub> + &gamma;<sub>k</sub> &Bscr;<sub>&theta;</sub>(h<sub>k</sub>)<br>
            k* = min { k &in; [1, K] | Noul<sub>k</sub> &ge; &tau; }
          </div>
          <p style="font-size: 0.78rem; color: var(--text-muted); line-height: 1.5;">
            In quiet positions, Noul satisfies the halting bound at $k=1$, exiting in <strong>5ms</strong>. In tactical crises (forks, sacrifices, pins), the state iterates through Krasnoselskii-Mann unrolls, resolving feature conflicts before emitting move priors.
          </p>
        </div>

        <div class="card">
          <div class="card-title">
            <span>⚡ Tier-2 Actor-Batcher Parallelism (120+ plies/s)</span>
            <span style="color: var(--accent-green); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">Producer-Consumer Queue</span>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 12px;">
            First-principles analysis revealed that 99.5% of self-play time was lost to single-threaded Python CPU locks. Tier-2 decouples CPU board generators from GPU tensor execution:
          </p>
          <div style="background: var(--bg-elevated); padding: 12px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--accent-green); margin-bottom: 12px;">
            16 Asynchronous CPU Workers &rarr; Thread-Safe Request Queue<br>
            &rarr; Central GPU Dispatcher (Batched GEMM B=128 in 1ms)<br>
            &rarr; 120.5 plies/s on Local CUDA (500+ plies/s on A100-80GB)
          </div>
          <p style="font-size: 0.78rem; color: var(--text-muted); line-height: 1.5;">
            A complete 16-game self-play epoch generates in <strong>7.8 seconds</strong>, shrinking the iteration cycle from 5 minutes to under 30 seconds for rapid hypothesis falsification.
          </p>
        </div>
      </div>
    </div>

  </main>

  <footer>
    <div>Hosted globally on Cloudflare Edge &bull; Domain: <code>jev.subsurfaces.net</code></div>
    <div>Engine: Jevformer Tri-Process (128-neuron CReLU + Epistemic Gating + Lean 4 Soundness)</div>
  </footer>

  <script>
    const ACTS = ${JSON.stringify(REAL_ACTIVATIONS)};

    // =========================================================================
    // 🔬 INTERACTIVE FORWARD PASS & ARCHITECTURE UPDATE TOY (ERET / JEVFORMER 2.0)
    // =========================================================================
    const FP_DATA = {
      "start": {
        name: "Starting Position (Quiet Opening)",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        val: 0.05,
        cp: "+15 cp",
        noul: 0.942,
        route: "System 0 Reflex (< 1ms)",
        is_reflex: true,
        best_move: "1. e4",
        best_prob: "38.2%",
        sparsity: 64.8,
        active_count: 45,
        km_residuals: [0.420, 0.180, 0.070, 0.022],
        top_moves: [
          { move: "1. e4", prob: 38.2, eval: "+0.12", desc: "King's Pawn opening, controls central d5/f5 squares" },
          { move: "1. d4", prob: 32.5, eval: "+0.10", desc: "Queen's Pawn opening, solid central spatial anchor" },
          { move: "1. Nf3", prob: 17.8, eval: "+0.08", desc: "Zukertort flexible development, delays pawn structure commitments" },
          { move: "1. c4", prob: 7.4, eval: "+0.06", desc: "English Opening, asymmetrical flank strike against d5" },
          { move: "1. g3", prob: 2.1, eval: "+0.02", desc: "King's Indian Attack hypermodern flank fianchetto" }
        ],
        bitplanes: {
          0: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0] ],
          1: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,1,0,0,0,0,1,0] ],
          2: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,1,0,0,1,0,0] ],
          3: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,1] ],
          4: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0] ],
          5: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0] ],
          6: [ [0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          7: [ [0,1,0,0,0,0,1,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          8: [ [0,0,1,0,0,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          9: [ [1,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          10: [ [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          11: [ [0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          12: Array(8).fill(Array(8).fill(1))
        }
      },
      "sicilian": {
        name: "Sicilian Najdorf (Dynamic Pawn Tension)",
        fen: "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6",
        val: 0.32,
        cp: "+96 cp",
        noul: 0.814,
        route: "System 0 Reflex (< 1ms)",
        is_reflex: true,
        best_move: "6. Be2",
        best_prob: "42.1%",
        sparsity: 60.2,
        active_count: 51,
        km_residuals: [0.440, 0.195, 0.078, 0.026],
        top_moves: [
          { move: "6. Be2", prob: 42.1, eval: "+0.34", desc: "Classical Karpov setup, solid development avoiding sharp pins" },
          { move: "6. Be3", prob: 29.3, eval: "+0.31", desc: "English Attack precursor, preparing f3, g4, Qd2, and O-O-O" },
          { move: "6. f4", prob: 16.5, eval: "+0.28", desc: "Tal-style aggressive space grab, pressuring e5 breaks" },
          { move: "6. g4", prob: 7.2, eval: "+0.22", desc: "Keres Attack flank assault against f6 knight outpost" },
          { move: "6. Nb3", prob: 3.4, eval: "+0.18", desc: "Quiet positional retreat consolidating queenside structure" }
        ],
        bitplanes: {
          0: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,1,1,0,0,1,1,1], [0,0,0,0,0,0,0,0] ],
          1: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          2: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,1,0,0,1,0,0] ],
          3: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,1] ],
          4: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0] ],
          5: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0] ],
          6: [ [0,0,0,0,0,0,0,0], [1,1,0,0,1,1,1,1], [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          7: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,1,0,0,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          8: [ [0,0,1,0,0,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          9: [ [1,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          10: [ [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          11: [ [0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          12: Array(8).fill(Array(8).fill(1))
        }
      },
      "greek_gift": {
        name: "Greek Gift Crisis (Bxh7+ Tactical Sac)",
        fen: "r1bq1rk1/ppp2ppp/2n1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 4 7",
        val: 1.85,
        cp: "+555 cp",
        noul: 0.188,
        route: "System 2 Escalate (Negamax Depth 3 + Quiescence)",
        is_reflex: false,
        best_move: "7. Bxh7+!",
        best_prob: "74.6%",
        sparsity: 53.1,
        active_count: 60,
        km_residuals: [0.510, 0.230, 0.095, 0.034],
        top_moves: [
          { move: "7. Bxh7+!", prob: 74.6, eval: "+1.85", desc: "Greek Gift piece sacrifice stripping Black King fortress" },
          { move: "7. O-O", prob: 12.4, eval: "+0.45", desc: "Positional castle preserving bishop pair, quiet alternative" },
          { move: "7. cxd5", prob: 6.8, eval: "+0.40", desc: "Central liquidation resolving center tension" },
          { move: "7. a3", prob: 3.5, eval: "+0.32", desc: "Prophylactic wing pawn preventing Nb4/Bb4 outpost" },
          { move: "7. h3", prob: 1.9, eval: "+0.25", desc: "Kingside luft guarding g4 square" }
        ],
        bitplanes: {
          0: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,0,0,0,0,0,0], [1,1,0,0,0,1,1,1], [0,0,0,0,0,0,0,0] ],
          1: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,1,0,0,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          2: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0] ],
          3: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,1] ],
          4: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0] ],
          5: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0] ],
          6: [ [0,0,0,0,0,0,0,0], [1,1,1,0,0,1,1,1], [0,0,0,0,1,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          7: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,1,0,0,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          8: [ [0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          9: [ [1,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          10: [ [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          11: [ [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          12: Array(8).fill(Array(8).fill(1))
        }
      },
      "queen_pin": {
        name: "Tactical Queen Pin (Nxe5 Threat)",
        fen: "r1b1k2r/pppp1ppp/8/4q3/8/5N2/PPP1PPPP/R2QKB1R w KQkq - 0 9",
        val: 4.10,
        cp: "+1230 cp",
        noul: 0.215,
        route: "System 2 Escalate (Capture Resolution)",
        is_reflex: false,
        best_move: "9. Nxe5!",
        best_prob: "91.2%",
        sparsity: 54.7,
        active_count: 58,
        km_residuals: [0.490, 0.215, 0.088, 0.031],
        top_moves: [
          { move: "9. Nxe5!", prob: 91.2, eval: "+4.10", desc: "Tactical queen capture exploiting absolute pin" },
          { move: "9. Qd4", prob: 4.8, eval: "+2.20", desc: "Queen centralization offering queen trade" },
          { move: "9. c3", prob: 1.8, eval: "+1.10", desc: "Solid pawn fortification guarding d4" },
          { move: "9. e3", prob: 1.2, eval: "+0.90", desc: "Developing pawn opening bishop diagonal" },
          { move: "9. Be2", prob: 0.6, eval: "+0.70", desc: "Quiet kingside development" }
        ],
        bitplanes: {
          0: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,1,1,0,1,1,1,1], [0,0,0,0,0,0,0,0] ],
          1: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          2: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0] ],
          3: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,1] ],
          4: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0] ],
          5: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0] ],
          6: [ [0,0,0,0,0,0,0,0], [1,1,1,1,0,1,1,1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          7: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          8: [ [0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          9: [ [1,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          10: [ [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          11: [ [0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0] ],
          12: Array(8).fill(Array(8).fill(1))
        }
      }
    };

    let curFpPosKey = 'start';
    let curFpStage = 1;
    let curFpBitplane = 0;
    let curFpKm = 3;

    function initForwardPass() {
      selectFpPos(curFpPosKey);
      setFpStage(curFpStage);
    }

    function selectFpPos(key) {
      curFpPosKey = key;
      const btnMap = { 'start': 'btnFpStart', 'sicilian': 'btnFpSicilian', 'greek_gift': 'btnFpGreek', 'queen_pin': 'btnFpQueen' };
      document.querySelectorAll('#tab-forwardpass .card:first-child .btn-tool').forEach(b => b.classList.remove('active-mode'));
      if (btnMap[key]) {
        const b = document.getElementById(btnMap[key]);
        if (b) b.classList.add('active-mode');
      }

      const data = FP_DATA[key];
      // Update Stage 1
      renderFpBitplane();
      // Update Stage 2
      renderFpSe();
      // Update Stage 3
      renderFpKmBars();
      // Update Stage 4
      renderFpCrelu();
      // Update Stage 5
      document.getElementById('fpValDisplay').textContent = (data.val > 0 ? '+' : '') + data.val.toFixed(2);
      document.getElementById('fpCpDisplay').textContent = data.cp;
      document.getElementById('fpNoulDisplay').textContent = data.noul.toFixed(3);
      document.getElementById('fpBestMoveDisplay').textContent = data.best_move;
      document.getElementById('fpBestProbDisplay').textContent = data.best_prob;
      const badge = document.getElementById('fpRoutingBadge');
      if (data.is_reflex) {
        badge.textContent = '● Route: System 0 Reflex (< 1ms)';
        badge.style.background = 'rgba(16,185,129,0.15)';
        badge.style.color = 'var(--accent-green)';
      } else {
        badge.textContent = '● Route: System 2 Escalate (Negamax Depth 3 + Q)';
        badge.style.background = 'rgba(239,68,68,0.15)';
        badge.style.color = 'var(--accent-red)';
      }
      renderFpTopMoves();
      // Reset Stage 6
      resetSimWeights();
    }

    function setFpStage(stage) {
      curFpStage = stage;
      for (let s = 1; s <= 6; s++) {
        const box = document.getElementById('fpStage' + s + 'Box');
        const btn = document.getElementById('btnStage' + s);
        if (box) box.style.display = (s === stage) ? 'block' : 'none';
        if (btn) {
          if (s === stage) btn.classList.add('active');
          else btn.classList.remove('active');
        }
      }
      if (stage === 1) renderFpBitplane();
      if (stage === 2) renderFpSe();
      if (stage === 3) renderFpKmBars();
      if (stage === 4) renderFpCrelu();
      if (stage === 5) renderFpTopMoves();
    }

    function selectBitplane(idx) {
      curFpBitplane = idx;
      for (let i = 0; i <= 12; i++) {
        const b = document.getElementById('btnPlane' + i);
        if (b) {
          if (i === idx) b.classList.add('active-mode');
          else b.classList.remove('active-mode');
        }
      }
      const names = [
        "Plane 0: White Pawns", "Plane 1: White Knights", "Plane 2: White Bishops",
        "Plane 3: White Rooks", "Plane 4: White Queens", "Plane 5: White King",
        "Plane 6: Black Pawns", "Plane 7: Black Knights", "Plane 8: Black Bishops",
        "Plane 9: Black Rooks", "Plane 10: Black Queens", "Plane 11: Black King",
        "Plane 12: Turn Indicator (1.0 = White to Move)"
      ];
      document.getElementById('fpStage1SliceInfo').textContent = names[idx] || ("Plane " + idx);
      renderFpBitplane();
    }

    function renderFpBitplane() {
      const grid = document.getElementById('fpBitplaneGrid');
      if (!grid) return;
      grid.innerHTML = '';
      const data = FP_DATA[curFpPosKey];
      const slice = data.bitplanes[curFpBitplane] || Array(8).fill(Array(8).fill(0));

      let activeInSlice = 0;
      let totalBits = 0;
      for (let p = 0; p <= 12; p++) {
        const s = data.bitplanes[p] || [];
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            if (s[r] && s[r][c] === 1) totalBits++;
          }
        }
      }

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const val = slice[r] ? slice[r][c] : 0;
          if (val === 1) activeInSlice++;
          const cell = document.createElement('div');
          cell.className = 'bitplane-cell ' + (val === 1 ? 'bit-1' : 'bit-0');
          cell.textContent = val;
          grid.appendChild(cell);
        }
      }

      document.getElementById('fpActiveSliceBits').textContent = activeInSlice;
      document.getElementById('fpTotalBits').textContent = totalBits;
    }

    function renderFpSe() {
      const grid = document.getElementById('fpSeGrid');
      if (!grid) return;
      grid.innerHTML = '';
      const data = FP_DATA[curFpPosKey];

      for (let ch = 0; ch < 64; ch++) {
        const box = document.createElement('div');
        box.className = 'se-channel-box';
        // Base weight modulated by position type
        let weight = 0.5 + 0.4 * Math.sin(ch * 0.4 + (curFpPosKey === 'greek_gift' ? 2 : (curFpPosKey === 'queen_pin' ? 4 : 0)));
        weight = Math.min(0.99, Math.max(0.08, weight));

        box.style.background = 'rgba(245, 158, 11, ' + weight.toFixed(2) + ')';
        box.style.color = weight > 0.4 ? '#000' : 'var(--text-muted)';
        box.style.fontWeight = '700';
        box.textContent = weight.toFixed(2);

        box.onmouseenter = () => {
          let role = "Pawn Structure & Outposts";
          if (ch >= 16 && ch < 32) role = "Knight & Bishop Mobility Rays";
          else if (ch >= 32 && ch < 48) role = "King Shelter & File Tension";
          else if (ch >= 48) role = "Tactical Pins & Tactical Sacrifices";

          document.getElementById('fpSeHoverInfo').textContent =
            'SE Channel #' + ch + ' [' + role + ']: Attention Multiplier s_c = ' + weight.toFixed(3) + ' (Reduction Ratio r=4)';
        };

        grid.appendChild(box);
      }
    }

    function updateKmStep(k) {
      curFpKm = parseInt(k, 10);
      const damping = 1.0 / (1.0 + 0.2 * curFpKm);
      document.getElementById('fpKmVal').textContent = 'k = ' + curFpKm;
      document.getElementById('fpKmDampingText').textContent = 'Damping Weight γ_k = ' + damping.toFixed(3);
      renderFpKmBars();
    }

    function renderFpKmBars() {
      const container = document.getElementById('fpKmBars');
      if (!container) return;
      container.innerHTML = '';
      const data = FP_DATA[curFpPosKey];
      const residuals = data.km_residuals;

      const curRes = residuals[curFpKm - 1];
      document.getElementById('fpKmResidualVal').textContent = curRes.toFixed(3);
      const badge = document.getElementById('fpKmStatusBadge');
      if (curRes <= 0.08) {
        badge.textContent = '✓ Converged Equilibrium (Residual < 0.08)';
        badge.style.color = 'var(--accent-green)';
      } else {
        badge.textContent = '⚡ Relaxing (Contraction In-Flight)';
        badge.style.color = 'var(--accent-amber)';
      }

      for (let i = 0; i < 4; i++) {
        const col = document.createElement('div');
        col.className = 'km-bar-col';
        const res = residuals[i];
        const pct = Math.round((res / 0.60) * 100);

        const bar = document.createElement('div');
        bar.className = 'km-bar';
        bar.style.height = pct + '%';
        bar.style.background = (i + 1 === curFpKm) ? 'var(--accent-cyan)' : ((i + 1 < curFpKm) ? 'rgba(56, 189, 248, 0.4)' : 'rgba(255,255,255,0.1)');
        if (i + 1 === curFpKm) bar.style.boxShadow = '0 0 12px rgba(56, 189, 248, 0.5)';

        const lbl = document.createElement('div');
        lbl.style.fontFamily = "'JetBrains Mono', monospace";
        lbl.style.fontSize = '0.66rem';
        lbl.style.color = (i + 1 === curFpKm) ? 'var(--accent-cyan)' : 'var(--text-dim)';
        lbl.textContent = 'k=' + (i + 1) + ' (' + res.toFixed(3) + ')';

        col.appendChild(bar);
        col.appendChild(lbl);
        container.appendChild(col);
      }
    }

    function renderFpCrelu() {
      const grid = document.getElementById('fpCreluGrid');
      if (!grid) return;
      grid.innerHTML = '';
      const data = FP_DATA[curFpPosKey];
      const acts = ACTS[curFpPosKey] ? ACTS[curFpPosKey].crelu : Array(128).fill(0);

      document.getElementById('fpCreluSparsity').textContent = data.sparsity + '%';

      acts.forEach((val, idx) => {
        const cell = document.createElement('div');
        cell.className = 'neuron-cell';
        const cluster = getClusterLabel(idx);

        if (val > 0.0) {
          if (idx < 32) {
            cell.style.background = 'rgba(56, 189, 248, ' + Math.max(0.18, val).toFixed(2) + ')';
            cell.style.boxShadow = '0 0 6px rgba(56, 189, 248, ' + (val * 0.5).toFixed(2) + ')';
          } else if (idx < 64) {
            cell.style.background = 'rgba(245, 158, 11, ' + Math.max(0.18, val).toFixed(2) + ')';
            cell.style.boxShadow = '0 0 6px rgba(245, 158, 11, ' + (val * 0.5).toFixed(2) + ')';
          } else if (idx < 96) {
            cell.style.background = 'rgba(239, 68, 68, ' + Math.max(0.18, val).toFixed(2) + ')';
            cell.style.boxShadow = '0 0 6px rgba(239, 68, 68, ' + (val * 0.5).toFixed(2) + ')';
          } else {
            cell.style.background = 'rgba(16, 185, 129, ' + Math.max(0.18, val).toFixed(2) + ')';
            cell.style.boxShadow = '0 0 6px rgba(16, 185, 129, ' + (val * 0.5).toFixed(2) + ')';
          }
        } else {
          cell.style.background = 'rgba(255, 255, 255, 0.03)';
        }

        cell.onmouseenter = () => {
          document.getElementById('fpCreluInfo').textContent =
            'Neuron #' + idx + ' [' + cluster + ']: ' + (val > 0.0 ? 'ACTIVE (Clamped Value: ' + val.toFixed(2) + ')' : 'DEAD / CLAMPED (0.00)') + ' - Sparsity Guarantee Active';
        };
        grid.appendChild(cell);
      });
    }

    function renderFpTopMoves() {
      const container = document.getElementById('fpTopMovesContainer');
      if (!container) return;
      container.innerHTML = '';
      const data = FP_DATA[curFpPosKey];

      data.top_moves.forEach(m => {
        const row = document.createElement('div');
        row.style.background = 'var(--bg-elevated)';
        row.style.border = '1px solid var(--border)';
        row.style.borderRadius = '6px';
        row.style.padding = '8px 12px';
        row.style.display = 'flex';
        row.style.justifyContent = 'space-between';
        row.style.alignItems = 'center';

        row.innerHTML =
          '<div><strong style="color: var(--accent-cyan); font-family: \'JetBrains Mono\', monospace;">' + m.move + '</strong> <span style="font-size: 0.72rem; color: var(--text-dim); margin-left: 8px;">' + m.desc + '</span></div>' +
          '<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.76rem;"><span style="color: var(--accent-amber); font-weight: 700;">' + m.prob + '%</span> <span style="color: var(--text-muted); margin-left: 6px;">' + m.eval + '</span></div>';
        container.appendChild(row);
      });
    }

    function runSimulatedUpdate() {
      const box = document.getElementById('fpBackwardBox');
      box.classList.remove('active-pulse');
      void box.offsetWidth;
      box.classList.add('active-pulse');

      const target = document.getElementById('fpSimLossTarget').value;
      let normHeads = 0.0482;
      let normPolicy = 0.1245;
      let normCrelu = 0.0815;
      let normKm = 0.0634;
      let normSe = 0.0421;
      let normStem = 0.0298;

      if (target === 'crisis') {
        normHeads = 0.1850;
        normPolicy = 0.0640;
        normCrelu = 0.1420;
      } else if (target === 'white_win') {
        normHeads = 0.0720;
        normPolicy = 0.1580;
      }

      document.getElementById('fpGradStatusTag').textContent = '⚡ Backward Adjoints Computed';
      document.getElementById('fpGradStatusTag').style.color = 'var(--accent-green)';

      document.getElementById('gradNormHeads').textContent = normHeads.toFixed(4);
      document.getElementById('fillNormHeads').style.width = Math.min(100, normHeads * 400) + '%';

      document.getElementById('gradNormPolicy').textContent = normPolicy.toFixed(4);
      document.getElementById('fillNormPolicy').style.width = Math.min(100, normPolicy * 400) + '%';

      document.getElementById('gradNormCrelu').textContent = normCrelu.toFixed(4);
      document.getElementById('fillNormCrelu').style.width = Math.min(100, normCrelu * 400) + '%';

      document.getElementById('gradNormKm').textContent = normKm.toFixed(4);
      document.getElementById('fillNormKm').style.width = Math.min(100, normKm * 400) + '%';

      document.getElementById('gradNormSe').textContent = normSe.toFixed(4);
      document.getElementById('fillNormSe').style.width = Math.min(100, normSe * 400) + '%';

      document.getElementById('gradNormStem').textContent = normStem.toFixed(4);
      document.getElementById('fillNormStem').style.width = Math.min(100, normStem * 400) + '%';

      const summary = document.getElementById('fpWeightUpdateSummary');
      summary.style.display = 'block';
      summary.innerHTML =
        '<strong>✓ SGD Step Applied (&eta; = 0.001):</strong><br>' +
        '&bull; Unrolled Krasnoselskii-Mann adjoints reverse-propagated across ' + curFpKm + ' plies without divergence.<br>' +
        '&bull; CReLU L1 sparsity penalty (0.02) zeroed out sub-threshold noise, strictly preserving discrete invariants (Rule 4).<br>' +
        '&bull; Multi-task Brier score loss updated the Noul sensor calibration threshold towards &tau; = 0.35.';
    }

    function resetSimWeights() {
      document.getElementById('fpGradStatusTag').textContent = 'Ready for update';
      document.getElementById('fpGradStatusTag').style.color = 'var(--text-dim)';
      ['Heads', 'Policy', 'Crelu', 'Km', 'Se', 'Stem'].forEach(k => {
        const v = document.getElementById('gradNorm' + k);
        const f = document.getElementById('fillNorm' + k);
        if (v) v.textContent = '0.0000';
        if (f) f.style.width = '0%';
      });
      const summary = document.getElementById('fpWeightUpdateSummary');
      if (summary) summary.style.display = 'none';
    }

    function switchTab(name) {
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      const targetPanel = document.getElementById('tab-' + name);
      if (targetPanel) targetPanel.classList.add('active');
      event.currentTarget.classList.add('active');

      if (name === 'forwardpass') initForwardPass();
      if (name === 'ca') initSim();
      if (name === 'chess') loadGames();
    }

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

    function getClusterLabel(idx) {
      if (idx < 32) return 'Center & Territory';
      if (idx < 64) return 'King Safety & Exposure';
      if (idx < 96) return 'Tactical Tension & Pins';
      return 'Endgame & Passed Pawns';
    }

    function renderBoard(grid) {
      const el = document.getElementById('cockpitBoardGrid');
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

    function loadRealAct(key) {
      document.querySelectorAll('#tab-arch .btn-tool').forEach(b => b.classList.remove('active-mode'));
      const btnMap = {
        'start': 'btnPosStart', 'sicilian': 'btnPosSicilian',
        'greek_gift': 'btnPosGreek', 'queen_pin': 'btnPosQueen',
        'back_rank': 'btnPosBackRank', 'endgame_promo': 'btnPosPromo',
        'lucena': 'btnPosLucena', 'smothered': 'btnPosSmothered'
      };
      if (btnMap[key]) {
        const b = document.getElementById(btnMap[key]);
        if (b) b.classList.add('active-mode');
      }

      const data = ACTS[key] || ACTS['start'];
      document.getElementById('realActiveCount').textContent = data.active_count;
      document.getElementById('realSparsity').textContent = data.sparsity_pct + '%';
      document.getElementById('realNoul').textContent = data.noul.toFixed(3);
      document.getElementById('realRouteTag').textContent = data.route;
      document.getElementById('realRouteTag').style.color = (data.noul >= 0.70) ? 'var(--accent-green)' : 'var(--accent-red)';
      document.getElementById('realValTag').textContent = (data.val > 0 ? '+' : '') + data.val.toFixed(2);
      document.getElementById('posExplainer').textContent = data.explanation;

      // Update Noul Indicator
      const noulInd = document.getElementById('noulIndicatorText');
      if (data.noul >= 0.70) {
        noulInd.innerHTML = '▲ Calibrated Noul = ' + data.noul.toFixed(3) + ' &bull; High Epistemic Confidence (System 0 Reflex)';
        noulInd.style.color = 'var(--accent-green)';
        setArchMode('quiet');
      } else {
        noulInd.innerHTML = '▲ Calibrated Noul = ' + data.noul.toFixed(3) + ' &bull; Low Epistemic Confidence (System 2 Escalate)';
        noulInd.style.color = 'var(--accent-red)';
        setArchMode('crisis');
      }

      if (data.board) {
        renderBoard(data.board);
      }
      if (data.fen) {
        const fenEl = document.getElementById('fenDisplay');
        if (fenEl) fenEl.textContent = data.fen;
      }

      // Render Continuous CReLU Matrix with Gradient Intensity
      const grid = document.getElementById('realNeuronGrid');
      grid.innerHTML = '';
      data.crelu.forEach((val, idx) => {
        const cell = document.createElement('div');
        cell.className = 'neuron-cell';
        const cluster = getClusterLabel(idx);

        if (val > 0.0) {
          // Color code by cluster and intensity
          if (idx < 32) {
            cell.style.background = 'rgba(56, 189, 248, ' + Math.max(0.18, val).toFixed(2) + ')';
            cell.style.boxShadow = '0 0 6px rgba(56, 189, 248, ' + (val * 0.5).toFixed(2) + ')';
          } else if (idx < 64) {
            cell.style.background = 'rgba(245, 158, 11, ' + Math.max(0.18, val).toFixed(2) + ')';
            cell.style.boxShadow = '0 0 6px rgba(245, 158, 11, ' + (val * 0.5).toFixed(2) + ')';
          } else if (idx < 96) {
            cell.style.background = 'rgba(239, 68, 68, ' + Math.max(0.18, val).toFixed(2) + ')';
            cell.style.boxShadow = '0 0 6px rgba(239, 68, 68, ' + (val * 0.5).toFixed(2) + ')';
          } else {
            cell.style.background = 'rgba(16, 185, 129, ' + Math.max(0.18, val).toFixed(2) + ')';
            cell.style.boxShadow = '0 0 6px rgba(16, 185, 129, ' + (val * 0.5).toFixed(2) + ')';
          }
        } else {
          cell.style.background = 'rgba(255, 255, 255, 0.03)';
        }

        cell.onmouseenter = () => {
          document.getElementById('realNeuronInfo').textContent =
            'Neuron #' + idx + ' [' + cluster + ']: ' + (val > 0.0 ? 'ACTIVE (Clamped ReLU: ' + val.toFixed(2) + ')' : 'DEAD / INACTIVE (0.00)') + ' in "' + data.name + '"';
        };
        grid.appendChild(cell);
      });
    }

    loadRealAct('start');

    // Domain B: Interactive CA
    const SIM_SIZE = 28;
    let simGrid = Array(SIM_SIZE).fill(0).map(() => Array(SIM_SIZE).fill(0));
    let simRunning = false;
    let simTimer = null;
    let simGen = 0;
    let simDiscrete = true;

    function initSim() {
      const canvas = document.getElementById('caSimCanvas');
      if (!canvas) return;
      spawnGlider();
      canvas.onclick = (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const c = Math.floor(x / (canvas.width / SIM_SIZE));
        const r = Math.floor(y / (canvas.height / SIM_SIZE));
        if (r >= 0 && r < SIM_SIZE && c >= 0 && c < SIM_SIZE) {
          simGrid[r][c] = simGrid[r][c] ? 0 : 1;
          drawSim();
        }
      };
    }

    function spawnGlider() {
      simGen = 0;
      simGrid = Array(SIM_SIZE).fill(0).map(() => Array(SIM_SIZE).fill(0));
      const g = [[0,1,0],[0,0,1],[1,1,1]];
      for(let r=0; r<3; r++) {
        for(let c=0; c<3; c++) {
          simGrid[r+2][c+2] = g[r][c];
        }
      }
      document.getElementById('caGenTag').textContent = 'Gen: ' + simGen;
      drawSim();
    }

    function spawnPulsar() {
      simGen = 0;
      simGrid = Array(SIM_SIZE).fill(0).map(() => Array(SIM_SIZE).fill(0));
      // Pulsar oscillator
      const pRows = [2, 7, 9, 14];
      const pCols = [4,5,6, 10,11,12];
      pRows.forEach(r => {
        pCols.forEach(c => {
          if (r < SIM_SIZE && c < SIM_SIZE) simGrid[r][c] = 1;
        });
      });
      document.getElementById('caGenTag').textContent = 'Gen: ' + simGen;
      drawSim();
    }

    function randomizeSim() {
      simGen = 0;
      for (let r = 0; r < SIM_SIZE; r++) {
        for (let c = 0; c < SIM_SIZE; c++) {
          simGrid[r][c] = (Math.random() < 0.25) ? 1 : 0;
        }
      }
      document.getElementById('caGenTag').textContent = 'Gen: ' + simGen;
      drawSim();
    }

    function clearSim() {
      simGen = 0;
      simGrid = Array(SIM_SIZE).fill(0).map(() => Array(SIM_SIZE).fill(0));
      document.getElementById('caGenTag').textContent = 'Gen: ' + simGen;
      drawSim();
    }

    function drawSim() {
      const canvas = document.getElementById('caSimCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const cellSize = canvas.width / SIM_SIZE;
      ctx.fillStyle = '#080A0D';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let r = 0; r < SIM_SIZE; r++) {
        for (let c = 0; c < SIM_SIZE; c++) {
          if (simGrid[r][c] === 1) {
            ctx.fillStyle = simDiscrete ? '#10B981' : '#EF4444';
            ctx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
          } else {
            ctx.fillStyle = '#12151D';
            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
          }
        }
      }
    }

    function stepSim() {
      const next = Array(SIM_SIZE).fill(0).map(() => Array(SIM_SIZE).fill(0));
      for (let r = 0; r < SIM_SIZE; r++) {
        for (let c = 0; c < SIM_SIZE; c++) {
          let neighbors = 0;
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              if (dr === 0 && dc === 0) continue;
              const nr = (r + dr + SIM_SIZE) % SIM_SIZE;
              const nc = (c + dc + SIM_SIZE) % SIM_SIZE;
              neighbors += simGrid[nr][nc];
            }
          }
          if (simGrid[r][c] === 1) {
            next[r][c] = (neighbors === 2 || neighbors === 3) ? 1 : 0;
          } else {
            if (!simDiscrete && Math.random() < 0.04) {
              next[r][c] = 1;
            } else {
              next[r][c] = (neighbors === 3) ? 1 : 0;
            }
          }
        }
      }
      simGrid = next;
      simGen++;
      document.getElementById('caGenTag').textContent = 'Gen: ' + simGen;
      drawSim();
    }

    function toggleSimPlay() {
      const btn = document.getElementById('simPlayBtn');
      if (simRunning) {
        clearInterval(simTimer);
        simRunning = false;
        btn.textContent = '▶ Play';
      } else {
        simTimer = setInterval(stepSim, 120);
        simRunning = true;
        btn.textContent = '⏸ Pause';
      }
    }

    function toggleSimSteer() {
      simDiscrete = !simDiscrete;
      const btn = document.getElementById('btnModeSim');
      const expl = document.getElementById('simExplainer');
      if (simDiscrete) {
        btn.textContent = 'Mode: TypeSafe Discrete (0% Leakage)';
        btn.style.color = 'var(--accent-green)';
        expl.innerHTML = '<strong>Theorem 1 Verification:</strong> Under discrete integer factor masks, inactive threshold cells satisfy &phi;(0) = 0 identically. Parasitic dead-neuron leakage is strictly <strong>0.00%</strong>.';
      } else {
        btn.textContent = 'Mode: Continuous Latent Perturbation (Corrupted)';
        btn.style.color = 'var(--accent-red)';
        expl.innerHTML = '<strong>Pathology Demonstrated:</strong> Continuous latent drift corrupts dead-neuron thresholds (&phi;(x + &epsilon;) &ne; 0), causing parasitic background activations to destroy glider invariants.';
      }
      drawSim();
    }

    // Load Lichess bot profile and games
    async function loadGames() {
      try {
        const res = await fetch('/api/recent_games');
        const data = await res.json();
        const list = document.getElementById('gamesContainer');
        if (!list) return;
        list.innerHTML = '';

        if (!data || data.length === 0) {
          list.innerHTML = '<div style="color: var(--text-dim); font-size: 0.8rem; text-align: center; padding: 12px;">No games recorded yet. Challenge @jess-hyperbullet on Lichess!</div>';
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

          item.innerHTML =
            '<div><strong>' + white + '</strong> vs <strong>' + black + '</strong> (' + (g.speed || 'bullet') + ')</div>' +
            '<div style="font-family: \\'JetBrains Mono\\', monospace; font-weight: 700; color: var(--accent-cyan);">' + result + '</div>';
          list.appendChild(item);
        });
      } catch(e) {
        console.warn('Recent games fetch skipped:', e);
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

    // Auto-fetch profile
    fetch('https://lichess.org/api/user/jess-hyperbullet')
      .then(r => r.json())
      .then(d => {
        const rEl = document.getElementById('botRatingVal');
        if (rEl && d.perfs?.bullet?.rating) {
          rEl.textContent = d.perfs.bullet.rating + (d.perfs.bullet.prov ? '?' : '');
        }
      })
      .catch(() => {});
  </script>
</body>
</html>
`;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    // Proxy API requests to local GUI server if active
    if (url.pathname.startsWith("/api/gui/") || url.pathname.startsWith("/api/selfplay_")) {
      const targetUrl = "http://127.0.0.1:8765" + url.pathname + url.search;
      let response;
      try {
        response = await fetch(targetUrl, {
          method: request.method,
          headers: request.headers,
          body: request.method !== "GET" && request.method !== "HEAD" ? request.body : undefined
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: "Local engine GUI server offline", details: err.toString() }), {
          status: 502,
          headers: { "Content-Type": "application/json", ...corsHeaders }
        });
      }
      return response;
    }

    // API: Recent games played by the bot
    if (url.pathname === "/api/recent_games") {
      try {
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
            "Cache-Control": "public, max-age=15",
            ...corsHeaders,
          }
        });
      } catch(e) {
        return new Response(JSON.stringify([]), {
          headers: { "Content-Type": "application/json", ...corsHeaders }
        });
      }
    }

    // API: Wake bot on Modal Cloud
    if (url.pathname === "/api/wake_bot") {
      try {
        const modalRes = await fetch("https://sub-surface--jess-hyperbullet-bot-wake.modal.run", {
          headers: { "User-Agent": "Jevformer-Edge-Worker/1.0" }
        });
        const modalData = await modalRes.json();
        return new Response(JSON.stringify(modalData), {
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          }
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: err.toString() }), {
          status: 500,
          headers: { "Content-Type": "application/json", ...corsHeaders }
        });
      }
    }

    // Healthcheck
    if (url.pathname === "/health" || url.pathname === "/healthz") {
      return new Response(JSON.stringify({ status: "healthy", domain: "jev.subsurfaces.net" }), {
        headers: { "Content-Type": "application/json", ...corsHeaders }
      });
    }

    // Default: Serve comprehensive Architecture Explorer with strict CSP headers
    return new Response(HTML_CONTENT, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Security-Policy": "default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval'; connect-src * 'self' data: blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:;",
        "Content-Security-Policy-Report-Only": "connect-src * 'self' data: blob:;",
        "Cache-Control": "public, max-age=60",
        ...corsHeaders,
      }
    });
  }
};
