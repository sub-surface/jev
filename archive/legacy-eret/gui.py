"""
=============================================================================
⚡ JEV / ERET INTERACTIVE GUI ENTRYPOINT
=============================================================================
Launches the local interactive web GUI on http://127.0.0.1:8765.
Features real-time 13-bitplane visualization, Krasnoselskii-Mann latent state,
CReLU accumulator sparsity meters, and Stockfish 19 analysis.

Usage:
    python gui.py                        # Launch GUI server on port 8765
    python gui.py --port 9000            # Launch on custom port
=============================================================================
"""

import os
import sys
import argparse
import subprocess

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="JEV / ERET Interactive GUI Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind server (default: 8765)")
    args = parser.parse_args()

    server_script = os.path.join("jev-vault", "src", "chess_gui_server.py")
    cmd = [sys.executable, "-u", server_script]
    print(f"⚡ Starting JEV Interactive GUI Server on http://127.0.0.1:{args.port}...")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
