import os
import sys
import subprocess

# Ensure UTF-8 output encoding across Windows streams
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "modal", "run", "modal_epistemic_search_scaled.py"]
    print("Launching Modal scaling benchmark via run_modal.py...", flush=True)
    res = subprocess.run(cmd, env=os.environ)
    sys.exit(res.returncode)
