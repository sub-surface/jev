"""
=============================================================================
Download Champion Checkpoint from Modal Volume
=============================================================================
Pulls the trained LoRA adapter + scorer + temperatures from the Modal Volume
to the local checkpoints directory for inference.

Usage:
  python decision_model/scripts/download_checkpoint.py
  python decision_model/scripts/download_checkpoint.py --target decision_model/checkpoints
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download Jev champion checkpoint from Modal Volume")
    parser.add_argument("--target", type=str, default="decision_model/checkpoints",
                        help="Local directory to download into")
    parser.add_argument("--volume", type=str, default="jev-model-artifacts",
                        help="Modal Volume name")
    parser.add_argument("--list-only", action="store_true",
                        help="Just list volume contents without downloading")
    args = parser.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    if args.list_only:
        print(f"Listing contents of Modal Volume '{args.volume}'...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "ls", args.volume, "/artifacts/"],
            check=False,
        )
        return

    print(f"Downloading from Modal Volume '{args.volume}' to {target}/", flush=True)
    print("=" * 60, flush=True)

    # 1. Download champion .pt checkpoint
    print("\n[1/2] Downloading jev_champion_latest.pt...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get",
         args.volume, "/artifacts/jev_champion_latest.pt",
         str(target / "jev_champion_latest.pt")],
        check=False,
    )
    if result.returncode == 0:
        size = (target / "jev_champion_latest.pt").stat().st_size / (1024 * 1024)
        print(f"  OK ({size:.1f} MB)", flush=True)
    else:
        print("  FAILED - checkpoint may not exist yet", flush=True)

    # 2. Download LoRA adapter directory
    print("\n[2/2] Downloading LoRA adapter directory...", flush=True)
    adapter_dir = target / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get",
         args.volume, "/artifacts/lora_adapter",
         str(adapter_dir)],
        check=False,
    )
    if result.returncode == 0:
        files = list(adapter_dir.rglob("*"))
        print(f"  OK ({len(files)} files)", flush=True)
    else:
        print("  FAILED or adapter directory doesn't exist", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Download complete. Run inference with:", flush=True)
    print(f"  python decision_model/scripts/inference_cli.py --checkpoint-dir {target}", flush=True)


if __name__ == "__main__":
    main()
