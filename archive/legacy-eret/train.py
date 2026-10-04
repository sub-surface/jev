"""
=============================================================================
⚡ JEV / ERET TRAINING ENTRYPOINT
=============================================================================
Trains the champion ERET (Epistemic Recurrent Equilibrium Transformer)
model either locally on GPU/CPU or on serverless Modal H100 cloud.

Usage:
    python train.py                      # Train tactical curriculum locally
    python train.py --epochs 20          # Train local curriculum for 20 epochs
    python train.py --modal              # Launch scaled H100 self-play training on Modal
=============================================================================
"""

import os
import sys
import argparse
import subprocess

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev-vault", "src")
sys.path.insert(0, SRC_DIR)


def train_local(epochs: int, batch_size: int, lr: float, out_path: str):
    import torch
    from train_eret_tactical_curriculum import train_curriculum

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Launching Local ERET Training | Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    train_curriculum(epochs=epochs, lr=lr, out_path=out_path)


def launch_modal_h100(epochs: int, games_per_epoch: int):
    print(f"🚀 Dispatching Scaled Training to Modal Cloud (NVIDIA H100 SXM5)...")
    modal_script = os.path.join("modal_cloud", "modal_h100_hybrid_scale.py")
    cmd = [sys.executable, "-m", "modal", "run", modal_script]
    print(f"   Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="JEV / ERET Training Entrypoint")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for local training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--out", type=str, default="data/jev_champion.pt", help="Output path for weights")
    parser.add_argument("--modal", action="store_true", help="Launch scaled training on Modal H100 SXM5")
    parser.add_argument("--games", type=int, default=1000, help="Games per epoch if running on Modal")
    args = parser.parse_args()

    if args.modal:
        launch_modal_h100(epochs=args.epochs, games_per_epoch=args.games)
    else:
        train_local(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_path=args.out)


if __name__ == "__main__":
    main()
