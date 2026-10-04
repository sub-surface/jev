"""
=============================================================================
Reproducibility & Strict Verification Harness
=============================================================================
Hygiene protocols matching OpenJev v2:
  1. Strict Checkpoint Loading: checks schema, state_dict keys, and SHA256 hashes.
     Never falls back silently to random initialization.
  2. Configuration & Weights Hashing: SHA256 fingerprints recorded for all runs.
  3. Environment Snapshotting: Python, PyTorch, CUDA, and pip freeze captured.
  4. Deterministic Seed Sealing: sets all PRNG seeds and CUDA determinism flags.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import random
import hashlib
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
import torch


def set_reproducible_seed(seed: int = 42):
    """Seal all random number generators deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_hash(d: dict) -> str:
    """Compute deterministic SHA256 hash of a dictionary."""
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def get_git_revision() -> tuple[str, bool]:
    """Return (commit_hash, is_dirty)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", False


def strict_load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    expected_hash: Optional[str] = None,
) -> dict:
    """
    Strictly load a checkpoint file.
    Raises ValueError on any missing keys, shape mismatches, or file corruption.
    NO silent fallback.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strict checkpoint loader failed: file does not exist at {path}")

    # Check hash if provided
    if expected_hash:
        actual_hash = compute_file_hash(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checkpoint SHA256 mismatch! Expected: {expected_hash}, Actual: {actual_hash}"
            )

    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Corrupt checkpoint: expected dict root, got {type(ckpt)}")

    if "scorer_state_dict" not in ckpt:
        raise KeyError("Strict loader error: 'scorer_state_dict' missing from checkpoint root")

    # Strict state dict loading on the decision scorer
    model.scorer.load_state_dict(ckpt["scorer_state_dict"], strict=True)
    print(f" Strict verification passed. Checkpoint successfully loaded from {path}", flush=True)
    return ckpt


def snapshot_environment(output_dir: Path) -> dict:
    """Capture complete reproducible environment snapshot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    commit, is_dirty = get_git_revision()

    env_info = {
        "git_commit": commit,
        "git_dirty": is_dirty,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "gpu_vram_gb": (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0.0,
    }

    with open(output_dir / "env_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    # Freeze pip dependencies
    try:
        reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
        with open(output_dir / "requirements_frozen.txt", "w", encoding="utf-8") as f:
            f.write(reqs)
    except Exception as e:
        print(f"Warning capturing pip freeze: {e}", flush=True)

    return env_info
