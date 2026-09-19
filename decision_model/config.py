"""
=============================================================================
Decision Model — Central Configuration
=============================================================================
All hyperparameters, paths, and budget tracking in one place.
"""
import os
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DECISION_MODEL_ROOT = Path(__file__).parent.resolve()
DATA_DIR = DECISION_MODEL_ROOT / "data"
CHECKPOINTS_DIR = DECISION_MODEL_ROOT / "checkpoints"
LOGS_DIR = DECISION_MODEL_ROOT / "logs"

# Ensure dirs exist
for d in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    max_examples_per_task: int = 30_000
    train_family_fraction: float = 0.80  # 80% task families for training
    in_task_val_fraction: float = 0.10   # 10% of training tasks for validation
    min_examples_per_task: int = 50      # Skip tasks with fewer examples
    max_seq_length: int = 512            # Token sequence length limit
    num_score_bins: int = 10             # Discretization bins for score primitive
    seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    base_model: str = "Qwen/Qwen2.5-0.5B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    num_option_markers: int = 26         # [OPT_A] through [OPT_Z]
    decision_head_hidden: int = 256      # Hidden dim for decision projection


@dataclass
class RLCDConfig:
    """RLCD training configuration."""
    # Stage 1: CE warmup
    ce_warmup_epochs: int = 2
    ce_warmup_lr: float = 2e-4
    ce_warmup_batch_size: int = 4

    # Stage 2: Proper scoring rule RL
    rlcd_epochs: int = 3
    rlcd_lr: float = 5e-5
    rlcd_batch_size: int = 4
    exploration_sigma_start: float = 0.1
    exploration_sigma_end: float = 0.01

    # Scoring rule weights & formulation
    use_brier_rlcr: bool = True        # MIT ICLR 2026 bounded Brier formulation (Theorem 1)
    brier_weight: float = 1.0
    log_score_weight: float = 1.0
    spherical_score_weight: float = 0.5
    rps_weight: float = 0.5            # Only for score primitive

    # Stage 3: Temperature scaling
    temp_scale_buckets: list = field(
        default_factory=lambda: [2, 5, 10, 999]  # [2], [3-5], [6-10], [11+]
    )


@dataclass
class BudgetConfig:
    """Modal compute budget tracking."""
    total_budget: float = 27.89
    eval_reserve_fraction: float = 0.20  # Reserve 20% for eval
    max_single_run_fraction: float = 0.33  # Max 1/3 of remaining per run
    budget_log_file: str = "budget_log.jsonl"

    @property
    def eval_reserve(self) -> float:
        return self.total_budget * self.eval_reserve_fraction

    @property
    def training_budget(self) -> float:
        return self.total_budget - self.eval_reserve


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rlcd: RLCDConfig = field(default_factory=RLCDConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    seed: int = 42
    experiment_name: str = "jev_decision_v1"

    def config_hash(self) -> str:
        """SHA256 hash of the full config for reproducibility."""
        config_str = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def save(self, path: Optional[Path] = None):
        """Persist config to JSON."""
        path = path or (LOGS_DIR / f"config_{self.experiment_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            data=DataConfig(**data["data"]),
            model=ModelConfig(**data["model"]),
            rlcd=RLCDConfig(**data["rlcd"]),
            budget=BudgetConfig(**data["budget"]),
            seed=data["seed"],
            experiment_name=data["experiment_name"],
        )


def log_budget_spend(
    run_name: str,
    gpu_type: str,
    duration_seconds: float,
    cost_usd: float,
    commit_hash: str = "",
    config_hash: str = "",
    notes: str = "",
):
    """Append a spend record to the budget log."""
    import datetime
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_name": run_name,
        "gpu_type": gpu_type,
        "duration_seconds": duration_seconds,
        "cost_usd": cost_usd,
        "commit_hash": commit_hash,
        "config_hash": config_hash,
        "notes": notes,
    }
    log_path = LOGS_DIR / "budget_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_cumulative_spend() -> float:
    """Read cumulative Modal spend from the budget log."""
    log_path = LOGS_DIR / "budget_log.jsonl"
    if not log_path.exists():
        return 0.0
    total = 0.0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                total += record.get("cost_usd", 0.0)
    return total
