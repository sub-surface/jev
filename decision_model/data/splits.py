"""
=============================================================================
Task-Family Splits — Zero-shot generalization evaluation
=============================================================================
Splits held-out data at the TASK-FAMILY level, not just the example level.
This ensures zero-shot generalization is measured honestly: entire task families
are reserved that the model never sees during training.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from decision_model.data.task_registry import TaskFamily, TaskSpec
from decision_model.data.tasksource_loader import JevExample

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """A complete train/eval data split with task-family-level holdouts."""
    # Training data (examples from training-family tasks)
    train_examples: list[JevExample]
    # In-task validation (held-out examples from training-family tasks)
    in_task_val_examples: list[JevExample]
    # Zero-shot family validation (ALL examples from held-out task families)
    zero_shot_examples: list[JevExample]

    # Metadata
    train_task_ids: set[str]
    zero_shot_task_ids: set[str]
    train_families: set[TaskFamily]
    zero_shot_families: set[TaskFamily]

    def summary(self) -> str:
        lines = [
            f"=== Data Split Summary ===",
            f"Training tasks:     {len(self.train_task_ids)} "
            f"({len(self.train_families)} families)",
            f"Zero-shot tasks:    {len(self.zero_shot_task_ids)} "
            f"({len(self.zero_shot_families)} families)",
            f"Train examples:     {len(self.train_examples):,}",
            f"In-task val:        {len(self.in_task_val_examples):,}",
            f"Zero-shot eval:     {len(self.zero_shot_examples):,}",
            f"",
            f"Training families:  {sorted(f.value for f in self.train_families)}",
            f"Zero-shot families: {sorted(f.value for f in self.zero_shot_families)}",
        ]
        return "\n".join(lines)


def create_task_family_split(
    task_specs: list[TaskSpec],
    task_examples: dict[str, list[JevExample]],
    train_family_fraction: float = 0.80,
    in_task_val_fraction: float = 0.10,
    seed: int = 42,
) -> DataSplit:
    """
    Create a train/eval split at the task-family level.

    1. Group tasks by family
    2. Hold out ~20% of families entirely (zero-shot eval)
    3. Within training families, hold out 10% of examples (in-task eval)

    This gives two separate evaluation axes:
      - In-task: tasks the model trained on (different examples)
      - Zero-shot: entire task families never seen during training
    """
    rng = np.random.RandomState(seed)

    # Group tasks by family
    family_to_tasks: dict[TaskFamily, list[str]] = defaultdict(list)
    for spec in task_specs:
        family_to_tasks[spec.family].append(spec.task_id)

    families = sorted(family_to_tasks.keys(), key=lambda f: f.value)
    n_families = len(families)

    if n_families < 3:
        logger.warning(
            f"Only {n_families} families found. Cannot do meaningful family-level split. "
            f"Falling back to random task-level split."
        )
        return _fallback_task_split(task_specs, task_examples, in_task_val_fraction, seed)

    # Determine how many families to hold out
    n_holdout = max(1, int(n_families * (1 - train_family_fraction)))
    n_train = n_families - n_holdout

    # Shuffle and split families
    family_indices = rng.permutation(n_families)
    train_family_indices = family_indices[:n_train]
    holdout_family_indices = family_indices[n_train:]

    train_families = {families[i] for i in train_family_indices}
    zero_shot_families = {families[i] for i in holdout_family_indices}

    # Collect task IDs
    train_task_ids = set()
    zero_shot_task_ids = set()
    for fam in train_families:
        train_task_ids.update(family_to_tasks[fam])
    for fam in zero_shot_families:
        zero_shot_task_ids.update(family_to_tasks[fam])

    # Build example lists
    train_examples = []
    in_task_val_examples = []
    zero_shot_examples = []

    for task_id, examples in task_examples.items():
        if task_id in zero_shot_task_ids:
            # Entire task goes to zero-shot eval
            zero_shot_examples.extend(examples)
        elif task_id in train_task_ids:
            # Split within task: 90% train, 10% in-task val
            n = len(examples)
            indices = rng.permutation(n)
            n_val = max(1, int(n * in_task_val_fraction))
            val_indices = set(indices[:n_val])

            for i, ex in enumerate(examples):
                if i in val_indices:
                    in_task_val_examples.append(ex)
                else:
                    train_examples.append(ex)

    split = DataSplit(
        train_examples=train_examples,
        in_task_val_examples=in_task_val_examples,
        zero_shot_examples=zero_shot_examples,
        train_task_ids=train_task_ids,
        zero_shot_task_ids=zero_shot_task_ids,
        train_families=train_families,
        zero_shot_families=zero_shot_families,
    )

    print(split.summary(), flush=True)
    return split


def _fallback_task_split(
    task_specs: list[TaskSpec],
    task_examples: dict[str, list[JevExample]],
    val_fraction: float,
    seed: int,
) -> DataSplit:
    """Fallback: random task-level split when too few families exist."""
    rng = np.random.RandomState(seed)
    task_ids = sorted(task_examples.keys())
    n = len(task_ids)
    n_holdout = max(1, int(n * 0.20))

    indices = rng.permutation(n)
    holdout_ids = {task_ids[i] for i in indices[:n_holdout]}
    train_ids = {task_ids[i] for i in indices[n_holdout:]}

    train_examples = []
    val_examples = []
    zs_examples = []

    for tid, exs in task_examples.items():
        if tid in holdout_ids:
            zs_examples.extend(exs)
        else:
            n_ex = len(exs)
            perm = rng.permutation(n_ex)
            n_val = max(1, int(n_ex * val_fraction))
            val_set = set(perm[:n_val])
            for i, ex in enumerate(exs):
                if i in val_set:
                    val_examples.append(ex)
                else:
                    train_examples.append(ex)

    # Infer families
    spec_map = {s.task_id: s for s in task_specs}
    train_fams = {spec_map[t].family for t in train_ids if t in spec_map}
    zs_fams = {spec_map[t].family for t in holdout_ids if t in spec_map}

    return DataSplit(
        train_examples=train_examples,
        in_task_val_examples=val_examples,
        zero_shot_examples=zs_examples,
        train_task_ids=train_ids,
        zero_shot_task_ids=holdout_ids,
        train_families=train_fams,
        zero_shot_families=zs_fams,
    )


def save_split_metadata(split: DataSplit, path: Path):
    """Save split metadata for reproducibility."""
    meta = {
        "train_task_ids": sorted(split.train_task_ids),
        "zero_shot_task_ids": sorted(split.zero_shot_task_ids),
        "train_families": sorted(f.value for f in split.train_families),
        "zero_shot_families": sorted(f.value for f in split.zero_shot_families),
        "n_train": len(split.train_examples),
        "n_in_task_val": len(split.in_task_val_examples),
        "n_zero_shot": len(split.zero_shot_examples),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Split metadata saved to {path}", flush=True)
