"""
=============================================================================
Tasksource Loader — Load & map tasksource tasks to Jev primitives
=============================================================================
Loads tasks from the tasksource collection (Sileo, 2023) and maps them to
the three Jev decision primitives: choice, noul, score.

Uses tasksource-instruct-v0 as the primary dataset (485 tasks, 30k cap,
5.3M examples), with fallback to direct dataset loading for tasks that
need custom handling.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from decision_model.data.task_registry import (
    JevPrimitive,
    TaskFamily,
    TaskSpec,
    infer_family,
    infer_primitive,
    check_license,
)

logger = logging.getLogger(__name__)


@dataclass
class JevExample:
    """A single training/eval example mapped to a Jev primitive."""
    task_id: str
    primitive: JevPrimitive
    family: TaskFamily

    # The prompt/context presented to the model
    instruction: str

    # For choice: list of option strings; for noul: ["True", "False"]; for score: bin labels
    options: list[str]

    # Ground truth index (into options) or float value (for score before binning)
    label: int

    # Number of options (for cardinality-bucket temperature scaling)
    num_options: int

    # Whether this is ordinal (for RPS computation in score tasks)
    is_ordinal: bool = False


def load_tasksource_instruct(
    max_tasks: Optional[int] = None,
    max_examples_per_task: int = 30_000,
    min_examples_per_task: int = 50,
    seed: int = 42,
) -> tuple[list[TaskSpec], dict[str, list[JevExample]]]:
    """
    Load tasksource-instruct-v0 from HuggingFace and map to Jev primitives.

    Returns:
        task_specs: List of TaskSpec metadata for each loaded task
        task_examples: Dict mapping task_id -> list of JevExample
    """
    from datasets import load_dataset

    logger.info("Loading tasksource-instruct-v0 from HuggingFace...")
    print("Loading tasksource-instruct-v0 from HuggingFace...", flush=True)

    # tasksource-instruct-v0 is a single dataset with a 'task' column identifying each task
    ds = load_dataset("sileod/tasksource-instruct-v0", split="train", trust_remote_code=True)

    # Group by task
    task_groups: dict[str, list] = {}
    for i, example in enumerate(ds):
        task_name = example.get("task", example.get("dataset", f"unknown_{i}"))
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append(example)

    print(f"Found {len(task_groups)} unique tasks in tasksource-instruct-v0", flush=True)

    if max_tasks is not None:
        # Take a deterministic subset for prototyping
        rng = np.random.RandomState(seed)
        task_names = sorted(task_groups.keys())
        selected = rng.choice(task_names, min(max_tasks, len(task_names)), replace=False)
        task_groups = {k: task_groups[k] for k in selected}
        print(f"Selected {len(task_groups)} tasks for prototyping", flush=True)

    task_specs: list[TaskSpec] = []
    task_examples: dict[str, list[JevExample]] = {}

    for task_name, raw_examples in task_groups.items():
        # Skip tiny tasks
        if len(raw_examples) < min_examples_per_task:
            logger.debug(f"Skipping {task_name}: only {len(raw_examples)} examples")
            continue

        # Cap per task
        if len(raw_examples) > max_examples_per_task:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(raw_examples), max_examples_per_task, replace=False)
            raw_examples = [raw_examples[i] for i in indices]

        # Analyze task to determine primitive
        mapped = _map_instruct_examples(task_name, raw_examples, seed)
        if mapped is None or len(mapped) == 0:
            continue

        # Create task spec
        first = mapped[0]
        spec = TaskSpec(
            task_id=task_name,
            primitive=first.primitive,
            family=first.family,
            num_options=first.num_options,
            is_ordinal=first.is_ordinal,
        )
        task_specs.append(spec)
        task_examples[task_name] = mapped

        print(
            f"  [{spec.primitive.value:6s}] {task_name}: "
            f"{len(mapped)} examples, family={spec.family.value}, "
            f"options={spec.num_options}",
            flush=True,
        )

    print(
        f"\nLoaded {len(task_specs)} tasks, "
        f"{sum(len(v) for v in task_examples.values())} total examples",
        flush=True,
    )
    return task_specs, task_examples


def _map_instruct_examples(
    task_name: str,
    raw_examples: list[dict],
    seed: int,
) -> Optional[list[JevExample]]:
    """
    Map raw tasksource-instruct-v0 examples to JevExamples.

    tasksource-instruct-v0 format has:
      - 'inputs': the instruction/context text
      - 'targets': the correct answer text
      - 'task': task name
      - Optionally 'options' for multiple choice
    """
    family = infer_family(task_name)
    mapped = []

    for ex in raw_examples:
        instruction = ex.get("inputs", "")
        target = ex.get("targets", "")

        if not instruction or not target:
            continue

        # Detect if this is a multiple-choice task (options in instruction)
        options = _extract_options(instruction, ex)

        if options and len(options) >= 2:
            # Multiple choice → choice primitive
            label_idx = _find_label_in_options(target, options)
            if label_idx is None:
                continue  # Can't match target to an option

            if len(options) == 2 and _is_binary_task(options, task_name):
                # Binary → noul
                mapped.append(JevExample(
                    task_id=task_name,
                    primitive=JevPrimitive.NOUL,
                    family=family,
                    instruction=instruction,
                    options=options,
                    label=label_idx,
                    num_options=2,
                    is_ordinal=False,
                ))
            else:
                mapped.append(JevExample(
                    task_id=task_name,
                    primitive=JevPrimitive.CHOICE,
                    family=family,
                    instruction=instruction,
                    options=options,
                    label=label_idx,
                    num_options=len(options),
                    is_ordinal=False,
                ))
        else:
            # Classification without explicit options → try to build option set
            # from observed targets across all examples
            # (handled at the batch level, skip for now if no options found)
            continue

    return mapped if mapped else None


def _extract_options(instruction: str, example: dict) -> Optional[list[str]]:
    """
    Extract options from a tasksource-instruct-v0 example.

    Options can appear as:
      1. Explicit 'options' field in the example dict
      2. Lettered choices in the instruction text (A. ... B. ... C. ...)
      3. Enumerated choices (1. ... 2. ... 3. ...)
    """
    # Check for explicit options field
    if "options" in example and example["options"]:
        opts = example["options"]
        if isinstance(opts, list) and len(opts) >= 2:
            return [str(o).strip() for o in opts]
        if isinstance(opts, str):
            # Sometimes options are newline-separated
            parts = [p.strip() for p in opts.split("\n") if p.strip()]
            if len(parts) >= 2:
                return parts

    # Try to extract from instruction text patterns
    import re

    # Pattern: "- option1\n- option2\n..."
    dash_pattern = re.findall(r"^[-*]\s+(.+)$", instruction, re.MULTILINE)
    if len(dash_pattern) >= 2:
        return dash_pattern

    # Pattern: "A) option1 B) option2" or "A. option1 B. option2"
    letter_pattern = re.findall(r"[A-Z][.)]\s*([^\n]+?)(?=\s+[A-Z][.)]|\s*$)", instruction)
    if len(letter_pattern) >= 2:
        return [o.strip() for o in letter_pattern]

    # Pattern: "1. option1 2. option2"
    num_pattern = re.findall(r"\d+[.)]\s*([^\n]+?)(?=\s+\d+[.)]|\s*$)", instruction)
    if len(num_pattern) >= 2:
        return [o.strip() for o in num_pattern]

    # Pattern: Options listed after "Options:" or "Choices:"
    options_section = re.search(
        r"(?:options|choices|answers?)\s*:\s*(.+)",
        instruction,
        re.IGNORECASE | re.DOTALL,
    )
    if options_section:
        text = options_section.group(1)
        # Try comma-separated
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) >= 2:
            return parts
        # Try newline-separated
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        if len(parts) >= 2:
            return parts

    return None


def _find_label_in_options(target: str, options: list[str]) -> Optional[int]:
    """Find which option index matches the target answer."""
    target_clean = target.strip().lower()

    # Exact match
    for i, opt in enumerate(options):
        if opt.strip().lower() == target_clean:
            return i

    # Prefix match (target might be abbreviated)
    for i, opt in enumerate(options):
        if opt.strip().lower().startswith(target_clean):
            return i
        if target_clean.startswith(opt.strip().lower()):
            return i

    # Letter match (target might be "A", "B", etc.)
    if len(target_clean) == 1 and target_clean.isalpha():
        idx = ord(target_clean) - ord("a")
        if 0 <= idx < len(options):
            return idx

    # Digit match
    if target_clean.isdigit():
        idx = int(target_clean)
        if 0 <= idx < len(options):
            return idx

    # Substring containment
    for i, opt in enumerate(options):
        if target_clean in opt.strip().lower() or opt.strip().lower() in target_clean:
            return i

    return None


def _is_binary_task(options: list[str], task_name: str) -> bool:
    """Check if a 2-option task should be treated as noul (binary confidence)."""
    binary_patterns = {
        ("true", "false"),
        ("yes", "no"),
        ("entailment", "not_entailment"),
        ("entailment", "contradiction"),
        ("positive", "negative"),
        ("correct", "incorrect"),
        ("valid", "invalid"),
        ("agree", "disagree"),
        ("support", "refute"),
        ("accept", "reject"),
    }
    opt_pair = tuple(o.strip().lower() for o in options[:2])
    if opt_pair in binary_patterns or opt_pair[::-1] in binary_patterns:
        return True

    # NLI-like tasks with 2 options are typically noul
    nli_keywords = ["nli", "entailment", "inference", "rte", "wnli"]
    if any(kw in task_name.lower() for kw in nli_keywords):
        return True

    return False


def load_tiny_prototype(
    num_tasks: int = 5,
    examples_per_task: int = 100,
    seed: int = 42,
) -> tuple[list[TaskSpec], dict[str, list[JevExample]]]:
    """
    Load a tiny slice for local prototyping (Phase 2).
    Minimal data to verify the full pipeline works end-to-end.
    """
    return load_tasksource_instruct(
        max_tasks=num_tasks,
        max_examples_per_task=examples_per_task,
        min_examples_per_task=10,
        seed=seed,
    )
