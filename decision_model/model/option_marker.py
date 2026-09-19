"""
=============================================================================
Option-Marker Scheme & Tokenization
=============================================================================
Defines the option-marker token format and provides formatting / tokenization
utilities for presenting multi-choice, noul, and score tasks to the model.

Format:
  [Instruction / Context text]

  Options:
  [OPT_A] Option 0 text
  [OPT_B] Option 1 text
  [OPT_C] Option 2 text
  ...

Special tokens: [OPT_A], [OPT_B], ..., [OPT_Z] (up to 26 options).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import string
from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from decision_model.data.tasksource_loader import JevExample
from decision_model.data.task_registry import JevPrimitive

# Option marker symbols: [OPT_A], [OPT_B], ..., [OPT_Z]
OPTION_LETTERS = list(string.ascii_uppercase)  # A-Z (26 options)
OPTION_MARKERS = [f"[OPT_{c}]" for c in OPTION_LETTERS]


def get_option_markers(num_options: int) -> list[str]:
    """Return the list of option markers for a given number of options."""
    if num_options > len(OPTION_MARKERS):
        raise ValueError(
            f"Requested {num_options} options, but maximum supported is {len(OPTION_MARKERS)}"
        )
    return OPTION_MARKERS[:num_options]


def format_prompt(
    example: JevExample,
    use_end_markers: bool = True,
    shuffle_options: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[str, list[str], int]:
    """
    Format a JevExample into a standardized prompt string.

    When use_end_markers=True (End-Marker Causal Parity):
      Presents all option texts first, then appends markers at the end.
      This ensures all candidate representations attend over ALL option texts
      simultaneously in causal decoders, eliminating position blindness.

    When shuffle_options=True (Permutation Invariance):
      Randomly permutes option order and remaps the ground truth label accordingly.

    Returns:
        prompt_text: Full formatted text
        markers_used: List of option marker tokens
        active_label: Correct label index (remapped if shuffled)
    """
    options = list(example.options)
    label = example.label

    if shuffle_options and len(options) > 1 and not example.is_ordinal:
        # Permute non-ordinal options
        if rng is None:
            rng = np.random.RandomState()
        perm = rng.permutation(len(options))
        options = [options[i] for i in perm]
        label = int(np.where(perm == example.label)[0][0])

    markers = get_option_markers(len(options))

    if use_end_markers:
        lines = [example.instruction.strip(), "", "Candidate Options:"]
        for letter, opt_text in zip(OPTION_LETTERS, options):
            lines.append(f"({letter}) {opt_text.strip()}")
        lines.append("")
        lines.append("Decision: " + " ".join(markers))
    else:
        lines = [example.instruction.strip(), "", "Options:"]
        for marker, opt_text in zip(markers, options):
            lines.append(f"{marker} {opt_text.strip()}")

    prompt_text = "\n".join(lines)
    return prompt_text, markers, label


@dataclass
class TokenizedBatch:
    """A batch ready for the JevDecisionModel forward pass."""
    input_ids: torch.Tensor             # (batch, seq_len)
    attention_mask: torch.Tensor        # (batch, seq_len)
    marker_positions: list[list[int]]   # List per item of token indices for each marker
    labels: torch.Tensor                # (batch,) ground truth option index
    num_options: list[int]              # Number of valid options per item
    is_ordinal: torch.Tensor            # (batch,) boolean
    primitive: list[JevPrimitive]       # Primitive type per item


def setup_tokenizer(tokenizer: PreTrainedTokenizerBase) -> tuple[PreTrainedTokenizerBase, list[int]]:
    """
    Ensure all option markers are present as special tokens in the tokenizer.
    Returns:
        tokenizer: Updated tokenizer
        marker_token_ids: Token IDs for each marker [OPT_A]..[OPT_Z]
    """
    # Check if pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

    # Add option markers as special tokens if not already present
    tokens_to_add = [m for m in OPTION_MARKERS if m not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    marker_token_ids = [tokenizer.convert_tokens_to_ids(m) for m in OPTION_MARKERS]
    return tokenizer, marker_token_ids


def encode_examples(
    examples: list[JevExample],
    tokenizer: PreTrainedTokenizerBase,
    marker_token_ids: list[int],
    max_length: int = 512,
    device: Optional[torch.device] = None,
    use_end_markers: bool = True,
    shuffle_options: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> TokenizedBatch:
    """
    Encode a list of JevExamples into a TokenizedBatch.
    Finds the exact token positions of the option markers in each sequence.
    """
    prompts = []
    num_opts = []
    labels = []
    ordinals = []
    primitives = []

    for ex in examples:
        text, _, active_label = format_prompt(
            ex,
            use_end_markers=use_end_markers,
            shuffle_options=shuffle_options,
            rng=rng,
        )
        prompts.append(text)
        num_opts.append(len(ex.options))
        labels.append(active_label)
        ordinals.append(ex.is_ordinal)
        primitives.append(ex.primitive)

    # Tokenize batch
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    batch_size, seq_len = input_ids.shape

    # Find position of each option marker in each sequence
    marker_set = set(marker_token_ids)
    all_marker_positions: list[list[int]] = []

    for b in range(batch_size):
        seq_ids = input_ids[b].tolist()
        k = num_opts[b]
        expected_marker_ids = marker_token_ids[:k]

        # Locate marker positions in order
        positions = []
        for target_id in expected_marker_ids:
            # Find the first or last occurrence of target_id
            try:
                pos = seq_ids.index(target_id)
                positions.append(pos)
            except ValueError:
                # If truncated, fallback to last non-padding token
                non_pad = (input_ids[b] != tokenizer.pad_token_id).nonzero()
                fallback_pos = non_pad[-1].item() if len(non_pad) > 0 else seq_len - 1
                positions.append(fallback_pos)

        all_marker_positions.append(positions)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    ordinals_tensor = torch.tensor(ordinals, dtype=torch.bool)

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_tensor = labels_tensor.to(device)
        ordinals_tensor = ordinals_tensor.to(device)

    return TokenizedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        marker_positions=all_marker_positions,
        labels=labels_tensor,
        num_options=num_opts,
        is_ordinal=ordinals_tensor,
        primitive=primitives,
    )
