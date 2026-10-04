"""
=============================================================================
Task Registry — Task metadata, family groupings, license flags
=============================================================================
Central registry mapping tasksource task IDs to Jev primitives, task families,
and license metadata. Task families are used for zero-shot generalization splits.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JevPrimitive(Enum):
    """The three Jev decision primitives."""
    CHOICE = "choice"   # Multi-option selection (probability distribution over options)
    NOUL = "noul"       # Binary confidence scalar in [0, 1]
    SCORE = "score"     # Ordinal/continuous score (distribution over K bins)


class TaskFamily(Enum):
    """
    Coarse task families for zero-shot generalization evaluation.
    Based on standard NLP task taxonomy aligned with tasksource categories.
    """
    NLI = "nli"                         # Natural language inference
    SENTIMENT = "sentiment"             # Sentiment analysis / opinion
    TOPIC = "topic"                     # Topic classification
    HATE_SPEECH = "hate_speech"         # Hate speech / toxicity detection
    PARAPHRASE = "paraphrase"           # Paraphrase detection / semantic similarity
    QA = "qa"                           # Question answering (extractive/MC)
    COMMONSENSE = "commonsense"         # Commonsense reasoning
    FACT_VERIFICATION = "fact_verify"   # Fact checking / claim verification
    EMOTION = "emotion"                 # Emotion detection
    STANCE = "stance"                   # Stance detection
    GRAMMAR = "grammar"                 # Grammar / linguistic acceptability
    DISCOURSE = "discourse"             # Discourse relation / coherence
    ETHICS = "ethics"                   # Ethical judgment
    SUMMARIZATION_EVAL = "summ_eval"    # Summarization quality evaluation
    TRANSLATION_EVAL = "trans_eval"     # Translation quality evaluation
    HUMOR = "humor"                     # Humor detection
    SARCASM = "sarcasm"                 # Sarcasm / irony detection
    INTENT = "intent"                   # Intent classification
    OTHER = "other"                     # Uncategorized


@dataclass
class TaskSpec:
    """Metadata for a single tasksource task."""
    task_id: str                        # HuggingFace dataset ID or tasksource name
    primitive: JevPrimitive             # Mapped Jev primitive type
    family: TaskFamily                  # Task family for zero-shot splits
    num_options: Optional[int] = None   # Number of options (for choice/noul)
    is_ordinal: bool = False            # Whether labels have meaningful order (for score)
    license: str = "unknown"            # License string
    license_ok: bool = True             # Whether license permits training
    needs_attribution: bool = False     # Whether license requires attribution
    max_examples: Optional[int] = None  # Override per-task cap
    notes: str = ""                     # Freeform notes


# ─── Family detection heuristics ───────────────────────────────────────────────
# Keywords in task name/description → family assignment
FAMILY_KEYWORDS = {
    TaskFamily.NLI: [
        "nli", "entailment", "inference", "mnli", "snli", "xnli", "anli",
        "wanli", "hans", "sick", "scitail", "wnli", "rte", "cb",
    ],
    TaskFamily.SENTIMENT: [
        "sentiment", "sst", "imdb", "yelp", "amazon_review", "movie_review",
        "opinion", "polarity", "dynasent",
    ],
    TaskFamily.TOPIC: [
        "topic", "agnews", "ag_news", "yahoo", "newsgroup", "20news",
        "dbpedia", "trec",
    ],
    TaskFamily.HATE_SPEECH: [
        "hate", "toxic", "offensive", "abuse", "dynahate", "hatespeech",
        "civil_comments", "jigsaw",
    ],
    TaskFamily.PARAPHRASE: [
        "paraphrase", "similarity", "sts", "mrpc", "qqp", "paws",
        "semantic_similarity", "stsb",
    ],
    TaskFamily.QA: [
        "qa", "question", "squad", "race", "arc", "openbookqa",
        "commonsenseqa", "social_iqa", "piqa", "boolq", "multirc",
        "copa", "hellaswag", "winogrande", "swag",
    ],
    TaskFamily.COMMONSENSE: [
        "commonsense", "winograd", "copa", "piqa", "hellaswag",
        "social_iqa", "cosmosqa", "abductive",
    ],
    TaskFamily.FACT_VERIFICATION: [
        "fact", "fever", "vitaminc", "claim", "verification",
        "ruletaker", "veridicality",
    ],
    TaskFamily.EMOTION: [
        "emotion", "goemotions", "affect", "empathy",
    ],
    TaskFamily.STANCE: [
        "stance", "perspectrum", "ibm_claim",
    ],
    TaskFamily.GRAMMAR: [
        "grammar", "cola", "acceptability", "blimp", "linguistic",
    ],
    TaskFamily.DISCOURSE: [
        "discourse", "pdtb", "rst", "connective", "coherence",
    ],
    TaskFamily.ETHICS: [
        "ethics", "moral", "delphi", "social_chem", "scruples",
    ],
    TaskFamily.HUMOR: [
        "humor", "joke", "pun", "funny",
    ],
    TaskFamily.SARCASM: [
        "sarcasm", "irony",
    ],
    TaskFamily.INTENT: [
        "intent", "atis", "banking", "clinic", "snips",
    ],
}


def infer_family(task_name: str) -> TaskFamily:
    """
    Infer the task family from the task name using keyword matching.
    Returns TaskFamily.OTHER if no match found.
    """
    name_lower = task_name.lower().replace("-", "_").replace("/", "_")
    # Score each family by number of keyword matches
    best_family = TaskFamily.OTHER
    best_score = 0
    for family, keywords in FAMILY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in name_lower)
        if score > best_score:
            best_score = score
            best_family = family
    return best_family


def infer_primitive(
    num_labels: int,
    task_type: str,
    is_ordinal: bool = False,
    is_regression: bool = False,
) -> JevPrimitive:
    """
    Infer the Jev primitive from task characteristics.

    Rules:
      - MultipleChoice → choice
      - Binary classification (2 labels, not ordinal) → noul
      - Multi-class classification (>2 labels, not ordinal) → choice
      - Ordinal classification or regression → score
    """
    if is_regression or is_ordinal:
        return JevPrimitive.SCORE

    if task_type.lower() in ("multiple_choice", "multiplechoice"):
        return JevPrimitive.CHOICE

    if num_labels == 2:
        return JevPrimitive.NOUL

    return JevPrimitive.CHOICE


# ─── Known permissive licenses ────────────────────────────────────────────────
PERMISSIVE_LICENSES = {
    "mit", "apache-2.0", "apache 2.0", "bsd-2-clause", "bsd-3-clause",
    "cc-by-4.0", "cc-by-3.0", "cc-by-2.0", "cc-by-sa-4.0", "cc-by-sa-3.0",
    "cc0-1.0", "public domain", "unlicense", "openrail",
    "odc-by", "odc-odbl", "pddl",
    # Many HF datasets don't specify → treat as permissive with attribution
    "", "unknown", "other",
}

RESTRICTIVE_LICENSES = {
    "cc-by-nc-4.0", "cc-by-nc-3.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0",
    "cc-by-nd-4.0", "gpl-3.0", "gpl-2.0", "proprietary",
}


def check_license(license_str: str) -> tuple[bool, bool]:
    """
    Check if a license permits training use.
    Returns (is_ok, needs_attribution).
    """
    normalized = license_str.lower().strip()
    if normalized in RESTRICTIVE_LICENSES:
        return False, False
    needs_attr = "cc-by" in normalized or "odc" in normalized
    return True, needs_attr
