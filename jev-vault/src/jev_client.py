"""
Jev Client - TypeSafe AI System 1 Decision Engine
=================================================

A first-principles interface for high-speed, type-safe semantic decisions.
Unlike conversational LLMs, Jev does not generate prose or tokens.
It evaluates state against schemas in a single parallel pass and returns
calibrated probability distributions.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

import typesafe_sdk
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeError,
)

__all__ = [
    "Choice",
    "ChoiceAnswer",
    "Noul",
    "NoulAnswer",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "DecisionEngine",
    "DecisionResult",
]


@dataclass(frozen=True)
class DecisionResult:
    """Represents the structured, calibrated evaluation of state."""
    choices: dict[str, ChoiceAnswer]
    scores: dict[str, ScoreAnswer]
    nouls: dict[str, NoulAnswer]
    latency_ms: float
    model: str
    is_simulated: bool = False

    def summary(self) -> str:
        """Render an epistemic breakdown of the decision."""
        lines = [
            f"=== Evaluation Summary (Model: {self.model}, Latency: {self.latency_ms:.1f}ms, Simulated: {self.is_simulated}) ==="
        ]
        if self.nouls:
            lines.append("  [Nouls / Binary Probabilities]:")
            for name, noul in self.nouls.items():
                lines.append(f"    * {name}: P(true) = {noul.noul:.4f}")
        if self.choices:
            lines.append("  [Choices / Categorical Distributions]:")
            for name, choice in self.choices.items():
                lines.append(f"    * {name}: selected='{choice.choice}' (confidence: {choice.confidence:.4f})")
                probs = ", ".join(f"{k}: {v:.3f}" for k, v in choice.probabilities.items())
                lines.append(f"      distribution: [{probs}]")
        if self.scores:
            lines.append("  [Scores / Ordered Rubric Positions]:")
            for name, score in self.scores.items():
                lines.append(f"    * {name}: score={score.score:.2f} (confidence: {score.confidence:.4f})")
                probs = ", ".join(f"level {k}: {v:.3f}" for k, v in score.probabilities.items())
                lines.append(f"      distribution: [{probs}]")
        return "\n".join(lines)


class DecisionEngine:
    """High-level facade over TypeSafe AI's Jev model."""

    def __init__(self, api_key: str | None = None, model: str = "jev-latest"):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.model = model
        self._client: TypeSafeClient | None = None

        if self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"):
            try:
                self._client = TypeSafeClient(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize TypeSafeClient with provided key: {e}", file=sys.stderr)
                self._client = None

    @property
    def has_active_credentials(self) -> bool:
        return self._client is not None

    def evaluate(
        self,
        state: str | Mapping[str, Any] | Sequence[Any],
        questions: Mapping[str, Noul | Choice | Score],
        allow_simulation_fallback: bool = True,
    ) -> DecisionResult:
        """
        Evaluate unstructured state against typed questions.
        If credentials are absent and fallback is enabled, returns a simulated calibrated decision.
        """
        if self._client is not None:
            t0 = time.perf_counter()
            try:
                res: SystemOneResponse = self._client.system_one(
                    state=state,
                    questions=questions,
                    model=self.model,
                )
                latency = (time.perf_counter() - t0) * 1000.0
                return DecisionResult(
                    choices=dict(res.choices),
                    scores=dict(res.scores),
                    nouls=dict(res.nouls),
                    latency_ms=latency,
                    model=res.model,
                    is_simulated=False,
                )
            except TypeSafeAuthenticationError as auth_err:
                if not allow_simulation_fallback:
                    raise
                print(f"[Notice] Live API authentication failed: {auth_err}. Falling back to simulation.", file=sys.stderr)
            except TypeSafeError as api_err:
                if not allow_simulation_fallback:
                    raise
                print(f"[Notice] TypeSafe API error: {api_err}. Falling back to simulation.", file=sys.stderr)

        if not allow_simulation_fallback:
            raise TypeSafeError("No valid TYPESAFE_API_KEY configured and simulation fallback is disabled.")

        # Simulation mode for testing and developer preview without active API key
        return self._simulate(state, questions)

    def _simulate(
        self,
        state: str | Mapping[str, Any] | Sequence[Any],
        questions: Mapping[str, Noul | Choice | Score],
    ) -> DecisionResult:
        """Deterministic, simulated System 1 evaluation for offline development."""
        import hashlib
        t0 = time.perf_counter()
        
        # Pseudo-hash to give consistent, pseudo-random yet repeatable probabilities
        content_repr = str(state)
        seed = int(hashlib.sha256(content_repr.encode()).hexdigest()[:8], 16)

        choices: dict[str, ChoiceAnswer] = {}
        scores: dict[str, ScoreAnswer] = {}
        nouls: dict[str, NoulAnswer] = {}

        for q_name, q in questions.items():
            q_seed = (seed + int(hashlib.sha256(q_name.encode()).hexdigest()[:8], 16)) % (10**6)
            unit_val = (q_seed % 1000) / 1000.0

            if isinstance(q, Noul):
                # Calibrated probability in [0.0, 1.0]
                nouls[q_name] = NoulAnswer(noul=round(unit_val, 4))

            elif isinstance(q, Choice):
                # Distribute probability across criteria keys
                keys = list(q.criteria.keys()) if isinstance(q.criteria, Mapping) else list(q.criteria)
                n = len(keys)
                raw_weights = [(unit_val * (i + 1) * 37) % 100 + 1 for i in range(n)]
                total = sum(raw_weights)
                probs = {k: round(w / total, 4) for k, w in zip(keys, raw_weights)}
                best_choice = max(probs.items(), key=lambda x: x[1])
                choices[q_name] = ChoiceAnswer(
                    choice=best_choice[0],
                    confidence=best_choice[1],
                    probabilities=probs,
                )

            elif isinstance(q, Score):
                # Determine score on rubric
                criteria = list(q.criteria)
                levels = len(criteria)
                raw_weights = [(unit_val * (i + 1) * 19) % 100 + 1 for i in range(levels)]
                total = sum(raw_weights)
                probs = {i + 1: round(w / total, 4) for i, w in zip(range(levels), raw_weights)}
                expected_score = sum(lvl * p for lvl, p in probs.items())
                best_level = max(probs.items(), key=lambda x: x[1])
                legend = {i + 1: str(c) for i, c in enumerate(criteria)}
                scores[q_name] = ScoreAnswer(
                    score=round(expected_score, 2),
                    confidence=best_level[1],
                    legend=legend,
                    probabilities=probs,
                )

        latency = (time.perf_counter() - t0) * 1000.0 + 85.0  # simulate ~85ms inference
        return DecisionResult(
            choices=choices,
            scores=scores,
            nouls=nouls,
            latency_ms=latency,
            model=f"{self.model}-simulated",
            is_simulated=True,
        )
