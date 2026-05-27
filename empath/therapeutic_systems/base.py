"""Base protocol for therapeutic reasoning modules."""

from __future__ import annotations

from typing import Any


class TherapeuticSystem:
    """A pluggable source of therapeutic hypotheses and intervention metadata."""

    source: str

    def load_ontology(self, kernel: Any) -> None:
        """Register relation facts on the coordinator kernel."""

        raise NotImplementedError

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        """Return a miniKanren goal that infers this system's patterns."""

        raise NotImplementedError

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        """Return an optional ranking bonus for this system."""

        return 0.0
