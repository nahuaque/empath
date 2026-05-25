"""Relational multi-turn pattern detection for coaching sessions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any

from kanren import Relation, fact, run, var
from pydantic import BaseModel, Field


HypothesisKey = tuple[str, str]


class LongitudinalSupport(BaseModel):
    """One graph item that helped support a longitudinal pattern."""

    turn: int
    kind: str
    label: str
    source: str | None = None


class LongitudinalPattern(BaseModel):
    """A tentative pattern inferred across multiple turns."""

    pattern: str
    label: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    turns: tuple[int, ...]
    support: tuple[LongitudinalSupport, ...] = Field(default_factory=tuple)


@dataclass(frozen=True)
class LongitudinalTurn:
    """Normalized turn record consumed by the relational longitudinal detector."""

    turn: int
    emotions: tuple[str, ...] = ()
    behaviors: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    hypotheses: tuple[HypothesisKey, ...] = ()
    intervention: str | None = None
    signals: frozenset[str] = frozenset()
    signal_support: Mapping[str, tuple[LongitudinalSupport, ...]] = field(
        default_factory=dict
    )


def longitudinal_turn_from_data(
    *,
    turn: int,
    extraction: Mapping[str, Any],
    kernel_snapshot: Mapping[str, Any],
    response_plan: Mapping[str, Any],
) -> LongitudinalTurn:
    """Build one normalized longitudinal record from a completed chat turn."""

    emotions = _labels(extraction.get("emotions"))
    behaviors = _labels(extraction.get("behaviors"))
    features = _labels(extraction.get("features"))
    values = _labels(extraction.get("values"))
    goals = _labels(extraction.get("goals"))
    hypotheses = tuple(
        (source, pattern)
        for item in _values(kernel_snapshot.get("hypotheses"))
        if (source := _label(_as_mapping(item).get("source")))
        and (pattern := _label(_as_mapping(item).get("pattern")))
    )
    intervention = _label(response_plan.get("intervention")) or None

    raw_record = LongitudinalTurn(
        turn=turn,
        emotions=emotions,
        behaviors=behaviors,
        features=features,
        values=values,
        goals=goals,
        hypotheses=hypotheses,
        intervention=intervention,
    )
    signals, support = _derive_signals(raw_record)
    return LongitudinalTurn(
        turn=turn,
        emotions=emotions,
        behaviors=behaviors,
        features=features,
        values=values,
        goals=goals,
        hypotheses=hypotheses,
        intervention=intervention,
        signals=frozenset(signals),
        signal_support=support,
    )


def detect_longitudinal_patterns(
    turns: tuple[LongitudinalTurn, ...] | list[LongitudinalTurn],
    *,
    limit: int = 8,
) -> tuple[LongitudinalPattern, ...]:
    """Infer recurring patterns from normalized turn records."""

    return LongitudinalPatternDetector(tuple(turns)).detect(limit=limit)


class LongitudinalPatternDetector:
    """A tiny miniKanren layer for relations over multiple turns."""

    def __init__(self, turns: tuple[LongitudinalTurn, ...]) -> None:
        self.turns = tuple(sorted(turns, key=lambda item: item.turn))
        self.turn_signal = Relation("turn_signal")
        self.turn_hypothesis = Relation("turn_hypothesis")
        self.turn_intervention = Relation("turn_intervention")
        self._load_facts()

    def detect(self, *, limit: int = 8) -> tuple[LongitudinalPattern, ...]:
        patterns: list[LongitudinalPattern] = []
        patterns.extend(self._specific_loops())
        patterns.extend(self._repeated_hypotheses())
        patterns.extend(self._repeated_interventions())
        patterns = _unique_patterns(patterns)
        patterns.sort(key=lambda item: (-item.confidence, item.pattern, item.turns))
        return tuple(patterns[:limit])

    def _load_facts(self) -> None:
        for record in self.turns:
            for signal in record.signals:
                fact(self.turn_signal, record.turn, signal)
            for source, pattern in record.hypotheses:
                fact(self.turn_hypothesis, record.turn, source, pattern)
            if record.intervention:
                fact(self.turn_intervention, record.turn, record.intervention)

    def _specific_loops(self) -> list[LongitudinalPattern]:
        return [
            pattern
            for pattern in (
                self._same_turn_pair_pattern(
                    "recurring_anxiety_avoidance_loop",
                    "recurring anxiety avoidance loop",
                    "Anxiety and avoidance have shown up together across turns.",
                    "anxiety",
                    "avoidance",
                ),
                self._same_turn_pair_pattern(
                    "recurring_shame_identity_threat",
                    "recurring shame identity threat",
                    "Shame is repeatedly getting linked with identity or self-worth pressure.",
                    "shame",
                    "identity_threat",
                ),
                self._same_turn_pair_pattern(
                    "recurring_certainty_avoidance_loop",
                    "recurring certainty avoidance loop",
                    "A need for certainty is repeatedly paired with avoiding or delaying action.",
                    "certainty_need",
                    "avoidance",
                ),
                self._same_turn_pair_pattern(
                    "recurring_values_action_gap",
                    "recurring values action gap",
                    "Values or goals keep appearing alongside blocked, delayed, or avoided action.",
                    "valued_direction",
                    "action_block",
                ),
                self._same_turn_pair_pattern(
                    "recurring_approval_threat_loop",
                    "recurring approval threat loop",
                    "Concern about approval, rejection, or judgment is repeatedly shaping the work.",
                    "approval_threat",
                    "interpersonal_threat",
                ),
                self._same_turn_pair_pattern(
                    "recurring_control_struggle_loop",
                    "recurring control struggle loop",
                    "Trying to get rid of an internal experience keeps showing up as part of the stuck point.",
                    "control_struggle",
                    "action_block",
                ),
                self._single_signal_pattern(
                    "recurring_high_distress_gating",
                    "recurring high distress gating",
                    "High activation has appeared across turns, so regulation should stay ahead of cognitive challenge.",
                    "high_distress",
                ),
                self._same_turn_pair_pattern(
                    "recurring_vulnerability_distress_loop",
                    "recurring vulnerability distress loop",
                    "Basic vulnerability factors and distress are repeatedly appearing together.",
                    "vulnerability",
                    "distress",
                ),
            )
            if pattern is not None
        ]

    def _repeated_hypotheses(self) -> list[LongitudinalPattern]:
        turn_var = var()
        source_var = var()
        pattern_var = var()
        rows = run(
            0,
            (turn_var, source_var, pattern_var),
            self.turn_hypothesis(turn_var, source_var, pattern_var),
        )
        grouped: dict[HypothesisKey, set[int]] = defaultdict(set)
        for turn, source, pattern in rows:
            key = (str(source), str(pattern))
            if key in _GENERIC_REPEAT_SKIP:
                continue
            grouped[key].add(int(turn))

        patterns = []
        for (source, pattern), turns in grouped.items():
            if len(turns) < 2:
                continue
            support = tuple(
                LongitudinalSupport(
                    turn=turn,
                    kind="hypothesis",
                    label=f"{source}: {pattern}",
                    source="longitudinal_kernel",
                )
                for turn in sorted(turns)
            )
            patterns.append(
                LongitudinalPattern(
                    pattern=f"recurring_{source}_{pattern}",
                    label=f"recurring {source} {pattern}".replace("_", " "),
                    description=(
                        f"The {source.upper() if source in {'act', 'cbt', 'dbt', 'rebt'} else source} "
                        f"pattern {pattern.replace('_', ' ')} appeared in multiple turns."
                    ),
                    confidence=_confidence(len(turns), len(support)),
                    turns=tuple(sorted(turns)),
                    support=support,
                )
            )
        return patterns

    def _repeated_interventions(self) -> list[LongitudinalPattern]:
        turn_var = var()
        intervention_var = var()
        rows = run(
            0,
            (turn_var, intervention_var),
            self.turn_intervention(turn_var, intervention_var),
        )
        grouped: dict[str, set[int]] = defaultdict(set)
        for turn, intervention in rows:
            grouped[str(intervention)].add(int(turn))

        patterns = []
        for intervention, turns in grouped.items():
            if len(turns) < 2:
                continue
            support = tuple(
                LongitudinalSupport(
                    turn=turn,
                    kind="intervention",
                    label=intervention,
                    source="longitudinal_kernel",
                )
                for turn in sorted(turns)
            )
            patterns.append(
                LongitudinalPattern(
                    pattern=f"repeated_intervention_{intervention}",
                    label=f"repeated {intervention}".replace("_", " "),
                    description=(
                        f"The intervention {intervention.replace('_', ' ')} has been used more than once recently."
                    ),
                    confidence=_confidence(len(turns), len(support)) - 0.04,
                    turns=tuple(sorted(turns)),
                    support=support,
                )
            )
        return patterns

    def _same_turn_pair_pattern(
        self,
        pattern: str,
        label: str,
        description: str,
        first_signal: str,
        second_signal: str,
    ) -> LongitudinalPattern | None:
        turn_var = var()
        rows = run(
            0,
            turn_var,
            self.turn_signal(turn_var, first_signal),
            self.turn_signal(turn_var, second_signal),
        )
        turns = tuple(sorted({int(turn) for turn in rows}))
        if len(turns) < 2:
            return None
        support = self._support_for(turns, (first_signal, second_signal))
        return LongitudinalPattern(
            pattern=pattern,
            label=label,
            description=description,
            confidence=_confidence(len(turns), len(support)),
            turns=turns,
            support=support,
        )

    def _single_signal_pattern(
        self,
        pattern: str,
        label: str,
        description: str,
        signal: str,
    ) -> LongitudinalPattern | None:
        turns = self._turns_for_signal(signal)
        if len(turns) < 2:
            return None
        support = self._support_for(turns, (signal,))
        return LongitudinalPattern(
            pattern=pattern,
            label=label,
            description=description,
            confidence=_confidence(len(turns), len(support)),
            turns=turns,
            support=support,
        )

    def _turns_for_signal(self, signal: str) -> tuple[int, ...]:
        turn_var = var()
        return tuple(
            sorted({int(turn) for turn in run(0, turn_var, self.turn_signal(turn_var, signal))})
        )

    def _support_for(
        self,
        turns: tuple[int, ...],
        signals: tuple[str, ...],
    ) -> tuple[LongitudinalSupport, ...]:
        by_turn = {record.turn: record for record in self.turns}
        support: list[LongitudinalSupport] = []
        seen = set()
        for turn in turns:
            record = by_turn.get(turn)
            if record is None:
                continue
            for signal in signals:
                for item in record.signal_support.get(signal, ()):
                    key = (item.turn, item.kind, item.label)
                    if key not in seen:
                        seen.add(key)
                        support.append(item)
        return tuple(support[:12])


def _derive_signals(
    record: LongitudinalTurn,
) -> tuple[set[str], dict[str, tuple[LongitudinalSupport, ...]]]:
    signals: set[str] = set()
    support: dict[str, list[LongitudinalSupport]] = defaultdict(list)

    def add(signal: str, kind: str, label: str) -> None:
        signals.add(signal)
        support[signal].append(
            LongitudinalSupport(
                turn=record.turn,
                kind=kind,
                label=label,
                source="longitudinal_kernel",
            )
        )

    for emotion in record.emotions:
        if emotion in {"anxiety", "shame", "sadness", "anger", "overwhelm"}:
            add(emotion, "emotion", emotion)
        if emotion in {"anxiety", "shame", "overwhelm"}:
            add("distress", "emotion", emotion)

    for behavior in record.behaviors:
        if behavior in {"avoidance", "procrastination", "withdrawal"}:
            add("avoidance", "behavior", behavior)
            add("action_block", "behavior", behavior)
        if behavior in {"inaction", "procrastination"}:
            add("action_block", "behavior", behavior)
        if behavior == "rumination":
            add("rumination", "behavior", behavior)

    for feature in record.features:
        if feature in {"high_distress", "emotion_dysregulation"}:
            add("distress", "feature", feature)
        if feature == "high_distress":
            add("high_distress", "feature", feature)
        if feature in {"identity_global_rating", "identity_fusion", "global_label"}:
            add("identity_threat", "feature", feature)
        if feature in {"certainty_demand", "certainty_demand_claim"}:
            add("certainty_need", "feature", feature)
        if feature in {"approval_threat", "approval_demand", "mind_reading_claim"}:
            add("approval_threat", "feature", feature)
            add("interpersonal_threat", "feature", feature)
        if feature == "control_struggle":
            add("control_struggle", "feature", feature)
        if feature == "vulnerability_factors":
            add("vulnerability", "feature", feature)

    if record.values or record.goals:
        for value in record.values[:3]:
            add("valued_direction", "value", value)
        for goal in record.goals[:3]:
            add("valued_direction", "goal", goal)

    for source, pattern in record.hypotheses:
        label = f"{source}: {pattern}"
        if (source, pattern) in _AVOIDANCE_HYPOTHESES:
            add("avoidance", "hypothesis", label)
            add("action_block", "hypothesis", label)
        if (source, pattern) in _IDENTITY_THREAT_HYPOTHESES:
            add("identity_threat", "hypothesis", label)
        if (source, pattern) in _CERTAINTY_HYPOTHESES:
            add("certainty_need", "hypothesis", label)
        if (source, pattern) in _APPROVAL_HYPOTHESES:
            add("approval_threat", "hypothesis", label)
            add("interpersonal_threat", "hypothesis", label)
        if (source, pattern) in _CONTROL_HYPOTHESES:
            add("control_struggle", "hypothesis", label)
        if (source, pattern) in _VULNERABILITY_HYPOTHESES:
            add("vulnerability", "hypothesis", label)
        if source == "emotion" and pattern in {"anxiety", "shame", "overwhelm"}:
            add("distress", "hypothesis", label)
        if (source, pattern) in {("policy", "high_distress"), ("loop", "high_distress_gating")}:
            add("high_distress", "hypothesis", label)

    return signals, {signal: tuple(items) for signal, items in support.items()}


_AVOIDANCE_HYPOTHESES = {
    ("act", "experiential_avoidance"),
    ("focus", "avoidance_escape"),
    ("focus", "motivation_persistence"),
    ("loop", "avoidance_identity_threat"),
    ("loop", "valued_action_procrastination"),
    ("loop", "certainty_avoidance_loop"),
}

_IDENTITY_THREAT_HYPOTHESES = {
    ("act", "fusion"),
    ("act", "self_as_content"),
    ("cbt", "global_labeling"),
    ("cbt", "overgeneralization"),
    ("focus", "self_efficacy"),
    ("rebt", "self_downing"),
    ("loop", "avoidance_identity_threat"),
    ("loop", "shame_self_worth_fusion"),
}

_CERTAINTY_HYPOTHESES = {
    ("dbt", "wise_mind_need"),
    ("focus", "decision_problem_solving"),
    ("rebt", "certainty_demandingness"),
    ("loop", "certainty_avoidance_loop"),
}

_APPROVAL_HYPOTHESES = {
    ("cbt", "mind_reading"),
    ("focus", "interpersonal_boundaries"),
    ("rebt", "approval_demandingness"),
    ("loop", "approval_threat_loop"),
}

_CONTROL_HYPOTHESES = {
    ("act", "unworkable_control"),
    ("loop", "control_struggle_loop"),
}

_VULNERABILITY_HYPOTHESES = {
    ("dbt", "vulnerability_factors"),
    ("loop", "vulnerability_distress_loop"),
}

_GENERIC_REPEAT_SKIP = {
    ("emotion", "sadness"),
    ("emotion", "anxiety"),
    ("emotion", "shame"),
    ("emotion", "anger"),
    ("emotion", "overwhelm"),
    ("focus", "avoidance_escape"),
    ("focus", "emotion_distress_regulation"),
    ("focus", "integration_review"),
    ("focus", "motivation_persistence"),
    ("policy", "needs_validation"),
    ("policy", "minimal_disclosure"),
}


def _confidence(turn_count: int, support_count: int) -> float:
    return min(0.88, round(0.62 + 0.08 * max(0, turn_count - 2) + 0.01 * support_count, 2))


def _unique_patterns(patterns: list[LongitudinalPattern]) -> list[LongitudinalPattern]:
    unique: dict[str, LongitudinalPattern] = {}
    for pattern in patterns:
        existing = unique.get(pattern.pattern)
        if existing is None or pattern.confidence > existing.confidence:
            unique[pattern.pattern] = pattern
    return list(unique.values())


def _as_mapping(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return {}


def _values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _labels(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    result = []
    for value in _values(values):
        label = _label(value)
        if label and label not in seen:
            seen.add(label)
            result.append(label)
    return tuple(result)


def _label(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").casefold()).strip("_")
