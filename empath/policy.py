"""Adaptive coaching policy memory.

This layer keeps user-correctable feedback separate from the therapeutic
kernel. The kernel still generates coherent candidates; policy memory nudges
ranking and planning based on what the user has marked helpful, costly, or
incorrect in this workspace.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any

from kanren import Relation, fact, run, var
from pydantic import BaseModel, Field

from .experiments import CoachingExperiment, ExperimentFeedbackAction
from .formulation import FeedbackAction, FormulationNode


class PolicyExperimentRecord(BaseModel):
    """Outcome feedback for one proposed coaching experiment."""

    experiment_id: str
    intervention: str
    focus: str
    outcome: ExperimentFeedbackAction
    created_turn: int
    pattern_keys: tuple[str, ...] = Field(default_factory=tuple)
    usefulness: int | None = Field(default=None, ge=0, le=10)
    friction_before: int | None = None
    friction_after: int | None = None
    emotional_shift: str | None = None
    action_taken: str | None = None
    learning: str | None = None
    note: str | None = None


class PolicyFormulationRecord(BaseModel):
    """User correction for one working-map node."""

    node_id: str
    kind: str
    label: str
    action: FeedbackAction
    status: str
    turn: int


class PolicyAdjustment(BaseModel):
    """One candidate score nudge produced by policy memory."""

    intervention: str
    base_score: float
    adjusted_score: float
    delta: float
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    evidence: tuple[str, ...] = Field(default_factory=tuple)


class PolicyMemory:
    """Workspace-scoped adaptive policy facts and candidate ranking nudges."""

    def __init__(self) -> None:
        self._experiment_records: list[PolicyExperimentRecord] = []
        self._formulation_records: list[PolicyFormulationRecord] = []
        self._rebuild_relations()

    def export_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for app persistence."""

        return {
            "experiment_records": [
                item.model_dump() for item in self._experiment_records
            ],
            "formulation_records": [
                item.model_dump() for item in self._formulation_records
            ],
        }

    def import_state(self, data: Mapping[str, Any] | None) -> None:
        """Restore policy facts from export_state output."""

        if not data:
            self._experiment_records = []
            self._formulation_records = []
            self._rebuild_relations()
            return

        self._experiment_records = [
            PolicyExperimentRecord.model_validate(item)
            for item in data.get("experiment_records", ()) or ()
        ]
        self._formulation_records = [
            PolicyFormulationRecord.model_validate(item)
            for item in data.get("formulation_records", ()) or ()
        ]
        self._rebuild_relations()

    def record_experiment(self, experiment: CoachingExperiment) -> None:
        """Record the latest feedback outcome for an experiment."""

        if experiment.outcome is None:
            return
        record = PolicyExperimentRecord(
            experiment_id=experiment.id,
            intervention=_clean_label(experiment.intervention),
            focus=_clean_label(experiment.focus),
            outcome=experiment.outcome,
            created_turn=experiment.created_turn,
            pattern_keys=tuple(_clean_label(item) for item in experiment.pattern_keys),
            usefulness=experiment.usefulness,
            friction_before=experiment.friction_before,
            friction_after=experiment.friction_after,
            emotional_shift=_clean_text(experiment.emotional_shift),
            action_taken=_clean_text(experiment.action_taken),
            learning=_clean_text(experiment.learning),
            note=_clean_text(experiment.note),
        )
        self._experiment_records = [
            item
            for item in self._experiment_records
            if item.experiment_id != record.experiment_id
        ]
        self._experiment_records.append(record)
        self._experiment_records = self._experiment_records[-80:]
        self._rebuild_relations()

    def record_formulation(self, node: FormulationNode, action: FeedbackAction) -> None:
        """Record the latest user correction for a working-map node."""

        record = PolicyFormulationRecord(
            node_id=node.id,
            kind=_clean_label(node.kind),
            label=_clean_label(node.label),
            action=action,
            status=str(node.status),
            turn=int(node.last_seen_turn or node.first_seen_turn or 0),
        )
        self._formulation_records = [
            item for item in self._formulation_records if item.node_id != record.node_id
        ]
        self._formulation_records.append(record)
        self._formulation_records = self._formulation_records[-120:]
        self._rebuild_relations()

    def apply_to_kernel_snapshot(
        self,
        kernel_snapshot: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a kernel snapshot with policy-adjusted candidate scores."""

        snapshot = copy.deepcopy(dict(kernel_snapshot))
        candidates = list(snapshot.get("candidates") or ())
        adjustments: list[PolicyAdjustment] = []
        adjusted_candidates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                adjusted_candidates.append(candidate)
                continue
            updated = dict(candidate)
            intervention = _clean_label(updated.get("intervention"))
            if not intervention:
                adjusted_candidates.append(updated)
                continue
            base_score = _float(updated.get("score"))
            delta, reasons, evidence = self._candidate_delta(updated)
            if delta:
                updated["base_score"] = base_score
                updated["policy_delta"] = round(delta, 2)
                updated["policy_reasons"] = reasons
                updated["score"] = round(base_score + delta, 2)
                adjustments.append(
                    PolicyAdjustment(
                        intervention=intervention,
                        base_score=base_score,
                        adjusted_score=updated["score"],
                        delta=round(delta, 2),
                        reasons=tuple(reasons),
                        evidence=tuple(evidence),
                    )
                )
            adjusted_candidates.append(updated)

        adjusted_candidates.sort(
            key=lambda item: (
                -_float(item.get("score") if isinstance(item, dict) else 0.0),
                str(item.get("intervention") if isinstance(item, dict) else item),
            )
        )
        snapshot["candidates"] = adjusted_candidates
        if adjustments:
            snapshot["policy_adjustments"] = [item.model_dump() for item in adjustments]

        report = {
            "summary": self.summary(),
            "adjustments": [item.model_dump() for item in adjustments],
        }
        return snapshot, report

    def prompt_context(self, *, limit: int = 6) -> str:
        """Summarize policy memory for the response-planning LLM."""

        summary = self.summary()
        if summary.get("empty"):
            return ""

        lines = [
            "Treat these workspace feedback signals as tentative outcome evidence, not fixed rules:"
        ]
        priors = summary.get("personalized_priors") or []
        if priors:
            lines.append("- Personalized outcome priors:")
            for item in priors[:limit]:
                lines.append(f"  - {item['description']}")
        helpful = summary.get("helpful") or []
        if helpful:
            lines.append("- Moves with positive feedback:")
            for item in helpful[:limit]:
                lines.append(
                    f"  - {_human_label(item['intervention'])}: {item['description']}"
                )
        costly = summary.get("costly") or []
        if costly:
            lines.append("- Moves to shrink, soften, or avoid for now:")
            for item in costly[:limit]:
                lines.append(
                    f"  - {_human_label(item['intervention'])}: {item['description']}"
                )
        corrections = summary.get("map_feedback") or []
        if corrections:
            lines.append("- User-corrected working-map facts:")
            for item in corrections[:limit]:
                lines.append(
                    f"  - {item['action']} {_human_label(item['kind'])}: {item['label']}"
                )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Return a compact API/sidebar summary of learned policy evidence."""

        helpful = self._outcome_summary(positive=True)
        costly = self._outcome_summary(positive=False)
        personalized_priors = self._personalized_priors()
        map_feedback = [
            {
                "kind": item.kind,
                "label": _human_label(item.label),
                "action": item.action,
                "status": item.status,
            }
            for item in self._formulation_records[-10:]
        ]
        return {
            "empty": not helpful
            and not costly
            and not personalized_priors
            and not map_feedback,
            "personalized_priors": personalized_priors,
            "helpful": helpful,
            "costly": costly,
            "map_feedback": map_feedback,
            "counts": {
                "experiment_outcomes": len(self._experiment_records),
                "map_corrections": len(self._formulation_records),
            },
        }

    def relation_facts(self) -> dict[str, tuple[tuple[Any, ...], ...]]:
        """Expose the policy facts in queryable tuple form for tests/debugging."""

        intervention = var()
        outcome = var()
        focus = var()
        pattern = var()
        usefulness = var()
        node_kind = var()
        node_label = var()
        action = var()
        return {
            "experiment_outcome": tuple(
                run(
                    0,
                    (intervention, outcome, focus),
                    self.experiment_outcome(intervention, outcome, focus),
                )
            ),
            "experiment_pattern": tuple(
                run(
                    0,
                    (intervention, outcome, focus, pattern),
                    self.experiment_pattern(intervention, outcome, focus, pattern),
                )
            ),
            "experiment_usefulness": tuple(
                run(
                    0,
                    (intervention, outcome, usefulness),
                    self.experiment_usefulness(intervention, outcome, usefulness),
                )
            ),
            "formulation_feedback": tuple(
                run(
                    0,
                    (node_kind, node_label, action),
                    self.formulation_feedback(node_kind, node_label, action),
                )
            ),
        }

    def _rebuild_relations(self) -> None:
        self.experiment_outcome = Relation("policy_experiment_outcome")
        self.experiment_pattern = Relation("policy_experiment_pattern")
        self.experiment_usefulness = Relation("policy_experiment_usefulness")
        self.formulation_feedback = Relation("policy_formulation_feedback")
        for record in self._experiment_records:
            fact(
                self.experiment_outcome,
                record.intervention,
                record.outcome,
                record.focus,
            )
            for pattern_key in record.pattern_keys:
                fact(
                    self.experiment_pattern,
                    record.intervention,
                    record.outcome,
                    record.focus,
                    pattern_key,
                )
            if record.usefulness is not None:
                fact(
                    self.experiment_usefulness,
                    record.intervention,
                    record.outcome,
                    record.usefulness,
                )
        for record in self._formulation_records:
            fact(
                self.formulation_feedback,
                record.kind,
                record.label,
                record.action,
            )

    def _candidate_delta(
        self,
        candidate: Mapping[str, Any],
    ) -> tuple[float, list[str], list[str]]:
        intervention = _clean_label(candidate.get("intervention"))
        reasons: list[str] = []
        evidence: list[str] = []
        delta = 0.0
        candidate_pattern_keys = _candidate_pattern_key_labels(candidate)

        experiment_delta = 0.0
        matching_records = [
            item
            for item in self._experiment_records[-20:]
            if item.intervention == intervention
        ]
        for weight, record in _decayed(matching_records):
            record_delta = _outcome_delta(record.outcome)
            record_delta += _usefulness_delta(record.usefulness)
            if record.friction_before is not None and record.friction_after is not None:
                record_delta += max(
                    -0.4,
                    min(0.4, (record.friction_before - record.friction_after) / 10),
                )
            context_weight = _context_weight(
                candidate_pattern_keys, record.pattern_keys
            )
            experiment_delta += record_delta * weight * context_weight
            evidence.append(f"experiment:{record.experiment_id}:{record.outcome}")
        experiment_delta = max(-2.0, min(2.0, experiment_delta))
        if experiment_delta:
            delta += experiment_delta
            latest = matching_records[-1]
            reasons.append(_experiment_reason(latest, candidate_pattern_keys))

        hypothesis_keys = _candidate_hypothesis_keys(candidate)
        map_delta = 0.0
        for record in self._formulation_records[-30:]:
            if record.kind == "hypothesis":
                parsed = _parse_hypothesis_label(record.label)
                if parsed and parsed in hypothesis_keys:
                    weight = 0.7 if record.action == "confirm" else -1.0
                    map_delta += weight
                    evidence.append(f"map:{record.node_id}:{record.action}")
                    reasons.append(_formulation_reason(record))
            elif record.kind == "intervention" and record.label == intervention:
                weight = 0.55 if record.action == "confirm" else -0.9
                map_delta += weight
                evidence.append(f"map:{record.node_id}:{record.action}")
                reasons.append(_formulation_reason(record))
        map_delta = max(-1.5, min(1.5, map_delta))
        if map_delta:
            delta += map_delta

        return round(delta, 2), _unique(reasons), _unique(evidence)

    def _outcome_summary(self, *, positive: bool) -> list[dict[str, Any]]:
        grouped: dict[str, list[PolicyExperimentRecord]] = {}
        for record in self._experiment_records:
            outcome_score = _outcome_delta(record.outcome)
            if outcome_score == 0:
                continue
            if (outcome_score > 0) != positive:
                continue
            grouped.setdefault(record.intervention, []).append(record)

        items = []
        for intervention, records in grouped.items():
            latest = records[-1]
            count = len(records)
            focus_text = f" around {latest.focus}" if latest.focus else ""
            usefulness_text = _avg_usefulness_text(records)
            if positive:
                description = f"{count} positive outcome{'s' if count != 1 else ''}{focus_text}; latest was {latest.outcome}."
            else:
                description = f"{count} costly outcome{'s' if count != 1 else ''}{focus_text}; latest was {latest.outcome}."
            if usefulness_text:
                description = f"{description[:-1]}; {usefulness_text}."
            items.append(
                {
                    "intervention": intervention,
                    "description": description,
                    "latest_outcome": latest.outcome,
                    "focus": latest.focus,
                }
            )
        items.sort(key=lambda item: (item["intervention"], item["latest_outcome"]))
        return items

    def _personalized_priors(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[PolicyExperimentRecord]] = {}
        for record in self._experiment_records[-60:]:
            context = _context_label(record.pattern_keys, record.focus)
            grouped.setdefault((record.intervention, context), []).append(record)

        priors = []
        for (intervention, context), records in grouped.items():
            useful_values = [
                item.usefulness for item in records if item.usefulness is not None
            ]
            avg_usefulness = (
                round(sum(useful_values) / len(useful_values), 1)
                if useful_values
                else None
            )
            positive_count = sum(
                1 for item in records if _outcome_delta(item.outcome) > 0
            )
            costly_count = sum(
                1 for item in records if _outcome_delta(item.outcome) < 0
            )
            latest = records[-1]
            score = positive_count - costly_count
            if avg_usefulness is not None:
                score += (avg_usefulness - 5.0) / 5.0
            description_parts = [
                f"{_human_label(intervention)} around {context}",
                f"{positive_count}/{len(records)} positive",
            ]
            if costly_count:
                description_parts.append(f"{costly_count} costly")
            if avg_usefulness is not None:
                description_parts.append(f"avg usefulness {avg_usefulness}/10")
            if latest.emotional_shift:
                description_parts.append(f"latest shift: {latest.emotional_shift}")
            priors.append(
                {
                    "intervention": intervention,
                    "context": context,
                    "observations": len(records),
                    "positive": positive_count,
                    "costly": costly_count,
                    "avg_usefulness": avg_usefulness,
                    "latest_outcome": latest.outcome,
                    "latest_action_taken": latest.action_taken,
                    "latest_shift": latest.emotional_shift,
                    "description": "; ".join(description_parts) + ".",
                    "score": round(score, 2),
                }
            )
        priors.sort(key=lambda item: (-_float(item["score"]), item["intervention"]))
        return priors[:8]


def _candidate_hypothesis_keys(candidate: Mapping[str, Any]) -> set[tuple[str, str]]:
    keys = set()
    for item in candidate.get("hypotheses") or ():
        if not isinstance(item, Mapping):
            continue
        source = _clean_label(item.get("source"))
        pattern = _clean_label(item.get("pattern"))
        if source and pattern:
            keys.add((source, pattern))
    return keys


def _candidate_pattern_key_labels(candidate: Mapping[str, Any]) -> set[str]:
    return {
        f"{source}:{pattern}"
        for source, pattern in _candidate_hypothesis_keys(candidate)
    }


def _parse_hypothesis_label(label: str) -> tuple[str, str] | None:
    if ":" not in label:
        return None
    source, pattern = label.split(":", 1)
    source = _clean_label(source)
    pattern = _clean_label(pattern)
    if not source or not pattern:
        return None
    return source, pattern


def _outcome_delta(outcome: str) -> float:
    return {
        "helped": 1.25,
        "completed": 0.75,
        "neutral": 0.0,
        "did_not_help": -1.25,
        "too_hard": -1.5,
        "skipped": -0.75,
    }.get(outcome, 0.0)


def _usefulness_delta(usefulness: int | None) -> float:
    if usefulness is None:
        return 0.0
    return max(-0.8, min(0.8, ((usefulness - 5) / 5) * 0.8))


def _context_weight(candidate_keys: set[str], record_keys: tuple[str, ...]) -> float:
    record_key_set = set(record_keys)
    if not candidate_keys or not record_key_set:
        return 1.0
    overlap = candidate_keys & record_key_set
    if overlap:
        return 1.0 + min(0.6, 0.2 * len(overlap))
    return 0.55


def _experiment_reason(
    record: PolicyExperimentRecord,
    candidate_keys: set[str] | None = None,
) -> str:
    label = _human_label(record.intervention)
    context = _shared_context_text(candidate_keys or set(), record.pattern_keys)
    usefulness = (
        f" and was rated {record.usefulness}/10 useful"
        if record.usefulness is not None
        else ""
    )
    prefix = f"Prior feedback{context} said {label}"
    if record.outcome == "too_hard":
        return f"{prefix} was too hard{usefulness}; shrink the dose or choose a gentler route."
    if record.outcome == "did_not_help":
        return f"{prefix} did not help{usefulness}; use it cautiously."
    if record.outcome == "skipped":
        return f"A prior {label} experiment{context} was skipped{usefulness}; keep the next step smaller."
    if record.outcome == "helped":
        return f"{prefix} helped{usefulness}."
    if record.outcome == "completed":
        return f"A prior {label} experiment{context} was completed{usefulness}."
    return f"Prior feedback on {label}{context} was {record.outcome}{usefulness}."


def _shared_context_text(candidate_keys: set[str], record_keys: tuple[str, ...]) -> str:
    overlap = candidate_keys & set(record_keys)
    if not overlap:
        return ""
    return f" in a similar {_human_context(overlap)} context"


def _context_label(pattern_keys: tuple[str, ...], fallback: str) -> str:
    if pattern_keys:
        return _human_context(pattern_keys[:2])
    if fallback:
        return _human_label(fallback)
    return "similar situations"


def _human_context(pattern_keys: Any) -> str:
    labels = []
    for key in pattern_keys:
        _, _, pattern = str(key).partition(":")
        labels.append(_human_label(pattern or str(key)))
    return " + ".join(_unique(labels)) or "similar situations"


def _avg_usefulness_text(records: list[PolicyExperimentRecord]) -> str | None:
    values = [item.usefulness for item in records if item.usefulness is not None]
    if not values:
        return None
    average = round(sum(values) / len(values), 1)
    return f"avg usefulness {average}/10"


def _formulation_reason(record: PolicyFormulationRecord) -> str:
    action = "confirmed" if record.action == "confirm" else "pushed back on"
    return f"The user {action} the working-map item {_human_label(record.label)}."


def _decayed(
    records: list[PolicyExperimentRecord],
) -> tuple[tuple[float, PolicyExperimentRecord], ...]:
    size = len(records)
    weighted = []
    for index, record in enumerate(records):
        age = size - index - 1
        weighted.append((max(0.35, 1.0 - 0.12 * age), record))
    return tuple(weighted)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean_label(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_text(value: Any) -> str | None:
    cleaned = _clean_label(value)
    return cleaned or None


def _human_label(value: str) -> str:
    return _clean_label(value).replace("_", " ")


def _unique(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
