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
    friction_before: int | None = None
    friction_after: int | None = None
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
            friction_before=experiment.friction_before,
            friction_after=experiment.friction_after,
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
            snapshot["policy_adjustments"] = [
                item.model_dump() for item in adjustments
            ]

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
            "empty": not helpful and not costly and not map_feedback,
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
        self.formulation_feedback = Relation("policy_formulation_feedback")
        for record in self._experiment_records:
            fact(
                self.experiment_outcome,
                record.intervention,
                record.outcome,
                record.focus,
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

        experiment_delta = 0.0
        matching_records = [
            item
            for item in self._experiment_records[-20:]
            if item.intervention == intervention
        ]
        for weight, record in _decayed(matching_records):
            record_delta = _outcome_delta(record.outcome)
            if record.friction_before is not None and record.friction_after is not None:
                record_delta += max(-0.4, min(0.4, (record.friction_before - record.friction_after) / 10))
            experiment_delta += record_delta * weight
            evidence.append(f"experiment:{record.experiment_id}:{record.outcome}")
        experiment_delta = max(-2.0, min(2.0, experiment_delta))
        if experiment_delta:
            delta += experiment_delta
            latest = matching_records[-1]
            reasons.append(_experiment_reason(latest))

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

    def _outcome_summary(self, *, positive: bool) -> list[dict[str, str]]:
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
            if positive:
                description = (
                    f"{count} positive outcome{'s' if count != 1 else ''}{focus_text}; latest was {latest.outcome}."
                )
            else:
                description = (
                    f"{count} costly outcome{'s' if count != 1 else ''}{focus_text}; latest was {latest.outcome}."
                )
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


def _experiment_reason(record: PolicyExperimentRecord) -> str:
    label = _human_label(record.intervention)
    if record.outcome == "too_hard":
        return f"Prior feedback said {label} was too hard; shrink the dose or choose a gentler route."
    if record.outcome == "did_not_help":
        return f"Prior feedback said {label} did not help; use it cautiously."
    if record.outcome == "skipped":
        return f"A prior {label} experiment was skipped; keep the next step smaller."
    if record.outcome == "helped":
        return f"Prior feedback said {label} helped."
    if record.outcome == "completed":
        return f"A prior {label} experiment was completed."
    return f"Prior feedback on {label} was {record.outcome}."


def _formulation_reason(record: PolicyFormulationRecord) -> str:
    action = "confirmed" if record.action == "confirm" else "pushed back on"
    return f"The user {action} the working-map item {_human_label(record.label)}."


def _decayed(records: list[PolicyExperimentRecord]) -> tuple[tuple[float, PolicyExperimentRecord], ...]:
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
