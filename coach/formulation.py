"""Longitudinal case formulation memory for coaching sessions.

The formulation graph is intentionally modest: it stores observations and
kernel hypotheses as tentative, user-correctable nodes with provenance. The
relational kernel remains the turn-level inference engine; this layer tracks
what appears to be recurring across turns.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from .longitudinal import (
    LongitudinalPattern,
    LongitudinalTurn,
    detect_longitudinal_patterns,
    longitudinal_turn_from_data,
)


NodeStatus = Literal["tentative", "confirmed", "rejected", "archived", "removed"]
FeedbackAction = Literal["confirm", "reject", "remove"]
DEFAULT_ACTIVE_NODE_LIMIT = 64
DEFAULT_ACTIVE_EDGE_LIMIT = 160
DEFAULT_ARCHIVE_AFTER_TURNS = 12


class FormulationProvenance(BaseModel):
    """Where one graph item came from."""

    turn: int
    source: str
    field: str
    evidence: str | None = None
    message_index: int | None = None


class FormulationNode(BaseModel):
    """One longitudinal observation, hypothesis, or intervention."""

    id: str
    kind: str
    label: str
    status: NodeStatus = "tentative"
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    first_seen_turn: int
    last_seen_turn: int
    seen_count: int = 1
    provenance: tuple[FormulationProvenance, ...] = Field(default_factory=tuple)


class FormulationEdge(BaseModel):
    """A tentative relationship between two formulation nodes."""

    id: str
    source: str
    target: str
    kind: str
    status: NodeStatus = "tentative"
    confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    first_seen_turn: int
    last_seen_turn: int
    seen_count: int = 1
    provenance: tuple[FormulationProvenance, ...] = Field(default_factory=tuple)


class FormulationGraph(BaseModel):
    """Serializable graph snapshot for API and UI use."""

    turn_count: int = 0
    nodes: tuple[FormulationNode, ...] = Field(default_factory=tuple)
    edges: tuple[FormulationEdge, ...] = Field(default_factory=tuple)
    archived_node_count: int = 0
    hidden_node_count: int = 0


class FormulationDelta(BaseModel):
    """What changed in the formulation during one turn."""

    turn: int
    summary: str
    added_nodes: tuple[FormulationNode, ...] = Field(default_factory=tuple)
    updated_nodes: tuple[FormulationNode, ...] = Field(default_factory=tuple)
    added_edges: tuple[FormulationEdge, ...] = Field(default_factory=tuple)
    updated_edges: tuple[FormulationEdge, ...] = Field(default_factory=tuple)
    longitudinal_patterns: tuple[LongitudinalPattern, ...] = Field(default_factory=tuple)
    graph: FormulationGraph


class FormulationFeedbackResult(BaseModel):
    """Graph state after user feedback."""

    node: FormulationNode
    graph: FormulationGraph


class FormulationMirror(BaseModel):
    """User-facing reflective summary generated from the working formulation."""

    text: str
    graph_turn: int
    node_ids: tuple[str, ...] = Field(default_factory=tuple)


class CaseMemory:
    """Accumulates a user-correctable working formulation across a session."""

    def __init__(
        self,
        *,
        active_node_limit: int = DEFAULT_ACTIVE_NODE_LIMIT,
        active_edge_limit: int = DEFAULT_ACTIVE_EDGE_LIMIT,
        archive_after_turns: int = DEFAULT_ARCHIVE_AFTER_TURNS,
    ) -> None:
        self.turn_count = 0
        self._nodes: dict[str, FormulationNode] = {}
        self._edges: dict[str, FormulationEdge] = {}
        self._recent_interventions: list[tuple[int, str]] = []
        self._turn_records: list[LongitudinalTurn] = []
        self._longitudinal_patterns: tuple[LongitudinalPattern, ...] = ()
        self.active_node_limit = max(1, active_node_limit)
        self.active_edge_limit = max(0, active_edge_limit)
        self.archive_after_turns = max(2, archive_after_turns)

    def export_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for app persistence."""

        return {
            "turn_count": self.turn_count,
            "nodes": [
                node.model_dump()
                for node in self.snapshot(
                    include_archived=True,
                    include_rejected=True,
                    include_removed=True,
                ).nodes
            ],
            "edges": [
                edge.model_dump()
                for edge in self.snapshot(
                    include_archived=True,
                    include_rejected=True,
                    include_removed=True,
                ).edges
            ],
            "recent_interventions": list(self._recent_interventions),
        }

    def import_state(self, data: Mapping[str, Any]) -> None:
        """Restore a snapshot produced by export_state."""

        self.turn_count = int(data.get("turn_count") or 0)
        self._nodes = {
            node.id: node
            for node in (
                FormulationNode.model_validate(item)
                for item in _values(data.get("nodes"))
            )
        }
        self._edges = {
            edge.id: edge
            for edge in (
                FormulationEdge.model_validate(item)
                for item in _values(data.get("edges"))
            )
        }
        recent = []
        for item in _values(data.get("recent_interventions")):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                turn = int(item[0])
            except (TypeError, ValueError):
                continue
            intervention = _clean_label(item[1])
            if intervention:
                recent.append((turn, intervention))
        self._recent_interventions = recent[-12:]
        self._turn_records = []
        self._longitudinal_patterns = ()

    def apply_turn(
        self,
        *,
        extraction: Mapping[str, Any] | Any,
        kernel_snapshot: Mapping[str, Any],
        response_plan: Mapping[str, Any] | Any,
        message_index: int | None = None,
    ) -> FormulationDelta:
        """Merge one completed chat turn into the working formulation."""

        self.turn_count += 1
        turn = self.turn_count
        extraction_data = _as_mapping(extraction)
        plan_data = _as_mapping(response_plan)
        added_node_ids: set[str] = set()
        updated_node_ids: set[str] = set()
        added_edge_ids: set[str] = set()
        updated_edge_ids: set[str] = set()

        def provenance(source: str, field: str, evidence: str | None = None) -> FormulationProvenance:
            return FormulationProvenance(
                turn=turn,
                source=source,
                field=field,
                evidence=evidence,
                message_index=message_index,
            )

        def upsert_node(
            kind: str,
            label: str,
            *,
            confidence: float = 0.65,
            source: str = "extraction",
            field: str,
            evidence: str | None = None,
        ) -> str | None:
            clean_label = _clean_label(label)
            if not clean_label:
                return None
            node_id = _node_id(kind, clean_label)
            item_provenance = provenance(source, field, evidence or clean_label)
            existing = self._nodes.get(node_id)
            if existing is None:
                self._nodes[node_id] = FormulationNode(
                    id=node_id,
                    kind=kind,
                    label=clean_label,
                    confidence=confidence,
                    first_seen_turn=turn,
                    last_seen_turn=turn,
                    provenance=(item_provenance,),
                )
                added_node_ids.add(node_id)
                return node_id
            if existing.status == "removed":
                return None
            if existing.status == "archived":
                existing.status = "tentative"
            existing.last_seen_turn = turn
            existing.seen_count += 1
            if existing.status == "tentative":
                existing.confidence = min(0.95, max(existing.confidence, confidence) + 0.04)
            existing.provenance = _append_provenance(existing.provenance, item_provenance)
            if node_id not in added_node_ids:
                updated_node_ids.add(node_id)
            return node_id

        def upsert_edge(
            source_id: str | None,
            target_id: str | None,
            kind: str,
            *,
            confidence: float = 0.55,
            source: str = "formulation",
            field: str,
            evidence: str | None = None,
        ) -> str | None:
            if not source_id or not target_id or source_id == target_id:
                return None
            source_node = self._nodes.get(source_id)
            target_node = self._nodes.get(target_id)
            if not source_node or not target_node:
                return None
            if source_node.status == "removed" or target_node.status == "removed":
                return None
            edge_id = _edge_id(source_id, target_id, kind)
            item_provenance = provenance(source, field, evidence)
            existing = self._edges.get(edge_id)
            if existing is None:
                self._edges[edge_id] = FormulationEdge(
                    id=edge_id,
                    source=source_id,
                    target=target_id,
                    kind=kind,
                    confidence=confidence,
                    first_seen_turn=turn,
                    last_seen_turn=turn,
                    provenance=(item_provenance,),
                )
                added_edge_ids.add(edge_id)
                return edge_id
            if existing.status == "removed":
                return None
            if existing.status == "archived":
                existing.status = "tentative"
            existing.last_seen_turn = turn
            existing.seen_count += 1
            if existing.status == "tentative":
                existing.confidence = min(0.9, max(existing.confidence, confidence) + 0.03)
            existing.provenance = _append_provenance(existing.provenance, item_provenance)
            if edge_id not in added_edge_ids:
                updated_edge_ids.add(edge_id)
            return edge_id

        situation_ids = [
            node_id
            for value in _values(extraction_data.get("situations"))
            if (
                node_id := upsert_node(
                    "situation",
                    value,
                    field="situations",
                    confidence=0.7,
                )
            )
        ]
        concern_ids = [
            node_id
            for value in _values(extraction_data.get("concerns"))
            if (
                node_id := upsert_node(
                    "concern",
                    value,
                    field="concerns",
                    confidence=0.7,
                )
            )
        ]
        task_ids = [
            node_id
            for value in _values(extraction_data.get("tasks"))
            if (
                node_id := upsert_node(
                    "task",
                    value,
                    field="tasks",
                    confidence=0.7,
                )
            )
        ]
        challenge_ids = [
            node_id
            for value in _values(extraction_data.get("challenges"))
            if (
                node_id := upsert_node(
                    "challenge",
                    value,
                    field="challenges",
                    confidence=0.66,
                )
            )
        ]
        objective_ids = [
            node_id
            for value in _values(extraction_data.get("objectives"))
            if (
                node_id := upsert_node(
                    "objective",
                    value,
                    field="objectives",
                    confidence=0.68,
                )
            )
        ]
        project_ids = [
            node_id
            for value in _values(extraction_data.get("projects"))
            if (
                node_id := upsert_node(
                    "project",
                    value,
                    field="projects",
                    confidence=0.68,
                )
            )
        ]
        key_result_ids = [
            node_id
            for value in _values(extraction_data.get("key_results"))
            if (
                node_id := upsert_node(
                    "key_result",
                    value,
                    field="key_results",
                    confidence=0.64,
                )
            )
        ]
        next_action_ids = [
            node_id
            for value in _values(extraction_data.get("next_actions"))
            if (
                node_id := upsert_node(
                    "next_action",
                    value,
                    field="next_actions",
                    confidence=0.72,
                )
            )
        ]
        obstacle_ids = [
            node_id
            for value in _values(extraction_data.get("obstacles"))
            if (
                node_id := upsert_node(
                    "obstacle",
                    value,
                    field="obstacles",
                    confidence=0.66,
                )
            )
        ]
        implementation_intention_ids = [
            node_id
            for value in _values(extraction_data.get("implementation_intentions"))
            if (
                node_id := upsert_node(
                    "implementation_intention",
                    value,
                    field="implementation_intentions",
                    confidence=0.68,
                )
            )
        ]
        waiting_for_ids = [
            node_id
            for value in _values(extraction_data.get("waiting_for"))
            if (
                node_id := upsert_node(
                    "waiting_for",
                    value,
                    field="waiting_for",
                    confidence=0.64,
                )
            )
        ]
        time_horizon_ids = [
            node_id
            for value in _values(extraction_data.get("time_horizons"))
            if (
                node_id := upsert_node(
                    "time_horizon",
                    value,
                    field="time_horizons",
                    confidence=0.62,
                )
            )
        ]
        success_measure_ids = [
            node_id
            for value in _values(extraction_data.get("success_measures"))
            if (
                node_id := upsert_node(
                    "success_measure",
                    value,
                    field="success_measures",
                    confidence=0.66,
                )
            )
        ]
        stake_ids = [
            node_id
            for value in _values(extraction_data.get("stakes"))
            if (
                node_id := upsert_node(
                    "stake",
                    value,
                    field="stakes",
                    confidence=0.64,
                )
            )
        ]
        domain_ids = [
            node_id
            for value in _values(extraction_data.get("domains"))
            if (
                node_id := upsert_node(
                    "domain",
                    value,
                    field="domains",
                    confidence=0.62,
                )
            )
        ]
        thought_ids = [
            node_id
            for value in _values(extraction_data.get("thoughts"))
            if (
                node_id := upsert_node(
                    "thought",
                    value,
                    field="thoughts",
                    confidence=0.68,
                )
            )
        ]
        belief_ids = [
            node_id
            for value in _values(extraction_data.get("beliefs"))
            if (
                node_id := upsert_node(
                    "belief",
                    value,
                    field="beliefs",
                    confidence=0.66,
                )
            )
        ]
        emotion_ids = [
            node_id
            for value in _values(extraction_data.get("emotions"))
            if (
                node_id := upsert_node(
                    "emotion",
                    value,
                    field="emotions",
                    confidence=0.72,
                )
            )
        ]
        urge_ids = [
            node_id
            for value in _values(extraction_data.get("urges"))
            if (
                node_id := upsert_node(
                    "urge",
                    value,
                    field="urges",
                    confidence=0.62,
                )
            )
        ]
        behavior_ids = [
            node_id
            for value in _values(extraction_data.get("behaviors"))
            if (
                node_id := upsert_node(
                    "behavior",
                    value,
                    field="behaviors",
                    confidence=0.7,
                )
            )
        ]
        consequence_ids = [
            node_id
            for value in _values(extraction_data.get("consequences"))
            if (
                node_id := upsert_node(
                    "consequence",
                    value,
                    field="consequences",
                    confidence=0.64,
                )
            )
        ]
        value_ids = [
            node_id
            for value in _values(extraction_data.get("values"))
            if (
                node_id := upsert_node(
                    "value",
                    value,
                    field="values",
                    confidence=0.62,
                )
            )
        ]
        goal_ids = [
            node_id
            for value in _values(extraction_data.get("goals"))
            if (
                node_id := upsert_node(
                    "goal",
                    value,
                    field="goals",
                    confidence=0.68,
                )
            )
        ]
        feature_ids = [
            node_id
            for value in _values(extraction_data.get("features"))
            if (
                node_id := upsert_node(
                    "feature",
                    value,
                    field="features",
                    confidence=0.6,
                )
            )
        ]

        text_node_ids = {
            _normalize_evidence_text(label): node_id
            for label, node_id in [
                *zip(_values(extraction_data.get("thoughts")), thought_ids, strict=False),
                *zip(_values(extraction_data.get("beliefs")), belief_ids, strict=False),
            ]
        }
        for field in ("thought_features", "belief_features"):
            for item in _values(extraction_data.get(field)):
                item_data = _as_mapping(item)
                text = _clean_label(item_data.get("text"))
                text_id = text_node_ids.get(_normalize_evidence_text(text))
                for feature in _values(item_data.get("features")):
                    feature_id = upsert_node(
                        "feature",
                        feature,
                        field=field,
                        confidence=0.62,
                    )
                    if feature_id:
                        feature_ids.append(feature_id)
                    upsert_edge(
                        text_id,
                        feature_id,
                        "has_feature",
                        source="extraction",
                        field=field,
                        evidence=text,
                    )

        for situation_id in situation_ids[:2]:
            for target_id in [
                *concern_ids,
                *task_ids,
                *project_ids,
                *next_action_ids,
                *thought_ids,
                *belief_ids,
                *emotion_ids,
                *behavior_ids,
                *goal_ids,
            ][:10]:
                upsert_edge(
                    situation_id,
                    target_id,
                    "context_for",
                    field="situations",
                    evidence="same turn",
                )
        for domain_id in domain_ids[:4]:
            for target_id in [*concern_ids, *task_ids, *project_ids, *objective_ids, *stake_ids, *goal_ids][:8]:
                upsert_edge(
                    domain_id,
                    target_id,
                    "domain_of",
                    field="domains",
                    evidence="same turn",
                    confidence=0.52,
                )
        for concern_id in concern_ids[:4]:
            for target_id in [*emotion_ids, *thought_ids, *belief_ids][:8]:
                upsert_edge(
                    concern_id,
                    target_id,
                    "about",
                    field="concerns",
                    evidence="same turn",
                    confidence=0.55,
                )
            for task_id in task_ids[:4]:
                upsert_edge(
                    concern_id,
                    task_id,
                    "involves_task",
                    field="concerns",
                    evidence="same turn",
                    confidence=0.54,
                )
            for project_id in project_ids[:4]:
                upsert_edge(
                    concern_id,
                    project_id,
                    "involves_project",
                    field="concerns",
                    evidence="same turn",
                    confidence=0.54,
                )
            for stake_id in stake_ids[:4]:
                upsert_edge(
                    stake_id,
                    concern_id,
                    "raises_stakes_for",
                    field="stakes",
                    evidence="same turn",
                    confidence=0.56,
                )
        for task_id in task_ids[:4]:
            for objective_id in objective_ids[:4]:
                upsert_edge(
                    task_id,
                    objective_id,
                    "aims_at",
                    field="tasks",
                    evidence="same turn",
                    confidence=0.58,
                )
            for goal_id in goal_ids[:4]:
                upsert_edge(
                    task_id,
                    goal_id,
                    "serves_goal",
                    field="tasks",
                    evidence="same turn",
                    confidence=0.56,
                )
            for next_action_id in next_action_ids[:4]:
                upsert_edge(
                    next_action_id,
                    task_id,
                    "advances_task",
                    field="next_actions",
                    evidence="same turn",
                    confidence=0.62,
                )
        for project_id in project_ids[:4]:
            for target_id in [*objective_ids, *goal_ids][:6]:
                upsert_edge(
                    project_id,
                    target_id,
                    "serves_direction",
                    field="projects",
                    evidence="same turn",
                    confidence=0.56,
                )
            for next_action_id in next_action_ids[:4]:
                upsert_edge(
                    next_action_id,
                    project_id,
                    "advances_project",
                    field="next_actions",
                    evidence="same turn",
                    confidence=0.62,
                )
        for challenge_id in challenge_ids[:4]:
            for target_id in [*task_ids, *project_ids, *objective_ids, *goal_ids, *behavior_ids][:8]:
                upsert_edge(
                    challenge_id,
                    target_id,
                    "blocks_or_complicates",
                    field="challenges",
                    evidence="same turn",
                    confidence=0.58,
                )
        for obstacle_id in obstacle_ids[:4]:
            for target_id in [*next_action_ids, *task_ids, *project_ids, *objective_ids, *goal_ids][:8]:
                upsert_edge(
                    obstacle_id,
                    target_id,
                    "may_block",
                    field="obstacles",
                    evidence="same turn",
                    confidence=0.58,
                )
            for intention_id in implementation_intention_ids[:3]:
                upsert_edge(
                    intention_id,
                    obstacle_id,
                    "plans_for",
                    field="implementation_intentions",
                    evidence="same turn",
                    confidence=0.62,
                )
        for objective_id in objective_ids[:4]:
            for goal_id in goal_ids[:4]:
                upsert_edge(
                    objective_id,
                    goal_id,
                    "refines_goal",
                    field="objectives",
                    evidence="same turn",
                    confidence=0.54,
                )
            for key_result_id in key_result_ids[:4]:
                upsert_edge(
                    key_result_id,
                    objective_id,
                    "measures_objective",
                    field="key_results",
                    evidence="same turn",
                    confidence=0.6,
                )
            for measure_id in success_measure_ids[:4]:
                upsert_edge(
                    measure_id,
                    objective_id,
                    "measures_objective",
                    field="success_measures",
                    evidence="same turn",
                    confidence=0.58,
                )
            for horizon_id in time_horizon_ids[:3]:
                upsert_edge(
                    objective_id,
                    horizon_id,
                    "has_horizon",
                    field="time_horizons",
                    evidence="same turn",
                    confidence=0.54,
                )
        for waiting_for_id in waiting_for_ids[:4]:
            for target_id in [*next_action_ids, *task_ids, *project_ids][:8]:
                upsert_edge(
                    target_id,
                    waiting_for_id,
                    "waiting_for",
                    field="waiting_for",
                    evidence="same turn",
                    confidence=0.56,
                )
        for stake_id in stake_ids[:4]:
            for target_id in [*task_ids, *objective_ids, *goal_ids][:8]:
                upsert_edge(
                    stake_id,
                    target_id,
                    "raises_stakes_for",
                    field="stakes",
                    evidence="same turn",
                    confidence=0.56,
                )
        for source_id in [*thought_ids, *belief_ids][:6]:
            for emotion_id in emotion_ids[:4]:
                upsert_edge(source_id, emotion_id, "may_trigger", field="extraction")
        for emotion_id in emotion_ids[:4]:
            for urge_id in urge_ids[:3]:
                upsert_edge(emotion_id, urge_id, "shows_up_as", field="extraction")
        for urge_id in urge_ids[:3]:
            for behavior_id in behavior_ids[:4]:
                upsert_edge(urge_id, behavior_id, "can_lead_to", field="extraction")
        for behavior_id in behavior_ids[:4]:
            for consequence_id in consequence_ids[:3]:
                upsert_edge(behavior_id, consequence_id, "leads_to", field="extraction")
        for value_id in value_ids[:4]:
            for target_id in [*goal_ids, *objective_ids, *project_ids][:6]:
                upsert_edge(value_id, target_id, "orients", field="extraction")

        hypothesis_ids_by_key: dict[tuple[str, str], str] = {}
        for hypothesis in _values(kernel_snapshot.get("hypotheses")):
            hypothesis_data = _as_mapping(hypothesis)
            hypothesis_source = _clean_label(hypothesis_data.get("source"))
            pattern = _clean_label(hypothesis_data.get("pattern"))
            if not hypothesis_source or not pattern:
                continue
            label = f"{hypothesis_source}: {pattern}"
            hypothesis_id = upsert_node(
                "hypothesis",
                label,
                source="kernel",
                field="hypotheses",
                evidence=pattern,
                confidence=0.58,
            )
            if not hypothesis_id:
                continue
            hypothesis_ids_by_key[(hypothesis_source, pattern)] = hypothesis_id
            for support_id in [
                *thought_ids,
                *belief_ids,
                *emotion_ids,
                *behavior_ids,
                *challenge_ids,
                *obstacle_ids,
                *project_ids,
                *next_action_ids,
                *stake_ids,
                *feature_ids,
            ][:10]:
                upsert_edge(
                    support_id,
                    hypothesis_id,
                    "supports_hypothesis",
                    source="kernel",
                    field="hypotheses",
                    evidence=pattern,
                )

        intervention = _clean_label(plan_data.get("intervention"))
        if intervention:
            intervention_id = upsert_node(
                "intervention",
                intervention,
                source="response_plan",
                field="intervention",
                evidence=plan_data.get("exercise") or plan_data.get("question"),
                confidence=0.7,
            )
            self._recent_interventions.append((turn, intervention))
            self._recent_interventions = self._recent_interventions[-12:]
            selected_candidate = _candidate_for_intervention(
                kernel_snapshot.get("candidates") or (),
                intervention,
            )
            selected_hypotheses = (
                selected_candidate.get("hypotheses")
                if selected_candidate
                else kernel_snapshot.get("hypotheses")
            ) or ()
            for hypothesis in _values(selected_hypotheses):
                hypothesis_data = _as_mapping(hypothesis)
                key = (
                    _clean_label(hypothesis_data.get("source")),
                    _clean_label(hypothesis_data.get("pattern")),
                )
                upsert_edge(
                    hypothesis_ids_by_key.get(key),
                    intervention_id,
                    "supports_intervention",
                    source="response_plan",
                    field="intervention",
                    evidence=intervention,
                    confidence=0.68,
                )

        turn_record = longitudinal_turn_from_data(
            turn=turn,
            extraction=extraction_data,
            kernel_snapshot=kernel_snapshot,
            response_plan=plan_data,
        )
        self._turn_records.append(turn_record)
        self._turn_records = self._turn_records[-12:]
        self._longitudinal_patterns = detect_longitudinal_patterns(
            tuple(self._turn_records)
        )
        for pattern in self._longitudinal_patterns:
            pattern_id = upsert_node(
                "longitudinal_pattern",
                pattern.label,
                source="longitudinal_kernel",
                field="patterns",
                evidence=pattern.description,
                confidence=pattern.confidence,
            )
            if pattern_id:
                pattern_node = self._nodes[pattern_id]
                pattern_node.first_seen_turn = min(pattern.turns)
                pattern_node.last_seen_turn = max(pattern.turns)
                pattern_node.seen_count = max(pattern_node.seen_count, len(pattern.turns))
            for support in pattern.support:
                support_id = _node_id(support.kind, support.label)
                upsert_edge(
                    support_id,
                    pattern_id,
                    "supports_longitudinal_pattern",
                    source="longitudinal_kernel",
                    field=pattern.pattern,
                    evidence=f"turn {support.turn}: {support.label}",
                    confidence=0.6,
                )

        self.compact()
        graph = self.snapshot()
        added_nodes = tuple(self._nodes[node_id] for node_id in sorted(added_node_ids))
        updated_nodes = tuple(self._nodes[node_id] for node_id in sorted(updated_node_ids))
        added_edges = tuple(self._edges[edge_id] for edge_id in sorted(added_edge_ids))
        updated_edges = tuple(self._edges[edge_id] for edge_id in sorted(updated_edge_ids))
        return FormulationDelta(
            turn=turn,
            summary=_delta_summary(added_nodes, updated_nodes, added_edges, updated_edges),
            added_nodes=added_nodes,
            updated_nodes=updated_nodes,
            added_edges=added_edges,
            updated_edges=updated_edges,
            longitudinal_patterns=self._longitudinal_patterns,
            graph=graph,
        )

    def apply_feedback(
        self,
        node_id: str,
        action: FeedbackAction,
        *,
        note: str | None = None,
    ) -> FormulationFeedbackResult:
        """Apply user correction to a node and any dependent edges."""

        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(node_id)

        feedback_provenance = FormulationProvenance(
            turn=self.turn_count,
            source="user_feedback",
            field=action,
            evidence=note,
        )
        if action == "confirm":
            node.status = "confirmed"
            node.confidence = max(node.confidence, 0.9)
        elif action == "reject":
            node.status = "rejected"
            node.confidence = min(node.confidence, 0.35)
        elif action == "remove":
            node.status = "removed"
            node.confidence = 0.0
            for edge in self._edges.values():
                if edge.source == node_id or edge.target == node_id:
                    edge.status = "removed"
                    edge.confidence = 0.0
        else:  # pragma: no cover - protected by pydantic/API types
            raise ValueError(f"Unknown feedback action: {action}")
        node.provenance = _append_provenance(node.provenance, feedback_provenance)
        return FormulationFeedbackResult(node=node.model_copy(deep=True), graph=self.snapshot())

    def compact(self) -> FormulationGraph:
        """Archive stale and low-priority map items while keeping provenance."""

        if not self._nodes:
            return self.snapshot()

        for node in self._nodes.values():
            if self._should_archive_as_stale(node):
                node.status = "archived"

        active_nodes = [
            node
            for node in self._nodes.values()
            if node.status not in {"archived", "rejected", "removed"}
        ]
        protected_ids = {
            node.id
            for node in active_nodes
            if self._is_protected_active_node(node)
        }
        available_slots = max(self.active_node_limit - len(protected_ids), 0)
        optional_nodes = [
            node
            for node in active_nodes
            if node.id not in protected_ids
        ]
        optional_nodes.sort(key=self._node_priority, reverse=True)
        keep_ids = protected_ids | {
            node.id
            for node in optional_nodes[:available_slots]
        }
        for node in optional_nodes[available_slots:]:
            node.status = "archived"

        active_node_ids = {
            node.id
            for node in self._nodes.values()
            if node.status not in {"archived", "rejected", "removed"}
        }
        for edge in self._edges.values():
            if edge.status == "removed":
                continue
            if edge.source not in active_node_ids or edge.target not in active_node_ids:
                edge.status = "archived"
            elif edge.status == "archived":
                edge.status = "tentative"

        active_edges = [
            edge
            for edge in self._edges.values()
            if edge.status not in {"archived", "rejected", "removed"}
            and edge.source in active_node_ids
            and edge.target in active_node_ids
        ]
        active_edges.sort(key=self._edge_priority, reverse=True)
        keep_edge_ids = {
            edge.id
            for edge in active_edges[: self.active_edge_limit]
        }
        for edge in active_edges[self.active_edge_limit:]:
            if edge.id not in keep_edge_ids:
                edge.status = "archived"

        return self.snapshot()

    def snapshot(
        self,
        *,
        include_archived: bool = False,
        include_rejected: bool = False,
        include_removed: bool = False,
    ) -> FormulationGraph:
        """Return a stable graph snapshot."""

        hidden_statuses = set()
        if not include_archived:
            hidden_statuses.add("archived")
        if not include_rejected:
            hidden_statuses.add("rejected")
        if not include_removed:
            hidden_statuses.add("removed")
        nodes = tuple(
            sorted(
                (
                    node
                    for node in self._nodes.values()
                    if node.status not in hidden_statuses
                ),
                key=lambda item: (item.kind, item.label.casefold(), item.id),
            )
        )
        visible_node_ids = {node.id for node in nodes}
        edges = tuple(
            sorted(
                (
                    edge
                    for edge in self._edges.values()
                    if edge.status not in hidden_statuses
                    and edge.source in visible_node_ids
                    and edge.target in visible_node_ids
                ),
                key=lambda item: (item.kind, item.source, item.target),
            )
        )
        return FormulationGraph(
            turn_count=self.turn_count,
            nodes=nodes,
            edges=edges,
            archived_node_count=sum(
                1 for node in self._nodes.values() if node.status == "archived"
            ),
            hidden_node_count=sum(
                1
                for node in self._nodes.values()
                if node.status in {"archived", "rejected", "removed"}
            ),
        )

    def _should_archive_as_stale(self, node: FormulationNode) -> bool:
        if node.status != "tentative":
            return False
        if self._is_protected_active_node(node):
            return False
        age = self.turn_count - node.last_seen_turn
        return age >= self.archive_after_turns and node.seen_count <= 1

    def _is_protected_active_node(self, node: FormulationNode) -> bool:
        if node.status == "confirmed":
            return True
        if node.kind == "longitudinal_pattern":
            return True
        if node.seen_count >= 2:
            return True
        return node.last_seen_turn == self.turn_count

    def _node_priority(self, node: FormulationNode) -> tuple[float, int, int, str]:
        age = max(self.turn_count - node.last_seen_turn, 0)
        recency = max(self.archive_after_turns - age, 0)
        score = (
            _kind_priority(node.kind)
            + 8.0 * node.confidence
            + 4.0 * min(node.seen_count, 4)
            + 0.4 * recency
        )
        return (score, node.last_seen_turn, node.seen_count, node.id)

    def _edge_priority(self, edge: FormulationEdge) -> tuple[float, int, int, str]:
        age = max(self.turn_count - edge.last_seen_turn, 0)
        recency = max(self.archive_after_turns - age, 0)
        score = (
            _edge_kind_priority(edge.kind)
            + 7.0 * edge.confidence
            + 3.0 * min(edge.seen_count, 4)
            + 0.35 * recency
        )
        return (score, edge.last_seen_turn, edge.seen_count, edge.id)

    def recent_interventions(self, *, limit: int = 3) -> tuple[str, ...]:
        """Most recent interventions, newest last, for kernel repetition penalties."""

        interventions = []
        seen = set()
        for _turn, intervention in reversed(self._recent_interventions):
            if intervention not in seen:
                seen.add(intervention)
                interventions.append(intervention)
            if len(interventions) >= limit:
                break
        return tuple(reversed(interventions))

    def longitudinal_patterns(self) -> tuple[LongitudinalPattern, ...]:
        """Current tentative multi-turn patterns."""

        return tuple(pattern.model_copy(deep=True) for pattern in self._longitudinal_patterns)

    def longitudinal_context(self, *, limit: int = 4) -> str:
        """Compact context string for the response planner."""

        patterns = self._longitudinal_patterns[:limit]
        if not patterns:
            return ""
        lines = ["Tentative multi-turn patterns from prior turns:"]
        for pattern in patterns:
            turns = ", ".join(str(turn) for turn in pattern.turns)
            lines.append(f"- {pattern.label}: {pattern.description} (turns {turns})")
        lines.append("Use these only as hypotheses to hold lightly and invite correction.")
        return "\n".join(lines)

    def mirror(self) -> FormulationMirror:
        """Create a reflective playback of the current working formulation."""

        return mirror_formulation(self.snapshot())


def mirror_formulation(graph: FormulationGraph) -> FormulationMirror:
    """Render the graph as a tentative reflective-listening summary."""

    nodes = [
        node
        for node in graph.nodes
        if node.status not in {"archived", "rejected", "removed"}
    ]
    if not nodes:
        return FormulationMirror(
            text=(
                "I do not have enough of a working map yet to mirror anything back. "
                "After a turn or two, I can reflect the pattern I am hearing and you can correct it."
            ),
            graph_turn=graph.turn_count,
        )

    by_kind = _nodes_by_kind(nodes)
    selected = []

    domains = _top_labels(by_kind, "domain", limit=2, selected=selected)
    situations = _top_labels(by_kind, "situation", limit=2, selected=selected)
    concerns = _top_labels(by_kind, "concern", limit=2, selected=selected)
    tasks = _top_labels(by_kind, "task", limit=2, selected=selected)
    challenges = _top_labels(by_kind, "challenge", limit=2, selected=selected)
    objectives = _top_labels(by_kind, "objective", limit=2, selected=selected)
    projects = _top_labels(by_kind, "project", limit=2, selected=selected)
    next_actions = _top_labels(by_kind, "next_action", limit=2, selected=selected)
    obstacles = _top_labels(by_kind, "obstacle", limit=2, selected=selected)
    success_measures = _top_labels(by_kind, "success_measure", limit=2, selected=selected)
    stakes = _top_labels(by_kind, "stake", limit=2, selected=selected)
    thoughts = _top_labels(by_kind, "thought", limit=2, selected=selected)
    beliefs = _top_labels(by_kind, "belief", limit=2, selected=selected)
    emotions = _top_labels(by_kind, "emotion", limit=3, selected=selected)
    urges = _top_labels(by_kind, "urge", limit=2, selected=selected)
    behaviors = _top_labels(by_kind, "behavior", limit=3, selected=selected)
    values = _top_labels(by_kind, "value", limit=2, selected=selected)
    goals = _top_labels(by_kind, "goal", limit=2, selected=selected)
    hypotheses = _top_labels(by_kind, "hypothesis", limit=3, selected=selected)
    interventions = _top_labels(by_kind, "intervention", limit=2, selected=selected)

    lines = ["If I mirror this back as a working hypothesis:"]
    if domains:
        lines.append(f"The area of life or work that stands out is {_join_labels(domains)}.")
    if situations:
        lines.append(f"The context that stands out is {_join_labels(situations)}.")
    if concerns or tasks or objectives or projects:
        focus_parts = []
        if concerns:
            focus_parts.append(f"the concern around {_join_labels(concerns)}")
        if tasks:
            focus_parts.append(f"the concrete task of {_join_labels(tasks)}")
        if objectives:
            focus_parts.append(f"the aim of {_join_labels(objectives)}")
        if projects:
            focus_parts.append(f"the project of {_join_labels(projects)}")
        lines.append(f"The focus seems to be {_join_labels(focus_parts)}.")
    if next_actions or obstacles or success_measures:
        execution_parts = []
        if next_actions:
            execution_parts.append(f"next action: {_join_labels(next_actions)}")
        if obstacles:
            execution_parts.append(f"obstacle: {_join_labels(obstacles)}")
        if success_measures:
            execution_parts.append(f"measure: {_join_labels(success_measures)}")
        lines.append(f"The execution map is holding {_join_labels(execution_parts)}.")
    if challenges or stakes:
        pressure_parts = []
        if challenges:
            pressure_parts.append(f"the friction of {_join_labels(challenges)}")
        if stakes:
            pressure_parts.append(f"the stakes around {_join_labels(stakes)}")
        lines.append(f"What gives it pressure may be {_join_labels(pressure_parts)}.")
    if thoughts or beliefs:
        lines.append(f"Around that, the mind seems to offer {_quote_labels([*thoughts, *beliefs])}.")
    if emotions:
        lines.append(f"Emotionally, the map is carrying {_join_labels(emotions)}.")
    if urges or behaviors:
        lines.append(f"The pull seems to be toward {_join_labels([*urges, *behaviors])}.")
    if values or goals:
        lines.append(
            "At the same time, "
            f"{_join_labels([*values, *goals])} still appears to matter."
        )
    if hypotheses:
        lines.append(
            "The tentative pattern I would hold lightly is "
            f"{_join_labels([_clean_hypothesis(label) for label in hypotheses])}."
        )
    if interventions:
        lines.append(
            "So the response has been leaning toward "
            f"{_join_labels([_humanize(label) for label in interventions])}."
        )
    lines.append("Any part of that may be wrong; the useful move is to correct the map.")

    return FormulationMirror(
        text="\n\n".join(lines),
        graph_turn=graph.turn_count,
        node_ids=tuple(node.id for node in selected),
    )


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


def _candidate_for_intervention(
    candidates: Any,
    intervention: str,
) -> Mapping[str, Any] | None:
    for candidate in _values(candidates):
        candidate_data = _as_mapping(candidate)
        if _clean_label(candidate_data.get("intervention")) == intervention:
            return candidate_data
    return None


def _nodes_by_kind(nodes: list[FormulationNode]) -> dict[str, list[FormulationNode]]:
    grouped: dict[str, list[FormulationNode]] = {}
    for node in nodes:
        grouped.setdefault(node.kind, []).append(node)
    for items in grouped.values():
        items.sort(
            key=lambda node: (
                node.status != "confirmed",
                -node.seen_count,
                -node.confidence,
                node.label.casefold(),
            )
        )
    return grouped


def _top_labels(
    grouped: dict[str, list[FormulationNode]],
    kind: str,
    *,
    limit: int,
    selected: list[FormulationNode],
) -> list[str]:
    labels = []
    for node in grouped.get(kind, ())[:limit]:
        labels.append(_humanize(node.label))
        selected.append(node)
    return labels


def _join_labels(labels: list[str]) -> str:
    cleaned = [label for label in labels if label]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _quote_labels(labels: list[str]) -> str:
    return _join_labels([f'"{label}"' for label in labels if label])


def _clean_hypothesis(label: str) -> str:
    cleaned = _humanize(label)
    if ":" not in cleaned:
        return cleaned
    source, pattern = cleaned.split(":", 1)
    source = source.strip().upper() if source.strip() in {"act", "cbt", "rebt"} else source.strip()
    return f"{source} {pattern.strip()}"


def _humanize(label: str) -> str:
    return str(label).replace("_", " ").strip()


def _kind_priority(kind: str) -> float:
    return {
        "longitudinal_pattern": 60.0,
        "objective": 46.0,
        "project": 45.5,
        "task": 45.0,
        "next_action": 44.0,
        "obstacle": 43.0,
        "key_result": 42.5,
        "belief": 42.0,
        "hypothesis": 40.0,
        "challenge": 39.0,
        "behavior": 38.0,
        "concern": 37.0,
        "emotion": 36.0,
        "value": 35.0,
        "goal": 35.0,
        "success_measure": 34.0,
        "implementation_intention": 33.0,
        "thought": 32.0,
        "time_horizon": 31.5,
        "stake": 31.0,
        "waiting_for": 30.5,
        "intervention": 30.0,
        "urge": 28.0,
        "domain": 26.0,
        "feature": 24.0,
        "situation": 20.0,
        "consequence": 18.0,
    }.get(kind, 16.0)


def _edge_kind_priority(kind: str) -> float:
    return {
        "supports_longitudinal_pattern": 50.0,
        "supports_intervention": 42.0,
        "supports_hypothesis": 40.0,
        "blocks_or_complicates": 38.0,
        "aims_at": 36.0,
        "advances_project": 36.0,
        "advances_task": 36.0,
        "measures_objective": 35.5,
        "raises_stakes_for": 35.0,
        "may_block": 34.0,
        "plans_for": 33.0,
        "context_for": 32.0,
        "serves_direction": 28.0,
        "may_trigger": 30.0,
        "can_lead_to": 28.0,
        "shows_up_as": 26.0,
        "orients": 25.0,
        "about": 25.0,
        "has_feature": 24.0,
        "serves_goal": 24.0,
        "refines_goal": 23.0,
        "leads_to": 22.0,
        "involves_task": 22.0,
        "involves_project": 22.0,
        "waiting_for": 22.0,
        "has_horizon": 21.0,
        "domain_of": 20.0,
    }.get(kind, 18.0)


def _node_id(kind: str, label: str) -> str:
    digest = hashlib.sha1(f"{kind}:{_normalize_evidence_text(label)}".encode()).hexdigest()
    return f"{kind}:{digest[:12]}"


def _edge_id(source: str, target: str, kind: str) -> str:
    digest = hashlib.sha1(f"{source}:{kind}:{target}".encode()).hexdigest()
    return f"edge:{digest[:12]}"


def _clean_label(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_evidence_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def _append_provenance(
    existing: tuple[FormulationProvenance, ...],
    item: FormulationProvenance,
    *,
    limit: int = 8,
) -> tuple[FormulationProvenance, ...]:
    key = (item.turn, item.source, item.field, item.evidence)
    if any((prov.turn, prov.source, prov.field, prov.evidence) == key for prov in existing):
        return existing
    return (*existing, item)[-limit:]


def _delta_summary(
    added_nodes: tuple[FormulationNode, ...],
    updated_nodes: tuple[FormulationNode, ...],
    added_edges: tuple[FormulationEdge, ...],
    updated_edges: tuple[FormulationEdge, ...],
) -> str:
    additions = len(added_nodes)
    updates = len(updated_nodes)
    recurring = sum(1 for node in updated_nodes if node.seen_count > 1)
    relationships = len(added_edges) + len(updated_edges)
    parts = []
    if additions:
        parts.append(f"{additions} new map item{'s' if additions != 1 else ''}")
    if recurring:
        parts.append(f"{recurring} recurring item{'s' if recurring != 1 else ''}")
    elif updates:
        parts.append(f"{updates} updated item{'s' if updates != 1 else ''}")
    if relationships:
        parts.append(f"{relationships} relationship{'s' if relationships != 1 else ''}")
    return "; ".join(parts) if parts else "No formulation changes"
