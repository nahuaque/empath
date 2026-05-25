"""Closed-loop coaching experiments generated from kernel-guided turns."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from .formulation import FormulationDelta


ExperimentStatus = Literal["proposed", "reviewed", "skipped"]
ExperimentFeedbackAction = Literal[
    "completed",
    "helped",
    "neutral",
    "did_not_help",
    "too_hard",
    "skipped",
]


class CoachingExperiment(BaseModel):
    """A small N-of-1 test attached to one assistant turn."""

    id: str
    status: ExperimentStatus = "proposed"
    created_turn: int
    message_index: int | None = None
    intervention: str
    focus: str
    title: str
    hypothesis: str
    action: str
    prediction: str
    measure: str
    timebox: str = "10 minutes"
    rationale: str
    support_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    outcome: ExperimentFeedbackAction | None = None
    note: str | None = None
    friction_before: int | None = Field(default=None, ge=0, le=10)
    friction_after: int | None = Field(default=None, ge=0, le=10)
    learning: str | None = None


class ExperimentFeedbackResult(BaseModel):
    """Result of closing the loop on one proposed experiment."""

    experiment: CoachingExperiment
    experiments: tuple[CoachingExperiment, ...]
    learning: str


class ExperimentStore:
    """In-memory store for proposed experiments and user outcome feedback."""

    def __init__(self) -> None:
        self._experiments: dict[str, CoachingExperiment] = {}

    def export_state(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot for app persistence."""

        return [item.model_dump() for item in self.snapshot()]

    def import_state(self, data: Any) -> None:
        """Restore experiments from export_state output."""

        self._experiments = {
            item.id: item
            for item in (
                CoachingExperiment.model_validate(raw)
                for raw in (data or ())
            )
        }

    def propose(
        self,
        *,
        turn: Any,
        formulation_delta: FormulationDelta,
        message_index: int | None,
    ) -> CoachingExperiment:
        experiment = propose_experiment(
            turn=turn,
            formulation_delta=formulation_delta,
            message_index=message_index,
        )
        self._experiments[experiment.id] = experiment
        return experiment

    def apply_feedback(
        self,
        experiment_id: str,
        action: ExperimentFeedbackAction,
        *,
        note: str | None = None,
        friction_before: int | None = None,
        friction_after: int | None = None,
    ) -> ExperimentFeedbackResult:
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)

        experiment.status = "skipped" if action == "skipped" else "reviewed"
        experiment.outcome = action
        experiment.note = _clean_text(note)
        experiment.friction_before = friction_before
        experiment.friction_after = friction_after
        experiment.learning = _learning_for_feedback(experiment, action)
        return ExperimentFeedbackResult(
            experiment=experiment.model_copy(deep=True),
            experiments=self.snapshot(),
            learning=experiment.learning,
        )

    def snapshot(self) -> tuple[CoachingExperiment, ...]:
        return tuple(
            sorted(
                (item.model_copy(deep=True) for item in self._experiments.values()),
                key=lambda item: (item.created_turn, item.message_index or 0, item.id),
            )
        )


def propose_experiment(
    *,
    turn: Any,
    formulation_delta: FormulationDelta,
    message_index: int | None,
) -> CoachingExperiment:
    """Create one tiny behavioral experiment from the selected turn plan."""

    plan = getattr(turn, "response_plan", None)
    prepared = getattr(turn, "prepared", None)
    kernel_snapshot = getattr(prepared, "kernel_snapshot", {}) or {}
    intervention = _clean_label(getattr(plan, "intervention", None)) or "gentle_check_in"
    plan_exercise = _clean_text(getattr(plan, "exercise", None))
    plan_question = _clean_text(getattr(plan, "question", None))
    selected = _candidate_for_intervention(kernel_snapshot, intervention)
    selected_hypotheses = _hypotheses(selected) or tuple(
        _as_mapping(item)
        for item in (kernel_snapshot.get("hypotheses") or ())
    )
    pattern = _primary_pattern(selected_hypotheses)
    emotion = _primary_emotion(getattr(prepared, "extraction", None), selected_hypotheses)
    focus = _focus_label(selected_hypotheses, pattern)
    template = _template_for(intervention, pattern)
    action = _action_for(
        intervention,
        _format_template(str(template["action"]), pattern=pattern, emotion=emotion),
        plan_exercise,
        plan_question,
    )
    support_node_ids = _support_nodes(
        formulation_delta=formulation_delta,
        intervention=intervention,
        hypotheses=selected_hypotheses,
    )
    title = str(template["title"])
    experiment_id = _experiment_id(
        formulation_delta.turn,
        message_index,
        intervention,
        pattern,
        title,
    )

    return CoachingExperiment(
        id=experiment_id,
        created_turn=formulation_delta.turn,
        message_index=message_index,
        intervention=intervention,
        focus=focus,
        title=title,
        hypothesis=_format_template(
            str(template["hypothesis"]),
            pattern=pattern,
            emotion=emotion,
        ),
        action=action,
        prediction=_format_template(
            str(template["prediction"]),
            pattern=pattern,
            emotion=emotion,
        ),
        measure=str(template["measure"]),
        timebox=str(template["timebox"]),
        rationale=_rationale(intervention, selected_hypotheses),
        support_node_ids=support_node_ids,
    )


def _candidate_for_intervention(
    kernel_snapshot: Mapping[str, Any],
    intervention: str,
) -> Mapping[str, Any]:
    candidates = kernel_snapshot.get("candidates") or ()
    for candidate in candidates:
        candidate_data = _as_mapping(candidate)
        if _clean_label(candidate_data.get("intervention")) == intervention:
            return candidate_data
    return _as_mapping(candidates[0]) if candidates else {}


def _hypotheses(candidate: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(_as_mapping(item) for item in (candidate.get("hypotheses") or ()))


def _primary_pattern(hypotheses: tuple[Mapping[str, Any], ...]) -> str:
    priority = (
        "loop",
        "goal_direction",
        "act",
        "focus",
        "dbt",
        "mbsr",
        "rebt",
        "cbt",
        "emotion",
        "policy",
    )
    for source in priority:
        for item in hypotheses:
            if _clean_label(item.get("source")) == source:
                pattern = _clean_label(item.get("pattern"))
                if pattern:
                    return pattern
    for item in hypotheses:
        pattern = _clean_label(item.get("pattern"))
        if pattern:
            return pattern
    return "current_pattern"


def _focus_label(hypotheses: tuple[Mapping[str, Any], ...], pattern: str) -> str:
    for item in hypotheses:
        if _clean_label(item.get("source")) == "focus":
            focus = _clean_label(item.get("pattern"))
            if focus:
                return _humanize(focus)
    return _pattern_text(pattern)


def _template_for(intervention: str, pattern: str) -> dict[str, str]:
    if intervention in {"validation", "gentle_check_in", "needs_exploration"}:
        return _TEMPLATES["gentle_observation"]
    if intervention in {
        "acceptance_committed_action",
        "committed_action",
        "goal_action_planning",
        "willingness_persistence_plan",
        "values_aligned_next_step",
        "willingness_practice",
        "define_next_action",
        "implementation_intention",
        "reduce_scope",
    }:
        return _TEMPLATES["valued_action"]
    if intervention in {
        "clarify_objective",
        "define_success_measure",
        "woop_obstacle_plan",
    }:
        return _TEMPLATES["goal_direction_test"]
    if intervention in {
        "capture_open_loops",
        "limit_work_in_progress",
        "waiting_for_review",
    }:
        return _TEMPLATES["execution_hygiene_test"]
    if intervention in {
        "avoidance_map",
        "behavioral_experiment",
        "exposure_exercise",
    }:
        return _TEMPLATES["avoidance_test"]
    if intervention in {
        "cognitive_defusion",
        "values_clarification",
        "values_direction_review",
    }:
        return _TEMPLATES["defusion_values"]
    if intervention in {
        "evidence_check",
        "decatastrophizing",
        "socratic_question",
        "continuum_technique",
        "cognitive_pattern_review",
        "cognitive_reframe",
        "decision_clarity_map",
        "balanced_evidence_review",
        "emotion_fact_separation",
        "responsibility_pie",
        "wise_mind_check",
        "dialectical_reframe",
    }:
        return _TEMPLATES["thought_test"]
    if intervention in {
        "rebt_disputation",
        "preference_rewrite",
        "unconditional_self_acceptance",
        "mastery_evidence_log",
        "self_compassion",
        "self_validation",
        "strength_evidence_log",
        "approval_preference_rewrite",
        "failure_tolerance_reframe",
        "self_as_context",
    }:
        return _TEMPLATES["self_worth_test"]
    if intervention in {
        "present_moment_grounding",
        "distress_tolerance_pause",
        "self_soothing",
        "emotion_regulation_check_facts",
        "opposite_action_planning",
        "emotion_regulation_plan",
        "mindfulness_observe_describe",
        "acceptance_practice",
        "creative_hopelessness",
        "frustration_tolerance_practice",
        "uncertainty_tolerance_practice",
        "vulnerability_reduction_plan",
        "mindful_breathing_space",
        "stress_response_labeling",
        "body_scan_check_in",
        "mindful_pause",
        "mindful_observe_return",
        "mindful_self_care_check",
    }:
        return _TEMPLATES["regulation_test"]
    if intervention in {
        "interpersonal_effectiveness_script",
        "boundary_effectiveness_plan",
    }:
        return _TEMPLATES["interpersonal_test"]
    if intervention == "environment_design_plan":
        return _TEMPLATES["environment_test"]
    if intervention in {"setback_recovery_plan", "learning_review"}:
        return _TEMPLATES["review_test"]
    if intervention == "weekly_review":
        return _TEMPLATES["review_test"]
    if pattern in {"minimal_disclosure", "sadness", "anxiety"} or "minimal_disclosure" in pattern:
        return _TEMPLATES["gentle_observation"]
    return _TEMPLATES["generic"]


_TEMPLATES: dict[str, dict[str, str]] = {
    "valued_action": {
        "title": "Tiny values-aligned rep",
        "hypothesis": "This tests whether {pattern} loosens when the target is one small rep rather than proving something global.",
        "action": "Do the smallest visible step toward the valued action, while allowing discomfort to come along.",
        "prediction": "If avoidance is being maintained by discomfort or identity pressure, a tiny rep should create at least a little movement or clarity.",
        "measure": "Rate urge-to-avoid before and after from 0-10, then note what changed.",
        "timebox": "10 minutes",
    },
    "goal_direction_test": {
        "title": "Direction-to-action check",
        "hypothesis": "This tests whether {pattern} improves when direction, obstacle, and progress evidence are made explicit.",
        "action": "Write the objective, the next visible action, the likely obstacle, and one observable sign of progress.",
        "prediction": "The goal should feel less vague and the next move should become easier to start.",
        "measure": "Rate clarity about what to do next from 0-10 before and after.",
        "timebox": "8 minutes",
    },
    "execution_hygiene_test": {
        "title": "Open-loop cleanup",
        "hypothesis": "This tests whether {pattern} is being amplified by too many open loops or active commitments.",
        "action": "Capture the open loops, choose at most three active items, and mark the rest as later, waiting, or parked.",
        "prediction": "Reducing active commitments should lower mental clutter and reveal the next action.",
        "measure": "Rate overload and next-action clarity from 0-10 before and after.",
        "timebox": "10 minutes",
    },
    "avoidance_test": {
        "title": "Avoidance cost check",
        "hypothesis": "This tests whether {pattern} is giving short-term relief while keeping the larger problem stuck.",
        "action": "Map the trigger, the feeling being avoided, the relief from avoiding, and one very small approach step.",
        "prediction": "If avoidance is the loop, naming the cost and trying a small approach step should make the next move clearer.",
        "measure": "Track relief, discomfort, and willingness from 0-10 before and after the approach step.",
        "timebox": "8 minutes",
    },
    "defusion_values": {
        "title": "Thought-distance rep",
        "hypothesis": "This tests whether {pattern} softens when the thought is held as a thought rather than treated as an instruction.",
        "action": "Name the thought with 'I am noticing the thought that...', then choose one value-consistent micro-action.",
        "prediction": "The thought may still be there, but it should have slightly less control over the next action.",
        "measure": "Rate thought-believability and willingness from 0-10 before and after.",
        "timebox": "5 minutes",
    },
    "thought_test": {
        "title": "Belief update test",
        "hypothesis": "This tests whether {pattern} is an interpretation that can be updated with facts.",
        "action": "Write one observed fact, one mind-made prediction, and one small piece of evidence that would update the prediction either way.",
        "prediction": "Separating facts from predictions should reduce certainty or reveal a concrete next check.",
        "measure": "Rate certainty in the original thought from 0-10 before and after.",
        "timebox": "7 minutes",
    },
    "self_worth_test": {
        "title": "Worth versus performance split",
        "hypothesis": "This tests whether {pattern} is turning a performance concern into a global verdict about you.",
        "action": "Write two columns: what happened or might happen, and what it does not prove about your whole worth or capacity.",
        "prediction": "The concern may remain, but the global self-verdict should feel a little less fused.",
        "measure": "Rate shame or self-attack from 0-10 before and after.",
        "timebox": "6 minutes",
    },
    "regulation_test": {
        "title": "Regulate before solving",
        "hypothesis": "This tests whether {pattern} needs a nervous-system pause before problem solving is useful.",
        "action": "Pause, slow the exhale, feel your feet, name the emotion and urge, then choose only the next non-harmful minute.",
        "prediction": "The problem may not be solved, but intensity or urgency should drop enough to choose more deliberately.",
        "measure": "Rate emotional intensity and action urge from 0-10 before and after.",
        "timebox": "2 minutes",
    },
    "interpersonal_test": {
        "title": "Direct ask rehearsal",
        "hypothesis": "This tests whether {pattern} becomes more workable when the ask and boundary are made explicit.",
        "action": "Draft one sentence for the situation, one clear ask or boundary, and one self-respect line.",
        "prediction": "Making the ask concrete should reduce ambiguity or reveal the real negotiation point.",
        "measure": "Rate clarity and willingness from 0-10 before and after drafting.",
        "timebox": "10 minutes",
    },
    "environment_test": {
        "title": "Friction redesign",
        "hypothesis": "This tests whether changing cues and friction changes follow-through more than trying to force motivation.",
        "action": "Remove one friction point or add one cue that makes the desired action easier to start.",
        "prediction": "A small environment change should make the next action easier or more automatic.",
        "measure": "Track whether the action started and rate start-up friction from 0-10.",
        "timebox": "10 minutes",
    },
    "review_test": {
        "title": "Learning-loop review",
        "hypothesis": "This tests whether {pattern} improves when the last attempt is treated as data rather than a verdict.",
        "action": "Write what happened, what helped, what broke down, and the smallest adjustment for the next attempt.",
        "prediction": "Reviewing the attempt as data should make recommitment feel more specific and less loaded.",
        "measure": "Rate clarity about the next adjustment from 0-10 before and after.",
        "timebox": "8 minutes",
    },
    "gentle_observation": {
        "title": "Gentle feeling-name exercise",
        "hypothesis": "This tests whether naming {emotion} gently gives you a little more contact with what is happening, without trying to fix it.",
        "action": "Say out loud or write: 'I am noticing {emotion} right now.' Then take one slow breath and notice where it lands in your body. Stop after that.",
        "prediction": "The feeling may not change much, but it should become a little clearer or less vague.",
        "measure": "Rate emotional clarity from 0-10 before and after the one-minute check.",
        "timebox": "1 minute",
    },
    "generic": {
        "title": "One-turn coaching test",
        "hypothesis": "This tests whether the selected intervention helps with {pattern} in this specific situation.",
        "action": "Try the selected exercise once, keeping it small enough to complete today.",
        "prediction": "The move should create a little more clarity, steadiness, or action-readiness.",
        "measure": "Rate usefulness from 0-10 and write one sentence about what happened.",
        "timebox": "10 minutes",
    },
}


def _action_for(
    intervention: str,
    template_action: str,
    plan_exercise: str | None,
    plan_question: str | None,
) -> str:
    if not plan_exercise:
        if plan_question and intervention not in {"validation", "gentle_check_in", "needs_exploration"}:
            return f"Answer this prompt in one or two sentences: {plan_question}"
        return template_action
    if intervention in {"validation", "gentle_check_in", "needs_exploration"}:
        return template_action
    if len(plan_exercise) <= 240:
        return plan_exercise
    return template_action


def _support_nodes(
    *,
    formulation_delta: FormulationDelta,
    intervention: str,
    hypotheses: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    labels = {
        intervention,
        *(
            f"{_clean_label(item.get('source'))}: {_clean_label(item.get('pattern'))}"
            for item in hypotheses
            if _clean_label(item.get("source")) and _clean_label(item.get("pattern"))
        ),
    }
    node_ids = []
    for node in (*formulation_delta.added_nodes, *formulation_delta.updated_nodes):
        if node.label in labels or node.kind in {
            "concern",
            "task",
            "challenge",
            "objective",
            "project",
            "next_action",
            "obstacle",
            "key_result",
            "success_measure",
            "time_horizon",
            "waiting_for",
            "stake",
            "goal",
            "behavior",
            "emotion",
        }:
            node_ids.append(node.id)
        if len(node_ids) >= 8:
            break
    return tuple(dict.fromkeys(node_ids))


def _rationale(intervention: str, hypotheses: tuple[Mapping[str, Any], ...]) -> str:
    patterns = [
        _pattern_text(_clean_label(item.get("pattern")))
        for item in hypotheses
        if _clean_label(item.get("pattern"))
    ]
    if not patterns:
        return f"Proposed because the selected move was {_humanize(intervention)}."
    return (
        f"Proposed because {_humanize(intervention)} was supported by "
        f"{_join_unique(patterns)}."
    )


def _learning_for_feedback(
    experiment: CoachingExperiment,
    action: ExperimentFeedbackAction,
) -> str:
    if action == "helped":
        return "Mark this as useful evidence and consider repeating the same tiny version once more."
    if action == "did_not_help":
        return "Treat that as data: the next experiment should change the action, dose, or hypothesis."
    if action == "too_hard":
        return "The useful learning is intensity: shrink the next experiment until it feels almost too small."
    if action == "skipped":
        return "Skipping is still data: the next step is to look at what made starting costly or unclear."
    if action == "neutral":
        return "Neutral data suggests keeping the observation but trying a sharper measure or smaller target."
    return "Record what happened and use it to update the next tiny experiment."


def _format_template(
    template: str,
    *,
    pattern: str,
    emotion: str | None = None,
) -> str:
    return template.format(
        pattern=_pattern_text(pattern),
        emotion=emotion or "the feeling",
    )


def _primary_emotion(
    extraction: Any,
    hypotheses: tuple[Mapping[str, Any], ...],
) -> str:
    extraction_data = _as_mapping(extraction)
    for emotion in _iter_strings(extraction_data.get("emotions")):
        return _humanize(emotion)
    for item in hypotheses:
        if _clean_label(item.get("source")) == "emotion":
            emotion = _clean_label(item.get("pattern"))
            if emotion:
                return _humanize(emotion)
    return "the feeling"


def _iter_strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    try:
        return tuple(str(value).strip() for value in values if str(value).strip())
    except TypeError:
        return ()


def _pattern_text(pattern: str) -> str:
    return _PATTERN_LABELS.get(pattern, _humanize(pattern))


_PATTERN_LABELS = {
    "minimal_disclosure_sad_anxious": "sadness or anxiety with little context",
    "minimal_disclosure": "a feeling with little context",
    "high_distress_gate": "high distress",
}


def _experiment_id(
    turn: int,
    message_index: int | None,
    intervention: str,
    pattern: str,
    title: str,
) -> str:
    seed = f"{turn}:{message_index}:{intervention}:{pattern}:{title}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"exp-{turn}-{digest}"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _clean_label(value: Any) -> str:
    return str(value or "").strip()


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip()


def _join_unique(values: list[str]) -> str:
    unique = []
    seen = set()
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    if not unique:
        return "the current pattern"
    if len(unique) == 1:
        return unique[0]
    return f"{', '.join(unique[:-1])}, and {unique[-1]}"
