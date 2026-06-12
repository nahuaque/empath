"""DBT-specific relational ontology and skill-target inference."""

from __future__ import annotations

from typing import Any

from kanren import conde, eq, facts, var

from .base import TherapeuticSystem


class DBTSystem(TherapeuticSystem):
    """DBT skill-module hypotheses and interventions."""

    source = "dbt"

    def load_ontology(self, kernel: Any) -> None:
        facts(
            kernel.pattern_intervention,
            ("crisis_survival", "distress_tolerance_pause"),
            ("crisis_survival", "self_soothing"),
            ("emotion_dysregulation", "emotion_regulation_check_facts"),
            ("emotion_dysregulation", "opposite_action_planning"),
            ("interpersonal_effectiveness_need", "interpersonal_effectiveness_script"),
            ("mindfulness_need", "present_moment_grounding"),
            ("mindfulness_need", "mindfulness_observe_describe"),
            ("self_invalidation", "self_validation"),
            ("wise_mind_need", "wise_mind_check"),
            ("vulnerability_factors", "vulnerability_reduction_plan"),
            ("vulnerability_factors", "self_soothing"),
            ("dialectical_stuckness", "dialectical_reframe"),
        )
        facts(
            kernel.intervention_modality,
            ("distress_tolerance_pause", "dbt"),
            ("self_soothing", "dbt"),
            ("emotion_regulation_check_facts", "dbt"),
            ("opposite_action_planning", "dbt"),
            ("interpersonal_effectiveness_script", "dbt"),
            ("present_moment_grounding", "dbt"),
            ("mindfulness_observe_describe", "dbt"),
            ("self_validation", "dbt"),
            ("wise_mind_check", "dbt"),
            ("vulnerability_reduction_plan", "dbt"),
            ("dialectical_reframe", "dbt"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "distress_tolerance_pause",
                "Pause for 60 seconds, name the urge, slow the exhale, and choose the next non-harmful minute.",
            ),
            (
                "self_soothing",
                "Pick one calming sensory input: temperature, sound, texture, scent, or visual focus.",
            ),
            (
                "emotion_regulation_check_facts",
                "Name the emotion, prompting event, interpretation, body signal, and action urge.",
            ),
            (
                "opposite_action_planning",
                "If the emotion fits poorly or is too intense, choose one small action opposite to the urge.",
            ),
            (
                "interpersonal_effectiveness_script",
                "Draft a direct ask or boundary: describe the situation, state the feeling, ask clearly.",
            ),
            (
                "mindfulness_observe_describe",
                "Observe and describe the present emotion as sensations, thoughts, and urges without judgment.",
            ),
            (
                "self_validation",
                "Reflect why the feeling makes sense before trying to change anything.",
            ),
            (
                "wise_mind_check",
                "Write the emotion-mind signal, the reasonable-mind facts, and one wise-mind next step.",
            ),
            (
                "vulnerability_reduction_plan",
                "Check sleep, food, movement, substances, illness, and overload; choose one vulnerability factor to reduce first.",
            ),
            (
                "dialectical_reframe",
                "Write both sides of the tension, then add an 'and' statement that preserves what is true in each.",
            ),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        return conde(
            [
                self._crisis_survivalo(kernel, state),
                eq(pattern, "crisis_survival"),
            ],
            [
                self._emotion_regulationo(kernel, state),
                eq(pattern, "emotion_dysregulation"),
            ],
            [
                self._interpersonal_effectivenesso(kernel, state),
                eq(pattern, "interpersonal_effectiveness_need"),
            ],
            [
                self._mindfulnesso(kernel, state),
                eq(pattern, "mindfulness_need"),
            ],
            [
                self._self_invalidationo(kernel, state),
                eq(pattern, "self_invalidation"),
            ],
            [
                self._wise_mindo(kernel, state),
                eq(pattern, "wise_mind_need"),
            ],
            [
                kernel.state_feature(state, "vulnerability_factors"),
                eq(pattern, "vulnerability_factors"),
            ],
            [
                self._dialectical_stucknesso(kernel, state),
                eq(pattern, "dialectical_stuckness"),
            ],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        if "crisis_survival" in patterns:
            if intervention == "distress_tolerance_pause":
                score += 1.75
            if intervention == "self_soothing":
                score += 1.0
        if "emotion_dysregulation" in patterns:
            if intervention == "emotion_regulation_check_facts":
                score += 1.75
            if intervention == "opposite_action_planning":
                score += 1.25
        if (
            "interpersonal_effectiveness_need" in patterns
            and intervention == "interpersonal_effectiveness_script"
        ):
            score += 1.25
        if (
            "mindfulness_need" in patterns
            and intervention == "mindfulness_observe_describe"
        ):
            score += 0.5
        if "self_invalidation" in patterns and intervention == "self_validation":
            score += 1.5
        if "wise_mind_need" in patterns and intervention == "wise_mind_check":
            score += 1.5
        if "vulnerability_factors" in patterns:
            if intervention == "vulnerability_reduction_plan":
                score += 1.5
            if intervention == "self_soothing":
                score += 0.5
        if (
            "dialectical_stuckness" in patterns
            and intervention == "dialectical_reframe"
        ):
            score += 1.0
        return score

    def _crisis_survivalo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "high_distress")],
            [kernel.state_feature(state, "crisis_urge")],
            [kernel.state_feature(state, "safety_risk")],
        )

    def _emotion_regulationo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "emotion_dysregulation")],
            [kernel.has_emotion(state, "overwhelm")],
            [
                kernel.activation_level(state, "medium"),
                self._has_strong_emotiono(kernel, state),
            ],
        )

    def _interpersonal_effectivenesso(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "interpersonal_conflict")],
            [kernel.state_feature(state, "boundary_difficulty")],
        )

    def _mindfulnesso(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "present_moment_disconnection")],
            [kernel.has_behavior(state, "rumination")],
        )

    def _self_invalidationo(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [kernel.state_feature(state, "self_invalidation")],
            [
                kernel.has_emotion(state, "shame"),
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "global_label"),
            ],
            [
                kernel.has_emotion(state, "shame"),
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "global_label"),
            ],
        )

    def _wise_mindo(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [kernel.state_feature(state, "decision_uncertainty")],
            [kernel.state_feature(state, "certainty_demand")],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "certainty_demand_claim"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "certainty_demand_claim"),
            ],
        )

    def _dialectical_stucknesso(self, kernel: Any, state: Any):
        thought = var()
        return conde(
            [
                kernel.state_feature(state, "decision_uncertainty"),
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "binary_evaluation"),
            ],
            [
                kernel.state_feature(state, "interpersonal_conflict"),
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "binary_evaluation"),
            ],
        )

    def _has_strong_emotiono(self, kernel: Any, state: Any):
        return conde(
            [kernel.has_emotion(state, "anxiety")],
            [kernel.has_emotion(state, "anger")],
            [kernel.has_emotion(state, "shame")],
            [kernel.has_emotion(state, "overwhelm")],
        )
