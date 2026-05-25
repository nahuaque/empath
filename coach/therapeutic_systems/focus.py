"""Cross-cutting coaching focus areas across therapeutic systems."""

from __future__ import annotations

from typing import Any

from kanren import conde, eq, facts, var

from .base import TherapeuticSystem


class CoachingFocusSystem(TherapeuticSystem):
    """Structured coaching focus hypotheses across ACT/CBT/REBT/DBT lenses."""

    source = "focus"

    def load_ontology(self, kernel: Any) -> None:
        facts(
            kernel.pattern_intervention,
            ("values_direction", "values_direction_review"),
            ("goal_activation", "goal_action_planning"),
            ("motivation_persistence", "willingness_persistence_plan"),
            ("cognitive_belief_work", "cognitive_pattern_review"),
            ("emotion_distress_regulation", "emotion_regulation_plan"),
            ("avoidance_escape", "avoidance_map"),
            ("self_efficacy", "mastery_evidence_log"),
            ("decision_problem_solving", "decision_clarity_map"),
            ("interpersonal_boundaries", "boundary_effectiveness_plan"),
            ("attention_environment_design", "environment_design_plan"),
            ("resilience_recovery", "setback_recovery_plan"),
            ("integration_review", "learning_review"),
        )
        facts(
            kernel.intervention_modality,
            ("values_direction_review", "coaching"),
            ("goal_action_planning", "coaching"),
            ("willingness_persistence_plan", "coaching"),
            ("cognitive_pattern_review", "coaching"),
            ("emotion_regulation_plan", "coaching"),
            ("avoidance_map", "coaching"),
            ("mastery_evidence_log", "coaching"),
            ("decision_clarity_map", "coaching"),
            ("boundary_effectiveness_plan", "coaching"),
            ("environment_design_plan", "coaching"),
            ("setback_recovery_plan", "coaching"),
            ("learning_review", "coaching"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "values_direction_review",
                "Name the value, the direction it points toward, and one behavior that would express it.",
            ),
            (
                "goal_action_planning",
                "Convert the desired outcome into one observable next action, one milestone, and one check-in.",
            ),
            (
                "willingness_persistence_plan",
                "Identify the discomfort that may show up and the action worth taking while it is present.",
            ),
            (
                "cognitive_pattern_review",
                "Separate situation, interpretation, belief, emotion, and action in one compact map.",
            ),
            (
                "emotion_regulation_plan",
                "Name the feeling, intensity, body cue, urge, and one skillful response.",
            ),
            (
                "avoidance_map",
                "Map the avoided trigger, feared experience, short-term relief, and long-term cost.",
            ),
            (
                "mastery_evidence_log",
                "List one piece of evidence for learning, coping, or recovering after difficulty.",
            ),
            (
                "decision_clarity_map",
                "Lay out options, values served, evidence needed, costs, and the reversible next step.",
            ),
            (
                "boundary_effectiveness_plan",
                "Draft the ask, the boundary, the relationship value, and the self-respect line.",
            ),
            (
                "environment_design_plan",
                "Choose one cue, friction point, routine, or workspace change that makes action easier.",
            ),
            (
                "setback_recovery_plan",
                "Review the trigger, what worked, what broke down, and the smallest recommitment.",
            ),
            (
                "learning_review",
                "Review the repeating pattern, what has been tried, what helped, and what to adjust.",
            ),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        return conde(
            [self._values_directiono(kernel, state), eq(pattern, "values_direction")],
            [self._goal_activationo(kernel, state), eq(pattern, "goal_activation")],
            [
                self._motivation_persistenceo(kernel, state),
                eq(pattern, "motivation_persistence"),
            ],
            [
                self._cognitive_belief_worko(kernel, state),
                eq(pattern, "cognitive_belief_work"),
            ],
            [
                self._emotion_distress_regulationo(kernel, state),
                eq(pattern, "emotion_distress_regulation"),
            ],
            [self._avoidance_escapeo(kernel, state), eq(pattern, "avoidance_escape")],
            [self._self_efficacyo(kernel, state), eq(pattern, "self_efficacy")],
            [
                self._decision_problem_solvingo(kernel, state),
                eq(pattern, "decision_problem_solving"),
            ],
            [
                self._interpersonal_boundarieso(kernel, state),
                eq(pattern, "interpersonal_boundaries"),
            ],
            [
                self._attention_environment_designo(kernel, state),
                eq(pattern, "attention_environment_design"),
            ],
            [self._resilience_recoveryo(kernel, state), eq(pattern, "resilience_recovery")],
            [self._integration_reviewo(kernel, state), eq(pattern, "integration_review")],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        focus_interventions = {
            "values_direction_review",
            "goal_action_planning",
            "willingness_persistence_plan",
            "cognitive_pattern_review",
            "emotion_regulation_plan",
            "avoidance_map",
            "mastery_evidence_log",
            "decision_clarity_map",
            "boundary_effectiveness_plan",
            "environment_design_plan",
            "setback_recovery_plan",
            "learning_review",
        }
        if intervention in focus_interventions:
            return -0.75
        return 0.0

    def _values_directiono(self, kernel: Any, state: Any):
        item = var()
        return conde(
            [kernel.has_value(state, item)],
            [kernel.state_feature(state, "values_unclear")],
        )

    def _goal_activationo(self, kernel: Any, state: Any):
        item = var()
        return conde(
            [kernel.has_goal(state, item)],
            [kernel.state_feature(state, "goal_setting")],
            [kernel.has_behavior(state, "inaction")],
        )

    def _motivation_persistenceo(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [kernel.state_feature(state, "motivation_block")],
            [kernel.state_feature(state, "control_struggle")],
            [kernel.has_behavior(state, "procrastination")],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "unbearable_claim"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "unbearable_claim"),
            ],
        )

    def _cognitive_belief_worko(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        feature = var()
        return conde(
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, feature),
                self._cognitive_featureo(feature),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, feature),
                self._cognitive_featureo(feature),
            ],
        )

    def _cognitive_featureo(self, feature: Any):
        return conde(
            [eq(feature, "future_disaster")],
            [eq(feature, "awful_outcome")],
            [eq(feature, "mind_reading_claim")],
            [eq(feature, "binary_evaluation")],
            [eq(feature, "single_event_global_conclusion")],
            [eq(feature, "global_label")],
            [eq(feature, "demanding_rule")],
            [eq(feature, "unbearable_claim")],
            [eq(feature, "identity_global_rating")],
            [eq(feature, "identity_fusion")],
            [eq(feature, "sticky_thought")],
            [eq(feature, "discounting_positive_claim")],
            [eq(feature, "feeling_as_fact")],
            [eq(feature, "personal_responsibility_claim")],
            [eq(feature, "negative_filter")],
            [eq(feature, "approval_demand")],
            [eq(feature, "certainty_demand_claim")],
            [eq(feature, "failure_intolerance_claim")],
            [eq(feature, "control_struggle")],
        )

    def _emotion_distress_regulationo(self, kernel: Any, state: Any):
        emotion = var()
        return conde(
            [kernel.has_emotion(state, emotion)],
            [kernel.state_feature(state, "emotion_dysregulation")],
            [kernel.state_feature(state, "high_distress")],
            [kernel.state_feature(state, "control_struggle")],
            [kernel.state_feature(state, "vulnerability_factors")],
        )

    def _avoidance_escapeo(self, kernel: Any, state: Any):
        return conde(
            [kernel.has_behavior(state, "avoidance")],
            [kernel.has_behavior(state, "procrastination")],
            [kernel.has_behavior(state, "withdrawal")],
        )

    def _self_efficacyo(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [kernel.state_feature(state, "self_efficacy_doubt")],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "identity_global_rating"),
            ],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "global_label"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "identity_global_rating"),
            ],
        )

    def _decision_problem_solvingo(self, kernel: Any, state: Any):
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

    def _interpersonal_boundarieso(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "interpersonal_conflict")],
            [kernel.state_feature(state, "boundary_difficulty")],
            [kernel.state_feature(state, "approval_threat")],
        )

    def _attention_environment_designo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "attention_environment")],
            [kernel.state_feature(state, "goal_setting")],
        )

    def _resilience_recoveryo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "setback_recovery")],
            [kernel.state_feature(state, "safety_risk")],
        )

    def _integration_reviewo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "integration_review")],
            [kernel.state_feature(state, "recurring_pattern")],
        )
