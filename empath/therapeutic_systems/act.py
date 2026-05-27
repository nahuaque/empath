"""ACT-specific relational ontology and pattern inference."""

from __future__ import annotations

from typing import Any

from kanren import conde, eq, facts, var

from .base import TherapeuticSystem


class ACTSystem(TherapeuticSystem):
    """ACT stuck-process hypotheses and interventions."""

    source = "act"

    def load_ontology(self, kernel: Any) -> None:
        facts(
            kernel.pattern_intervention,
            ("fusion", "cognitive_defusion"),
            ("experiential_avoidance", "acceptance_committed_action"),
            ("experiential_avoidance", "committed_action"),
            ("values_unclear", "values_clarification"),
            ("inaction", "committed_action"),
            ("rumination", "present_moment_grounding"),
            ("present_moment_disconnection", "present_moment_grounding"),
            ("self_as_content", "self_as_context"),
            ("unworkable_control", "acceptance_practice"),
            ("unworkable_control", "creative_hopelessness"),
            ("unwillingness", "willingness_practice"),
            ("values_action_gap", "values_aligned_next_step"),
            ("values_action_gap", "committed_action"),
        )
        facts(
            kernel.intervention_modality,
            ("present_moment_grounding", "act"),
            ("cognitive_defusion", "act"),
            ("values_clarification", "act"),
            ("committed_action", "act"),
            ("acceptance_committed_action", "act"),
            ("acceptance_practice", "act"),
            ("self_as_context", "act"),
            ("creative_hopelessness", "act"),
            ("willingness_practice", "act"),
            ("values_aligned_next_step", "act"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "cognitive_defusion",
                "Name it as: I am noticing the thought that ...",
            ),
            (
                "acceptance_committed_action",
                "Choose a 10-20 minute valued action while making room for discomfort.",
            ),
            (
                "committed_action",
                "Define the next visible action small enough to start today.",
            ),
            (
                "values_clarification",
                "Ask what quality of action the user wants to stand for here.",
            ),
            (
                "present_moment_grounding",
                "Orient to breath, feet, and five things currently visible.",
            ),
            (
                "acceptance_practice",
                "Make room for one difficult sensation or feeling for 30 seconds without trying to solve it.",
            ),
            (
                "self_as_context",
                "Notice the role, label, or self-story, then name the observing part of you that can see it.",
            ),
            (
                "creative_hopelessness",
                "Briefly map what trying to control or eliminate the feeling has cost, and whether it has worked.",
            ),
            (
                "willingness_practice",
                "Name the discomfort you are willing to carry for a few minutes in service of one valued action.",
            ),
            (
                "values_aligned_next_step",
                "Pick one next step that expresses the value at 1 percent intensity rather than waiting for motivation.",
            ),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        thought = var()
        belief = var()
        return conde(
            [
                kernel.has_behavior(state, "avoidance"),
                eq(pattern, "experiential_avoidance"),
            ],
            [
                kernel.has_behavior(state, "procrastination"),
                eq(pattern, "experiential_avoidance"),
            ],
            [
                kernel.has_behavior(state, "withdrawal"),
                eq(pattern, "experiential_avoidance"),
            ],
            [kernel.has_behavior(state, "rumination"), eq(pattern, "rumination")],
            [kernel.has_behavior(state, "inaction"), eq(pattern, "inaction")],
            [kernel.state_feature(state, "values_unclear"), eq(pattern, "values_unclear")],
            [
                kernel.state_feature(state, "present_moment_disconnection"),
                eq(pattern, "present_moment_disconnection"),
            ],
            [
                kernel.state_feature(state, "control_struggle"),
                eq(pattern, "unworkable_control"),
            ],
            [
                kernel.state_feature(state, "motivation_block"),
                eq(pattern, "unwillingness"),
            ],
            [
                self._values_action_gapo(kernel, state),
                eq(pattern, "values_action_gap"),
            ],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "identity_fusion"),
                eq(pattern, "fusion"),
            ],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "sticky_thought"),
                eq(pattern, "fusion"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "identity_global_rating"),
                eq(pattern, "fusion"),
            ],
            [
                self._self_as_contento(kernel, state),
                eq(pattern, "self_as_content"),
            ],
            [
                self._text_control_struggleo(kernel, state),
                eq(pattern, "unworkable_control"),
            ],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        if "fusion" in patterns and intervention == "cognitive_defusion":
            score += 1.5
        if (
            "experiential_avoidance" in patterns
            and intervention == "acceptance_committed_action"
        ):
            score += 1.5
        if "values_unclear" in patterns and intervention == "values_clarification":
            score += 1.0
        if "self_as_content" in patterns and intervention == "self_as_context":
            score += 1.25
        if "unworkable_control" in patterns and intervention == "acceptance_practice":
            score += 1.25
        if "unworkable_control" in patterns and intervention == "creative_hopelessness":
            score += 0.75
        if "unwillingness" in patterns and intervention == "willingness_practice":
            score += 1.25
        if "values_action_gap" in patterns and intervention == "values_aligned_next_step":
            score += 1.5
        return score

    def _self_as_contento(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "identity_fusion"),
            ],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "identity_global_rating"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "identity_fusion"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "identity_global_rating"),
            ],
        )

    def _text_control_struggleo(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "control_struggle"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "control_struggle"),
            ],
        )

    def _values_action_gapo(self, kernel: Any, state: Any):
        item = var()
        return conde(
            [kernel.has_value(state, item), kernel.has_behavior(state, "avoidance")],
            [kernel.has_value(state, item), kernel.has_behavior(state, "procrastination")],
            [kernel.has_value(state, item), kernel.has_behavior(state, "inaction")],
            [kernel.has_goal(state, item), kernel.has_behavior(state, "avoidance")],
            [kernel.has_goal(state, item), kernel.has_behavior(state, "procrastination")],
            [kernel.has_goal(state, item), kernel.has_behavior(state, "inaction")],
        )
