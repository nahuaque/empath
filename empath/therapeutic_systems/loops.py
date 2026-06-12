"""Cross-system therapeutic loops built from lower-level observations."""

from __future__ import annotations

from typing import Any

from kanren import conde, eq, facts, var

from .base import TherapeuticSystem


class LoopSystem(TherapeuticSystem):
    """Named multi-signal loops that coordinate ACT/CBT/REBT hypotheses."""

    source = "loop"

    def load_ontology(self, kernel: Any) -> None:
        facts(
            kernel.pattern_intervention,
            (
                "avoidance_identity_threat",
                "acceptance_committed_action",
            ),
            ("avoidance_identity_threat", "cognitive_defusion"),
            ("minimal_disclosure_sad_anxious", "validation"),
            ("minimal_disclosure_sad_anxious", "gentle_check_in"),
            ("shame_self_worth_fusion", "self_compassion"),
            ("shame_self_worth_fusion", "unconditional_self_acceptance"),
            ("shame_self_worth_fusion", "cognitive_defusion"),
            ("valued_action_procrastination", "acceptance_committed_action"),
            ("valued_action_procrastination", "committed_action"),
            ("high_distress_gating", "validation"),
            ("high_distress_gating", "present_moment_grounding"),
            ("certainty_avoidance_loop", "wise_mind_check"),
            ("certainty_avoidance_loop", "decision_clarity_map"),
            ("certainty_avoidance_loop", "committed_action"),
            ("approval_threat_loop", "self_validation"),
            ("approval_threat_loop", "approval_preference_rewrite"),
            ("approval_threat_loop", "evidence_check"),
            ("control_struggle_loop", "acceptance_practice"),
            ("control_struggle_loop", "willingness_practice"),
            ("control_struggle_loop", "present_moment_grounding"),
            ("vulnerability_distress_loop", "vulnerability_reduction_plan"),
            ("vulnerability_distress_loop", "self_soothing"),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        return conde(
            [
                self._avoidanceo(kernel, state),
                self._identity_threato(kernel, state),
                eq(pattern, "avoidance_identity_threat"),
            ],
            [
                kernel.state_feature(state, "minimal_disclosure"),
                self._sadness_or_anxietyo(kernel, state),
                eq(pattern, "minimal_disclosure_sad_anxious"),
            ],
            [
                kernel.has_emotion(state, "shame"),
                self._identity_threato(kernel, state),
                eq(pattern, "shame_self_worth_fusion"),
            ],
            [
                kernel.has_behavior(state, "procrastination"),
                self._has_concrete_value_or_goalo(kernel, state),
                eq(pattern, "valued_action_procrastination"),
            ],
            [
                kernel.state_feature(state, "high_distress"),
                eq(pattern, "high_distress_gating"),
            ],
            [
                self._avoidanceo(kernel, state),
                self._certainty_needo(kernel, state),
                eq(pattern, "certainty_avoidance_loop"),
            ],
            [
                self._approval_threato(kernel, state),
                eq(pattern, "approval_threat_loop"),
            ],
            [
                self._control_struggleo(kernel, state),
                eq(pattern, "control_struggle_loop"),
            ],
            [
                kernel.state_feature(state, "vulnerability_factors"),
                self._distress_or_dysregulationo(kernel, state),
                eq(pattern, "vulnerability_distress_loop"),
            ],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        if "avoidance_identity_threat" in patterns:
            if intervention in {"acceptance_committed_action", "cognitive_defusion"}:
                score += 1.25
        if "minimal_disclosure_sad_anxious" in patterns:
            if intervention == "validation":
                score += 1.25
            if intervention == "gentle_check_in":
                score += 1.0
        if "shame_self_worth_fusion" in patterns:
            if intervention in {"self_compassion", "unconditional_self_acceptance"}:
                score += 1.25
            if intervention == "cognitive_defusion":
                score += 0.75
        if "valued_action_procrastination" in patterns:
            if intervention == "acceptance_committed_action":
                score += 1.5
            if intervention == "committed_action":
                score += 1.0
        if "high_distress_gating" in patterns:
            if intervention == "validation":
                score += 2.0
            if intervention == "present_moment_grounding":
                score += 1.75
        if "certainty_avoidance_loop" in patterns:
            if intervention == "wise_mind_check":
                score += 1.5
            if intervention == "decision_clarity_map":
                score += 1.25
            if intervention == "committed_action":
                score += 0.75
        if "approval_threat_loop" in patterns:
            if intervention in {"self_validation", "approval_preference_rewrite"}:
                score += 1.25
            if intervention == "evidence_check":
                score += 0.75
        if "control_struggle_loop" in patterns:
            if intervention == "acceptance_practice":
                score += 1.5
            if intervention == "willingness_practice":
                score += 1.25
            if intervention == "present_moment_grounding":
                score += 0.75
        if "vulnerability_distress_loop" in patterns:
            if intervention == "vulnerability_reduction_plan":
                score += 1.5
            if intervention == "self_soothing":
                score += 1.0
        return score

    def _avoidanceo(self, kernel: Any, state: Any):
        return conde(
            [kernel.has_behavior(state, "avoidance")],
            [kernel.has_behavior(state, "procrastination")],
            [kernel.has_behavior(state, "withdrawal")],
        )

    def _identity_threato(self, kernel: Any, state: Any):
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
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "global_label"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "identity_global_rating"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "global_label"),
            ],
        )

    def _sadness_or_anxietyo(self, kernel: Any, state: Any):
        return conde(
            [kernel.has_emotion(state, "sadness")],
            [kernel.has_emotion(state, "anxiety")],
        )

    def _has_concrete_value_or_goalo(self, kernel: Any, state: Any):
        item = var()
        return conde(
            [kernel.has_value(state, item)],
            [kernel.has_goal(state, item)],
        )

    def _certainty_needo(self, kernel: Any, state: Any):
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

    def _approval_threato(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [kernel.state_feature(state, "approval_threat")],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "approval_demand"),
            ],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "mind_reading_claim"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "approval_demand"),
            ],
        )

    def _control_struggleo(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [kernel.state_feature(state, "control_struggle")],
            [
                kernel.has_thought(state, thought),
                kernel.thought_feature(thought, "control_struggle"),
            ],
            [
                kernel.has_belief(state, belief),
                kernel.belief_feature(belief, "control_struggle"),
            ],
        )

    def _distress_or_dysregulationo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "high_distress")],
            [kernel.state_feature(state, "emotion_dysregulation")],
            [kernel.has_emotion(state, "anxiety")],
            [kernel.has_emotion(state, "shame")],
            [kernel.has_emotion(state, "overwhelm")],
        )
