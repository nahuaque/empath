"""REBT-specific relational ontology and pattern inference."""

from __future__ import annotations

from typing import Any

from kanren import Relation, conde, facts, lall, var

from .base import TherapeuticSystem


class REBTSystem(TherapeuticSystem):
    """Irrational belief hypotheses and REBT interventions."""

    source = "rebt"

    def __init__(self) -> None:
        self.feature_rebt_belief = Relation("feature_rebt_belief")

    def load_ontology(self, kernel: Any) -> None:
        facts(
            self.feature_rebt_belief,
            ("demanding_rule", "demandingness"),
            ("awful_outcome", "awfulizing"),
            ("unbearable_claim", "low_frustration_tolerance"),
            ("identity_global_rating", "self_downing"),
            ("global_label", "self_downing"),
            ("approval_demand", "approval_demandingness"),
            ("certainty_demand_claim", "certainty_demandingness"),
            ("failure_intolerance_claim", "failure_intolerance"),
        )
        facts(
            kernel.pattern_intervention,
            ("demandingness", "preference_rewrite"),
            ("demandingness", "rebt_disputation"),
            ("awfulizing", "decatastrophizing"),
            ("low_frustration_tolerance", "acceptance_practice"),
            ("low_frustration_tolerance", "frustration_tolerance_practice"),
            ("self_downing", "unconditional_self_acceptance"),
            ("self_downing", "rebt_disputation"),
            ("approval_demandingness", "approval_preference_rewrite"),
            ("certainty_demandingness", "uncertainty_tolerance_practice"),
            ("failure_intolerance", "failure_tolerance_reframe"),
        )
        facts(
            kernel.intervention_modality,
            ("preference_rewrite", "rebt"),
            ("rebt_disputation", "rebt"),
            ("unconditional_self_acceptance", "rebt"),
            ("acceptance_practice", "act"),
            ("frustration_tolerance_practice", "rebt"),
            ("approval_preference_rewrite", "rebt"),
            ("uncertainty_tolerance_practice", "rebt"),
            ("failure_tolerance_reframe", "rebt"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "unconditional_self_acceptance",
                "Separate the person's worth from one outcome or performance.",
            ),
            (
                "rebt_disputation",
                "Ask what evidence turns one bad outcome into a global verdict.",
            ),
            (
                "preference_rewrite",
                "Rewrite must/should language as a strong preference plus flexibility.",
            ),
            (
                "frustration_tolerance_practice",
                "Rewrite 'I cannot stand this' as 'This is hard, and I can stand a small dose of it for a short time.'",
            ),
            (
                "approval_preference_rewrite",
                "Rewrite the approval demand as: I strongly prefer their approval, but I can still act with self-respect without it.",
            ),
            (
                "uncertainty_tolerance_practice",
                "Name the decision, the uncertainty you want removed, and one reversible step you can take without full certainty.",
            ),
            (
                "failure_tolerance_reframe",
                "Separate failing at a task from being unable to recover, learn, or choose the next action.",
            ),
        )
        facts(kernel.high_intensity_intervention, ("rebt_disputation",))
        facts(kernel.premature_without_validation, ("rebt_disputation",))
        facts(
            kernel.safety_deferred_intervention,
            ("rebt_disputation",),
            ("approval_preference_rewrite",),
            ("uncertainty_tolerance_practice",),
            ("failure_tolerance_reframe",),
        )

    def irrational_beliefo(self, kernel: Any, belief: Any, belief_type: Any):
        feature = var()
        return lall(
            kernel.belief_feature(belief, feature),
            self.feature_rebt_belief(feature, belief_type),
        )

    def thought_irrational_beliefo(
        self,
        kernel: Any,
        thought: Any,
        belief_type: Any,
    ):
        feature = var()
        return lall(
            kernel.thought_feature(thought, feature),
            self.feature_rebt_belief(feature, belief_type),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        belief = var()
        thought = var()
        return conde(
            [
                kernel.has_belief(state, belief),
                self.irrational_beliefo(kernel, belief, pattern),
            ],
            [
                kernel.has_thought(state, thought),
                self.thought_irrational_beliefo(kernel, thought, pattern),
            ],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        if (
            "self_downing" in patterns
            and intervention == "unconditional_self_acceptance"
        ):
            score += 1.25
        if "demandingness" in patterns and intervention == "preference_rewrite":
            score += 1.0
        if (
            "low_frustration_tolerance" in patterns
            and intervention == "frustration_tolerance_practice"
        ):
            score += 1.25
        if (
            "approval_demandingness" in patterns
            and intervention == "approval_preference_rewrite"
        ):
            score += 1.25
        if (
            "certainty_demandingness" in patterns
            and intervention == "uncertainty_tolerance_practice"
        ):
            score += 1.25
        if (
            "failure_intolerance" in patterns
            and intervention == "failure_tolerance_reframe"
        ):
            score += 1.25
        return score
