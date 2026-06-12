"""CBT-specific relational ontology and pattern inference."""

from __future__ import annotations

from typing import Any

from kanren import Relation, facts, lall, var

from .base import TherapeuticSystem


class CBTSystem(TherapeuticSystem):
    """Cognitive distortion hypotheses and CBT interventions."""

    source = "cbt"

    def __init__(self) -> None:
        self.feature_distortion = Relation("feature_distortion")

    def load_ontology(self, kernel: Any) -> None:
        facts(
            self.feature_distortion,
            ("future_disaster", "catastrophizing"),
            ("awful_outcome", "catastrophizing"),
            ("mind_reading_claim", "mind_reading"),
            ("binary_evaluation", "all_or_nothing"),
            ("single_event_global_conclusion", "overgeneralization"),
            ("global_label", "global_labeling"),
            ("discounting_positive_claim", "discounting_positive"),
            ("feeling_as_fact", "emotional_reasoning"),
            ("personal_responsibility_claim", "personalization"),
            ("negative_filter", "mental_filter"),
        )
        facts(
            kernel.pattern_intervention,
            ("catastrophizing", "decatastrophizing"),
            ("catastrophizing", "socratic_question"),
            ("mind_reading", "evidence_check"),
            ("all_or_nothing", "continuum_technique"),
            ("overgeneralization", "behavioral_experiment"),
            ("global_labeling", "cognitive_reframe"),
            ("discounting_positive", "strength_evidence_log"),
            ("emotional_reasoning", "emotion_fact_separation"),
            ("personalization", "responsibility_pie"),
            ("mental_filter", "balanced_evidence_review"),
        )
        facts(
            kernel.intervention_modality,
            ("decatastrophizing", "cbt"),
            ("socratic_question", "cbt"),
            ("evidence_check", "cbt"),
            ("continuum_technique", "cbt"),
            ("behavioral_experiment", "cbt"),
            ("cognitive_reframe", "cbt"),
            ("strength_evidence_log", "cbt"),
            ("emotion_fact_separation", "cbt"),
            ("responsibility_pie", "cbt"),
            ("balanced_evidence_review", "cbt"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "decatastrophizing",
                "Estimate likelihood, worst case, best case, and most likely case.",
            ),
            (
                "evidence_check",
                "Separate observed facts from guesses about what others think.",
            ),
            (
                "continuum_technique",
                "Place the situation on a 0-100 scale instead of either/or labels.",
            ),
            (
                "behavioral_experiment",
                "Design a small test that could update the thought either way.",
            ),
            (
                "strength_evidence_log",
                "Write one concrete action you took, one skill it used, and why it counts even if it was imperfect.",
            ),
            (
                "emotion_fact_separation",
                "Write the feeling in one line, then list only observable facts in a second line.",
            ),
            (
                "responsibility_pie",
                "Draw a rough responsibility pie with all contributing factors before assigning blame to yourself.",
            ),
            (
                "balanced_evidence_review",
                "List the evidence your mind is focusing on, then add one neutral or positive fact it is leaving out.",
            ),
        )
        facts(
            kernel.premature_without_validation,
            ("cognitive_reframe",),
            ("behavioral_experiment",),
            ("emotion_fact_separation",),
            ("responsibility_pie",),
            ("balanced_evidence_review",),
        )
        facts(
            kernel.safety_deferred_intervention,
            ("behavioral_experiment",),
            ("cognitive_reframe",),
            ("decatastrophizing",),
            ("continuum_technique",),
            ("evidence_check",),
            ("strength_evidence_log",),
            ("emotion_fact_separation",),
            ("responsibility_pie",),
            ("balanced_evidence_review",),
        )

    def cognitive_distortiono(self, kernel: Any, thought: Any, distortion: Any):
        feature = var()
        return lall(
            kernel.thought_feature(thought, feature),
            self.feature_distortion(feature, distortion),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        thought = var()
        return lall(
            kernel.has_thought(state, thought),
            self.cognitive_distortiono(kernel, thought, pattern),
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        if (
            "discounting_positive" in patterns
            and intervention == "strength_evidence_log"
        ):
            score += 1.25
        if (
            "emotional_reasoning" in patterns
            and intervention == "emotion_fact_separation"
        ):
            score += 1.25
        if "personalization" in patterns and intervention == "responsibility_pie":
            score += 1.25
        if "mental_filter" in patterns and intervention == "balanced_evidence_review":
            score += 1.25
        return score
