"""Neutral consultative facilitation layer for non-coaching requests."""

from __future__ import annotations

from typing import Any

from kanren import conde, eq, facts

from .base import TherapeuticSystem


class ConsultativeSystem(TherapeuticSystem):
    """Relational fallback for factual, advisory, analytical, and repair turns."""

    source = "consultative"

    def load_ontology(self, kernel: Any) -> None:
        facts(
            kernel.pattern_intervention,
            ("consultative_facilitation", "consultative_problem_structuring"),
            ("concise_factual_answer", "concise_factual_answer"),
            ("advisory_problem_solving", "advisory_recommendation"),
            ("instructional_explanation", "instructional_explanation"),
            ("analytical_synthesis", "analytical_synthesis"),
            ("creative_ideation", "creative_ideation"),
            ("socratic_inquiry", "socratic_inquiry"),
            ("interaction_repair", "active_listening_repair"),
        )
        facts(
            kernel.intervention_modality,
            ("consultative_problem_structuring", "consultative"),
            ("concise_factual_answer", "consultative"),
            ("advisory_recommendation", "consultative"),
            ("instructional_explanation", "consultative"),
            ("analytical_synthesis", "consultative"),
            ("creative_ideation", "consultative"),
            ("socratic_inquiry", "consultative"),
            ("active_listening_repair", "supportive"),
            ("active_listening_repair", "consultative"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "consultative_problem_structuring",
                "Clarify the objective, constraints, options, tradeoffs, and one practical next step.",
            ),
            (
                "concise_factual_answer",
                "Answer directly in two to four concise sentences, then note that Empath's strongest area is coaching and emotional support if that angle would be useful.",
            ),
            (
                "advisory_recommendation",
                "Give a recommendation, the rationale, the key tradeoff, and the next implementation step.",
            ),
            (
                "instructional_explanation",
                "Explain the concept briefly, give one example, and name one common mistake.",
            ),
            (
                "analytical_synthesis",
                "State the decision criteria, compare the options, and give a provisional conclusion.",
            ),
            (
                "creative_ideation",
                "Generate a small set of options, cluster them, and suggest one direction to refine.",
            ),
            (
                "socratic_inquiry",
                "Ask one question that tests the objective, assumption, evidence, or constraint.",
            ),
            (
                "active_listening_repair",
                "Reflect the frustration without defensiveness, keep unconditional positive regard, and offer Socratic inquiry, emotional support, or coaching expertise if desired.",
            ),
        )
        facts(
            kernel.pattern_recipe,
            ("consultative_facilitation", "clarify_structure_recommend"),
            ("advisory_problem_solving", "recommend_tradeoff_next_step"),
            ("analytical_synthesis", "criteria_compare_conclude"),
            ("creative_ideation", "diverge_cluster_refine"),
            ("interaction_repair", "reflect_repair_offer"),
        )
        facts(
            kernel.recipe_step,
            ("clarify_structure_recommend", 0, "consultative_problem_structuring"),
            ("clarify_structure_recommend", 1, "socratic_inquiry"),
            ("recommend_tradeoff_next_step", 0, "advisory_recommendation"),
            ("recommend_tradeoff_next_step", 1, "consultative_problem_structuring"),
            ("criteria_compare_conclude", 0, "analytical_synthesis"),
            ("criteria_compare_conclude", 1, "advisory_recommendation"),
            ("diverge_cluster_refine", 0, "creative_ideation"),
            ("diverge_cluster_refine", 1, "advisory_recommendation"),
            ("reflect_repair_offer", 0, "active_listening_repair"),
            ("reflect_repair_offer", 1, "socratic_inquiry"),
        )
        facts(
            kernel.recipe_rationale,
            (
                "clarify_structure_recommend",
                "Use a neutral facilitation stance when the request is practical rather than therapeutic.",
            ),
            (
                "recommend_tradeoff_next_step",
                "Give direct advice while showing the reasoning and the next step.",
            ),
            (
                "criteria_compare_conclude",
                "Keep analysis bounded by explicit criteria and a provisional conclusion.",
            ),
            (
                "diverge_cluster_refine",
                "Generate options first, then narrow toward a direction worth developing.",
            ),
            (
                "reflect_repair_offer",
                "Respond to aggression with non-defensive reflection and an offer of useful help.",
            ),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        return conde(
            [
                kernel.state_feature(state, "consultative_request"),
                eq(pattern, "consultative_facilitation"),
            ],
            [
                kernel.state_feature(state, "factual_question"),
                eq(pattern, "concise_factual_answer"),
            ],
            [
                kernel.state_feature(state, "advisory_request"),
                eq(pattern, "advisory_problem_solving"),
            ],
            [
                kernel.state_feature(state, "instructional_request"),
                eq(pattern, "instructional_explanation"),
            ],
            [
                kernel.state_feature(state, "analytical_request"),
                eq(pattern, "analytical_synthesis"),
            ],
            [
                kernel.state_feature(state, "creative_ideation_request"),
                eq(pattern, "creative_ideation"),
            ],
            [
                kernel.state_feature(state, "socratic_exploration_request"),
                eq(pattern, "socratic_inquiry"),
            ],
            [
                kernel.state_feature(state, "empath_directed_aggression"),
                eq(pattern, "interaction_repair"),
            ],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        if "interaction_repair" in patterns and intervention == "active_listening_repair":
            score += 5.0
        if "concise_factual_answer" in patterns and intervention == "concise_factual_answer":
            score += 3.5
        if "advisory_problem_solving" in patterns and intervention == "advisory_recommendation":
            score += 3.0
        if "instructional_explanation" in patterns and intervention == "instructional_explanation":
            score += 2.75
        if "analytical_synthesis" in patterns and intervention == "analytical_synthesis":
            score += 3.0
        if "creative_ideation" in patterns and intervention == "creative_ideation":
            score += 2.75
        if "socratic_inquiry" in patterns and intervention == "socratic_inquiry":
            score += 2.5
        if (
            "consultative_facilitation" in patterns
            and intervention == "consultative_problem_structuring"
        ):
            score += 1.25
        return score
