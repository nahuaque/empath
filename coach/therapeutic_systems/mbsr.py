"""MBSR-specific stress-management ontology and mindfulness skills."""

from __future__ import annotations

from typing import Any

from kanren import conde, eq, facts, var

from .base import TherapeuticSystem


class MBSRSystem(TherapeuticSystem):
    """Mindfulness-Based Stress Reduction hypotheses and interventions."""

    source = "mbsr"

    def load_ontology(self, kernel: Any) -> None:
        facts(
            kernel.pattern_intervention,
            ("stress_load", "mindful_breathing_space"),
            ("stress_load", "stress_response_labeling"),
            ("stress_load", "body_scan_check_in"),
            ("somatic_stress", "mindful_breathing_space"),
            ("somatic_stress", "body_scan_check_in"),
            ("autopilot_reactivity", "mindful_pause"),
            ("rumination_stress", "mindful_observe_return"),
            ("vulnerability_stress", "mindful_self_care_check"),
        )
        facts(
            kernel.intervention_modality,
            ("mindful_breathing_space", "mbsr"),
            ("stress_response_labeling", "mbsr"),
            ("body_scan_check_in", "mbsr"),
            ("mindful_pause", "mbsr"),
            ("mindful_observe_return", "mbsr"),
            ("mindful_self_care_check", "mbsr"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "mindful_breathing_space",
                "Take three slow breaths, feel the body sitting or standing, and notice the stress response without needing to solve it yet.",
            ),
            (
                "stress_response_labeling",
                "Name the stress loop in three parts: body sensations, thoughts, and urges; then choose one kind next step.",
            ),
            (
                "body_scan_check_in",
                "Scan from face to shoulders to chest to belly, noticing tension, temperature, pressure, or ease without trying to force change.",
            ),
            (
                "mindful_pause",
                "Before reacting, pause for one breath, name 'reactivity is here', and choose the next action deliberately.",
            ),
            (
                "mindful_observe_return",
                "Notice the mind looping, label it 'thinking', and gently return attention to one breath or one body contact point.",
            ),
            (
                "mindful_self_care_check",
                "Check the basics: sleep, food, movement, overload, and recovery; choose one small support for the nervous system.",
            ),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        return conde(
            [self._stress_loado(kernel, state), eq(pattern, "stress_load")],
            [self._somatic_stresso(kernel, state), eq(pattern, "somatic_stress")],
            [
                self._autopilot_reactivityo(kernel, state),
                eq(pattern, "autopilot_reactivity"),
            ],
            [self._rumination_stresso(kernel, state), eq(pattern, "rumination_stress")],
            [
                self._vulnerability_stresso(kernel, state),
                eq(pattern, "vulnerability_stress"),
            ],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        if "stress_load" in patterns:
            if intervention == "mindful_breathing_space":
                score += 2.25
            if intervention == "stress_response_labeling":
                score += 1.25
            if intervention == "body_scan_check_in":
                score += 1.75
        if "somatic_stress" in patterns:
            if intervention == "body_scan_check_in":
                score += 2.5
            if intervention == "mindful_breathing_space":
                score += 1.25
        if "autopilot_reactivity" in patterns and intervention == "mindful_pause":
            score += 1.5
        if "rumination_stress" in patterns and intervention == "mindful_observe_return":
            score += 2.0
        if "vulnerability_stress" in patterns and intervention == "mindful_self_care_check":
            score += 1.25
        return score

    def _stress_loado(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "stress_load")],
            [kernel.has_emotion(state, "overwhelm")],
            [
                kernel.activation_level(state, "medium"),
                self._stress_related_emotiono(kernel, state),
            ],
        )

    def _somatic_stresso(self, kernel: Any, state: Any):
        sensation = var()
        return conde(
            [kernel.state_feature(state, "body_tension")],
            [
                kernel.has_bodily_sensation(state, sensation),
                self._stress_loado(kernel, state),
            ],
        )

    def _autopilot_reactivityo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "autopilot_reactivity")],
            [kernel.state_feature(state, "crisis_urge")],
        )

    def _rumination_stresso(self, kernel: Any, state: Any):
        return conde(
            [
                kernel.has_behavior(state, "rumination"),
                self._stress_loado(kernel, state),
            ],
            [
                kernel.has_behavior(state, "rumination"),
                kernel.state_feature(state, "body_tension"),
            ],
        )

    def _vulnerability_stresso(self, kernel: Any, state: Any):
        return conde(
            [
                kernel.state_feature(state, "vulnerability_factors"),
                self._stress_loado(kernel, state),
            ],
            [
                kernel.state_feature(state, "vulnerability_factors"),
                kernel.state_feature(state, "body_tension"),
            ],
        )

    def _stress_related_emotiono(self, kernel: Any, state: Any):
        return conde(
            [kernel.has_emotion(state, "anxiety")],
            [kernel.has_emotion(state, "anger")],
            [kernel.has_emotion(state, "overwhelm")],
        )
