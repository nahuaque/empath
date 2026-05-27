"""Focusing-specific relational ontology and felt-sense inference."""

from __future__ import annotations

from typing import Any

from kanren import conde, eq, facts, var

from .base import TherapeuticSystem


class FocusingSystem(TherapeuticSystem):
    """Gendlin-style Focusing hypotheses and interventions."""

    source = "focusing"

    def load_ontology(self, kernel: Any) -> None:
        facts(
            kernel.pattern_intervention,
            ("felt_sense_contact", "felt_sense_pause"),
            ("felt_sense_contact", "felt_sense_description"),
            ("unclear_felt_meaning", "felt_sense_description"),
            ("unclear_felt_meaning", "resonant_word_check"),
            ("symbolization_needed", "resonant_word_check"),
            ("symbolization_needed", "felt_shift_tracking"),
            ("inner_critic_presence", "inner_critic_distance"),
            ("felt_shift_possible", "felt_shift_tracking"),
        )
        facts(
            kernel.intervention_modality,
            ("felt_sense_pause", "focusing"),
            ("felt_sense_description", "focusing"),
            ("resonant_word_check", "focusing"),
            ("inner_critic_distance", "focusing"),
            ("felt_shift_tracking", "focusing"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "felt_sense_pause",
                "Pause and notice the whole felt sense of this situation in the body, without forcing an explanation.",
            ),
            (
                "felt_sense_description",
                "Describe the felt sense as a texture, shape, temperature, weight, image, or word.",
            ),
            (
                "resonant_word_check",
                "Try one word or phrase for the feeling, then check whether the body says yes, no, or not quite.",
            ),
            (
                "inner_critic_distance",
                "Notice the critical voice as one part of the experience, and gently ask it to give a little space.",
            ),
            (
                "felt_shift_tracking",
                "Notice whether anything softens, tightens, opens, or changes as you name it.",
            ),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        return conde(
            [
                self._felt_sense_contacto(kernel, state),
                eq(pattern, "felt_sense_contact"),
            ],
            [
                self._unclear_felt_meaningo(kernel, state),
                eq(pattern, "unclear_felt_meaning"),
            ],
            [
                self._symbolization_neededo(kernel, state),
                eq(pattern, "symbolization_needed"),
            ],
            [
                self._inner_critic_presenceo(kernel, state),
                eq(pattern, "inner_critic_presence"),
            ],
            [
                kernel.state_feature(state, "felt_shift"),
                eq(pattern, "felt_shift_possible"),
            ],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        if "felt_sense_contact" in patterns:
            if intervention == "felt_sense_pause":
                score += 1.5
            if intervention == "felt_sense_description":
                score += 1.0
        if "unclear_felt_meaning" in patterns:
            if intervention == "felt_sense_description":
                score += 1.25
            if intervention == "resonant_word_check":
                score += 1.25
        if "symbolization_needed" in patterns and intervention == "resonant_word_check":
            score += 1.5
        if "inner_critic_presence" in patterns and intervention == "inner_critic_distance":
            score += 1.5
        if "felt_shift_possible" in patterns and intervention == "felt_shift_tracking":
            score += 1.5
        return score

    def _felt_sense_contacto(self, kernel: Any, state: Any):
        sensation = var()
        return conde(
            [kernel.has_bodily_sensation(state, sensation)],
            [kernel.state_feature(state, "felt_sense")],
            [kernel.state_feature(state, "unclear_felt_sense")],
        )

    def _unclear_felt_meaningo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "unclear_felt_sense")],
            [kernel.state_feature(state, "hard_to_name")],
            [kernel.state_feature(state, "felt_sense")],
        )

    def _symbolization_neededo(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [kernel.state_feature(state, "symbolization_needed")],
            [kernel.has_thought(state, thought), kernel.thought_feature(thought, "symbolization_need")],
            [kernel.has_belief(state, belief), kernel.belief_feature(belief, "symbolization_need")],
        )

    def _inner_critic_presenceo(self, kernel: Any, state: Any):
        thought = var()
        belief = var()
        return conde(
            [kernel.state_feature(state, "inner_critic")],
            [kernel.has_thought(state, thought), kernel.thought_feature(thought, "inner_critic_claim")],
            [kernel.has_belief(state, belief), kernel.belief_feature(belief, "inner_critic_claim")],
        )
