import unittest

from empath.longitudinal import (
    LongitudinalTurn,
    detect_longitudinal_patterns,
    longitudinal_turn_from_data,
)


class LongitudinalPatternTests(unittest.TestCase):
    def test_detects_recurring_anxiety_avoidance_loop(self):
        turns = (
            longitudinal_turn_from_data(
                turn=1,
                extraction={
                    "emotions": ("anxiety",),
                    "behaviors": ("avoidance",),
                },
                kernel_snapshot={
                    "hypotheses": (
                        {"source": "act", "pattern": "experiential_avoidance"},
                    )
                },
                response_plan={"intervention": "acceptance_committed_action"},
            ),
            longitudinal_turn_from_data(
                turn=2,
                extraction={
                    "emotions": ("anxiety",),
                    "behaviors": ("procrastination",),
                },
                kernel_snapshot={
                    "hypotheses": (
                        {"source": "focus", "pattern": "avoidance_escape"},
                    )
                },
                response_plan={"intervention": "committed_action"},
            ),
        )

        patterns = {item.pattern: item for item in detect_longitudinal_patterns(turns)}

        self.assertIn("recurring_anxiety_avoidance_loop", patterns)
        self.assertEqual((1, 2), patterns["recurring_anxiety_avoidance_loop"].turns)
        self.assertTrue(patterns["recurring_anxiety_avoidance_loop"].support)

    def test_repeated_hypothesis_is_reported(self):
        turns = (
            LongitudinalTurn(
                turn=1,
                hypotheses=(("cbt", "mind_reading"),),
                signals=frozenset(),
            ),
            LongitudinalTurn(
                turn=2,
                hypotheses=(("cbt", "mind_reading"),),
                signals=frozenset(),
            ),
        )

        patterns = {item.pattern for item in detect_longitudinal_patterns(turns)}

        self.assertIn("recurring_cbt_mind_reading", patterns)


if __name__ == "__main__":
    unittest.main()
