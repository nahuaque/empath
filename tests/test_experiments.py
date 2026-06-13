from types import SimpleNamespace
import unittest

from empath.chat import ExtractedCoachingState, ResponsePlan
from empath.experiments import ExperimentStore, propose_experiment
from empath.formulation import CaseMemory


class CoachingExperimentTests(unittest.TestCase):
    def test_proposes_experiment_from_selected_intervention(self):
        kernel_snapshot = {
            "hypotheses": [
                {"source": "act", "pattern": "experiential_avoidance"},
                {"source": "focus", "pattern": "goal_activation"},
            ],
            "candidates": [
                {
                    "intervention": "acceptance_committed_action",
                    "hypotheses": [
                        {"source": "act", "pattern": "experiential_avoidance"},
                        {"source": "focus", "pattern": "goal_activation"},
                    ],
                }
            ],
        }
        plan = ResponsePlan(
            validation="That makes sense.",
            intervention="acceptance_committed_action",
            exercise="Open the draft for ten minutes.",
        )
        memory = CaseMemory()
        delta = memory.apply_turn(
            extraction=ExtractedCoachingState(
                emotions=("anxiety",),
                behaviors=("avoidance",),
                goals=("send investor update",),
            ),
            kernel_snapshot=kernel_snapshot,
            response_plan=plan,
            message_index=2,
        )
        turn = SimpleNamespace(
            response_plan=plan,
            prepared=SimpleNamespace(kernel_snapshot=kernel_snapshot),
        )

        experiment = propose_experiment(
            turn=turn,
            formulation_delta=delta,
            message_index=2,
        )

        self.assertTrue(experiment.id.startswith("exp-1-"))
        self.assertEqual("acceptance_committed_action", experiment.intervention)
        self.assertEqual("goal activation", experiment.focus)
        self.assertIn("Open the draft", experiment.action)
        self.assertIn("urge-to-avoid", experiment.measure)
        self.assertIn("act:experiential_avoidance", experiment.pattern_keys)
        self.assertIn("focus:goal_activation", experiment.pattern_keys)
        self.assertTrue(experiment.support_node_ids)

    def test_store_records_feedback_and_learning(self):
        kernel_snapshot = {
            "hypotheses": [{"source": "cbt", "pattern": "mind_reading"}],
            "candidates": [
                {
                    "intervention": "evidence_check",
                    "hypotheses": [{"source": "cbt", "pattern": "mind_reading"}],
                }
            ],
        }
        plan = ResponsePlan(validation="That fits.", intervention="evidence_check")
        memory = CaseMemory()
        delta = memory.apply_turn(
            extraction=ExtractedCoachingState(thoughts=("They will judge me.",)),
            kernel_snapshot=kernel_snapshot,
            response_plan=plan,
        )
        turn = SimpleNamespace(
            response_plan=plan,
            prepared=SimpleNamespace(kernel_snapshot=kernel_snapshot),
        )
        store = ExperimentStore()
        experiment = store.propose(
            turn=turn,
            formulation_delta=delta,
            message_index=2,
        )

        result = store.apply_feedback(
            experiment.id,
            "too_hard",
            usefulness=3,
            friction_before=8,
            friction_after=7,
            action_taken="Looked at the evidence prompt but did not finish.",
            emotional_shift="Still anxious, slightly clearer.",
        )

        self.assertEqual("reviewed", result.experiment.status)
        self.assertEqual("too_hard", result.experiment.outcome)
        self.assertEqual(3, result.experiment.usefulness)
        self.assertEqual(
            "Looked at the evidence prompt but did not finish.",
            result.experiment.action_taken,
        )
        self.assertEqual(
            "Still anxious, slightly clearer.",
            result.experiment.emotional_shift,
        )
        self.assertIn("shrink", result.learning)
        self.assertEqual(1, len(result.experiments))

    def test_gentle_check_in_has_concrete_feeling_exercise(self):
        kernel_snapshot = {
            "hypotheses": [
                {"source": "loop", "pattern": "minimal_disclosure_sad_anxious"},
                {"source": "emotion", "pattern": "sadness"},
            ],
            "candidates": [
                {
                    "intervention": "gentle_check_in",
                    "hypotheses": [
                        {"source": "loop", "pattern": "minimal_disclosure_sad_anxious"},
                        {"source": "emotion", "pattern": "sadness"},
                    ],
                }
            ],
        }
        plan = ResponsePlan(
            validation="I hear you.",
            intervention="gentle_check_in",
            question="What is the sadness like today?",
        )
        memory = CaseMemory()
        delta = memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("sadness",)),
            kernel_snapshot=kernel_snapshot,
            response_plan=plan,
        )
        turn = SimpleNamespace(
            response_plan=plan,
            prepared=SimpleNamespace(
                extraction=ExtractedCoachingState(emotions=("sadness",)),
                kernel_snapshot=kernel_snapshot,
            ),
        )

        experiment = propose_experiment(
            turn=turn,
            formulation_delta=delta,
            message_index=2,
        )

        self.assertEqual("Gentle feeling-name exercise", experiment.title)
        self.assertIn("I am noticing sadness right now", experiment.action)
        self.assertNotIn("selected exercise", experiment.action)
        self.assertIn("one-minute", experiment.measure)


if __name__ == "__main__":
    unittest.main()
