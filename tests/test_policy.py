import unittest

from empath.experiments import CoachingExperiment
from empath.formulation import FormulationNode
from empath.policy import PolicyMemory


class PolicyMemoryTests(unittest.TestCase):
    def test_helpful_experiment_boosts_matching_candidate(self):
        policy = PolicyMemory()
        policy.record_experiment(
            CoachingExperiment(
                id="exp-1",
                created_turn=1,
                intervention="cognitive_defusion",
                focus="identity fusion",
                title="Defusion test",
                hypothesis="Test",
                action="Try defusion.",
                prediction="More space.",
                measure="Rate usefulness.",
                rationale="Because fusion was present.",
                outcome="helped",
            )
        )

        snapshot, report = policy.apply_to_kernel_snapshot(
            {
                "candidates": [
                    {"intervention": "evidence_check", "score": 4.5},
                    {"intervention": "cognitive_defusion", "score": 4.0},
                ]
            }
        )

        self.assertEqual("cognitive_defusion", snapshot["candidates"][0]["intervention"])
        self.assertGreater(snapshot["candidates"][0]["score"], 4.0)
        self.assertIn("Prior feedback", snapshot["candidates"][0]["policy_reasons"][0])
        self.assertEqual("cognitive_defusion", report["adjustments"][0]["intervention"])
        self.assertIn("cognitive defusion", policy.prompt_context())
        self.assertIn(
            ("cognitive_defusion", "helped", "identity fusion"),
            policy.relation_facts()["experiment_outcome"],
        )

    def test_too_hard_experiment_penalizes_matching_candidate(self):
        policy = PolicyMemory()
        policy.record_experiment(
            CoachingExperiment(
                id="exp-2",
                created_turn=1,
                intervention="rebt_disputation",
                focus="demandingness",
                title="Disputation test",
                hypothesis="Test",
                action="Try disputation.",
                prediction="More flexibility.",
                measure="Rate usefulness.",
                rationale="Because demandingness was present.",
                outcome="too_hard",
            )
        )

        snapshot, _report = policy.apply_to_kernel_snapshot(
            {
                "candidates": [
                    {"intervention": "rebt_disputation", "score": 5.0},
                ]
            }
        )

        candidate = snapshot["candidates"][0]
        self.assertLess(candidate["score"], 5.0)
        self.assertIn("too hard", candidate["policy_reasons"][0])
        self.assertIn("shrink", policy.prompt_context())

    def test_rejected_hypothesis_penalizes_supported_candidate(self):
        policy = PolicyMemory()
        policy.record_formulation(
            FormulationNode(
                id="node-1",
                kind="hypothesis",
                label="cbt: mind_reading",
                status="rejected",
                first_seen_turn=1,
                last_seen_turn=1,
            ),
            "reject",
        )

        snapshot, report = policy.apply_to_kernel_snapshot(
            {
                "candidates": [
                    {
                        "intervention": "evidence_check",
                        "score": 4.0,
                        "hypotheses": [
                            {"source": "cbt", "pattern": "mind_reading"},
                        ],
                    }
                ]
            }
        )

        candidate = snapshot["candidates"][0]
        self.assertLess(candidate["score"], 4.0)
        self.assertIn("pushed back", candidate["policy_reasons"][0])
        self.assertEqual("evidence_check", report["adjustments"][0]["intervention"])
        self.assertIn(
            ("hypothesis", "cbt: mind_reading", "reject"),
            policy.relation_facts()["formulation_feedback"],
        )


if __name__ == "__main__":
    unittest.main()
