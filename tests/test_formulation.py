import unittest

from empath.chat import ExtractedCoachingState, ResponsePlan
from empath.formulation import CaseMemory, mirror_formulation


class CaseMemoryTests(unittest.TestCase):
    def test_memory_accumulates_observations_hypotheses_and_intervention(self):
        memory = CaseMemory()
        extraction = ExtractedCoachingState(
            situations=("Sending investor update",),
            thoughts=("They will think I am incompetent.",),
            emotions=("anxiety",),
            behaviors=("avoidance",),
            goals=("send investor update",),
        )
        kernel_snapshot = {
            "hypotheses": [
                {"source": "cbt", "pattern": "mind_reading"},
                {"source": "act", "pattern": "experiential_avoidance"},
            ],
            "candidates": [
                {
                    "intervention": "acceptance_committed_action",
                    "hypotheses": [
                        {"source": "act", "pattern": "experiential_avoidance"},
                    ],
                }
            ],
        }
        plan = ResponsePlan(
            validation="That makes sense.",
            intervention="acceptance_committed_action",
            exercise="Open the draft for ten minutes.",
        )

        delta = memory.apply_turn(
            extraction=extraction,
            kernel_snapshot=kernel_snapshot,
            response_plan=plan,
            message_index=2,
        )
        graph = memory.snapshot()
        labels = {node.label for node in graph.nodes}

        self.assertIn("They will think I am incompetent.", labels)
        self.assertIn("cbt: mind_reading", labels)
        self.assertIn("acceptance_committed_action", labels)
        self.assertIn("new map item", delta.summary)
        self.assertTrue(graph.edges)
        self.assertEqual(
            ("acceptance_committed_action",), memory.recent_interventions()
        )

    def test_memory_marks_recurring_items(self):
        memory = CaseMemory()
        extraction = ExtractedCoachingState(
            thoughts=("I am not cut out for this.",),
            emotions=("shame",),
        )
        plan = ResponsePlan(
            validation="That sounds painful.", intervention="self_compassion"
        )

        memory.apply_turn(extraction=extraction, kernel_snapshot={}, response_plan=plan)
        delta = memory.apply_turn(
            extraction=extraction, kernel_snapshot={}, response_plan=plan
        )
        node = next(node for node in memory.snapshot().nodes if node.label == "shame")

        self.assertEqual(2, node.seen_count)
        self.assertIn("recurring", delta.summary)

    def test_memory_clear_resets_graph_and_recent_interventions(self):
        memory = CaseMemory()
        extraction = ExtractedCoachingState(
            thoughts=("They will think I am incompetent.",),
            behaviors=("avoidance",),
        )
        plan = ResponsePlan(
            validation="That makes sense.",
            intervention="acceptance_committed_action",
        )

        memory.apply_turn(extraction=extraction, kernel_snapshot={}, response_plan=plan)
        cleared = memory.clear()

        self.assertEqual(0, cleared.turn_count)
        self.assertEqual((), cleared.nodes)
        self.assertEqual((), cleared.edges)
        self.assertEqual((), memory.recent_interventions())

    def test_memory_tracks_focus_context_nodes_and_edges(self):
        memory = CaseMemory()
        extraction = ExtractedCoachingState(
            situations=("Preparing for investor meeting",),
            concerns=("investor presentation",),
            tasks=("prepare investor presentation",),
            challenges=("avoidance or procrastination",),
            objectives=("deliver investor presentation",),
            stakes=("investor judgment",),
            domains=("work", "identity"),
            thoughts=("They will think I am incompetent.",),
            emotions=("anxiety",),
            behaviors=("avoidance",),
            goals=("complete a visible next step",),
        )
        plan = ResponsePlan(
            validation="That sounds high pressure.",
            intervention="acceptance_committed_action",
            exercise="Open the deck for ten minutes.",
        )

        memory.apply_turn(
            extraction=extraction,
            kernel_snapshot={},
            response_plan=plan,
        )
        graph = memory.snapshot()
        labels_by_kind = {}
        for node in graph.nodes:
            labels_by_kind.setdefault(node.kind, set()).add(node.label)
        edge_kinds = {edge.kind for edge in graph.edges}
        mirror = mirror_formulation(graph)

        self.assertIn("investor presentation", labels_by_kind["concern"])
        self.assertIn("prepare investor presentation", labels_by_kind["task"])
        self.assertIn("avoidance or procrastination", labels_by_kind["challenge"])
        self.assertIn("deliver investor presentation", labels_by_kind["objective"])
        self.assertIn("investor judgment", labels_by_kind["stake"])
        self.assertEqual({"work", "identity"}, labels_by_kind["domain"])
        self.assertIn("domain_of", edge_kinds)
        self.assertIn("about", edge_kinds)
        self.assertIn("involves_task", edge_kinds)
        self.assertIn("aims_at", edge_kinds)
        self.assertIn("blocks_or_complicates", edge_kinds)
        self.assertIn("raises_stakes_for", edge_kinds)
        self.assertIn("investor presentation", mirror.text)
        self.assertIn("avoidance or procrastination", mirror.text)

    def test_memory_adds_multi_turn_pattern_nodes(self):
        memory = CaseMemory()
        first = ExtractedCoachingState(
            emotions=("anxiety",),
            behaviors=("avoidance",),
        )
        second = ExtractedCoachingState(
            emotions=("anxiety",),
            behaviors=("procrastination",),
        )
        kernel_snapshot = {
            "hypotheses": ({"source": "act", "pattern": "experiential_avoidance"},)
        }
        plan = ResponsePlan(
            validation="That makes sense.",
            intervention="acceptance_committed_action",
        )

        memory.apply_turn(
            extraction=first,
            kernel_snapshot=kernel_snapshot,
            response_plan=plan,
        )
        delta = memory.apply_turn(
            extraction=second,
            kernel_snapshot=kernel_snapshot,
            response_plan=plan,
        )
        graph = memory.snapshot()

        self.assertIn(
            "recurring anxiety avoidance loop",
            {node.label for node in graph.nodes if node.kind == "longitudinal_pattern"},
        )
        self.assertTrue(delta.longitudinal_patterns)
        self.assertIn("Tentative multi-turn patterns", memory.longitudinal_context())

    def test_feedback_can_confirm_reject_and_remove_nodes(self):
        memory = CaseMemory()
        extraction = ExtractedCoachingState(emotions=("sadness",))
        plan = ResponsePlan(validation="That sounds hard.")
        memory.apply_turn(extraction=extraction, kernel_snapshot={}, response_plan=plan)
        sadness = next(
            node for node in memory.snapshot().nodes if node.label == "sadness"
        )

        confirmed = memory.apply_feedback(sadness.id, "confirm")
        rejected = memory.apply_feedback(sadness.id, "reject")
        removed = memory.apply_feedback(sadness.id, "remove")

        self.assertEqual("confirmed", confirmed.node.status)
        self.assertEqual("rejected", rejected.node.status)
        self.assertNotIn(sadness.id, {node.id for node in rejected.graph.nodes})
        self.assertEqual("removed", removed.node.status)
        self.assertNotIn(sadness.id, {node.id for node in removed.graph.nodes})

    def test_compaction_archives_stale_singletons(self):
        memory = CaseMemory(active_node_limit=20, archive_after_turns=2)
        plan = ResponsePlan(validation="That sounds hard.")

        memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("first",)),
            kernel_snapshot={},
            response_plan=plan,
        )
        first_id = next(
            node.id for node in memory.snapshot().nodes if node.label == "first"
        )
        memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("second",)),
            kernel_snapshot={},
            response_plan=plan,
        )
        memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("third",)),
            kernel_snapshot={},
            response_plan=plan,
        )

        self.assertNotIn(first_id, {node.id for node in memory.snapshot().nodes})
        archived = {
            node.id: node for node in memory.snapshot(include_archived=True).nodes
        }
        self.assertEqual("archived", archived[first_id].status)
        self.assertEqual(1, memory.snapshot().archived_node_count)

    def test_compaction_reactivates_archived_node_when_seen_again(self):
        memory = CaseMemory(active_node_limit=20, archive_after_turns=2)
        plan = ResponsePlan(validation="That sounds hard.")

        memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("sadness",)),
            kernel_snapshot={},
            response_plan=plan,
        )
        sadness_id = next(
            node.id for node in memory.snapshot().nodes if node.label == "sadness"
        )
        memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("anxiety",)),
            kernel_snapshot={},
            response_plan=plan,
        )
        memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("anger",)),
            kernel_snapshot={},
            response_plan=plan,
        )

        self.assertNotIn(sadness_id, {node.id for node in memory.snapshot().nodes})

        memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("sadness",)),
            kernel_snapshot={},
            response_plan=plan,
        )
        sadness = next(
            node for node in memory.snapshot().nodes if node.id == sadness_id
        )

        self.assertEqual("tentative", sadness.status)
        self.assertEqual(2, sadness.seen_count)

    def test_bounded_active_map_archives_low_priority_optional_nodes(self):
        memory = CaseMemory(active_node_limit=4, archive_after_turns=100)
        plan = ResponsePlan(validation="That sounds hard.")

        for index in range(8):
            memory.apply_turn(
                extraction=ExtractedCoachingState(emotions=(f"emotion-{index}",)),
                kernel_snapshot={},
                response_plan=plan,
            )

        graph = memory.snapshot()
        full_graph = memory.snapshot(include_archived=True)

        self.assertLessEqual(len(graph.nodes), 4)
        self.assertGreater(full_graph.archived_node_count, 0)
        self.assertIn("emotion-7", {node.label for node in graph.nodes})
        self.assertNotIn("emotion-0", {node.label for node in graph.nodes})

    def test_compaction_keeps_confirmed_node_active_when_bounded(self):
        memory = CaseMemory(active_node_limit=2, archive_after_turns=100)
        plan = ResponsePlan(validation="That sounds hard.")

        memory.apply_turn(
            extraction=ExtractedCoachingState(emotions=("important",)),
            kernel_snapshot={},
            response_plan=plan,
        )
        important = next(
            node for node in memory.snapshot().nodes if node.label == "important"
        )
        memory.apply_feedback(important.id, "confirm")
        for index in range(5):
            memory.apply_turn(
                extraction=ExtractedCoachingState(emotions=(f"optional-{index}",)),
                kernel_snapshot={},
                response_plan=plan,
            )

        graph = memory.snapshot()
        labels = {node.label for node in graph.nodes}

        self.assertIn("important", labels)
        self.assertIn("optional-4", labels)
        self.assertLessEqual(len(graph.nodes), 2)

    def test_mirror_formulation_excludes_rejected_nodes(self):
        memory = CaseMemory()
        extraction = ExtractedCoachingState(
            thoughts=("I am not cut out for this.",),
            emotions=("shame",),
            behaviors=("avoidance",),
        )
        plan = ResponsePlan(
            validation="That sounds painful.", intervention="self_compassion"
        )
        memory.apply_turn(extraction=extraction, kernel_snapshot={}, response_plan=plan)
        shame = next(node for node in memory.snapshot().nodes if node.label == "shame")
        memory.apply_feedback(shame.id, "reject")

        mirror = mirror_formulation(memory.snapshot())

        self.assertIn("working hypothesis", mirror.text)
        self.assertIn("avoidance", mirror.text)
        self.assertNotIn("shame", mirror.text)


if __name__ == "__main__":
    unittest.main()
