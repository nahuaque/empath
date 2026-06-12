from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import anyio

from empath.storage import SurrealStateBackend


class SurrealStateBackendTests(unittest.TestCase):
    def test_memory_backend_round_trips_with_persistent_session(self):
        async def scenario() -> None:
            backend = SurrealStateBackend(
                url="mem://",
                namespace="empath_test",
                database="storage",
                record_id="app_state:memory",
            )
            await backend.save({"version": 1, "users": {"default": {}}})

            self.assertEqual(
                {"version": 1, "users": {"default": {}}},
                await backend.load(),
            )
            await backend.close()

        anyio.run(scenario)

    def test_save_projects_queryable_graph_records(self):
        async def scenario() -> None:
            backend = SurrealStateBackend(
                url="mem://",
                namespace="empath_test",
                database="projection",
                record_id="app_state:projection",
            )
            await backend.save(
                {
                    "version": 1,
                    "users": {
                        "default": {
                            "workspace-a": {
                                "activity_counter": 4,
                                "memory": {
                                    "turn_count": 2,
                                    "nodes": [
                                        {
                                            "id": "emotion:sadness",
                                            "kind": "emotion",
                                            "label": "sadness",
                                            "status": "tentative",
                                            "confidence": 0.72,
                                            "first_seen_turn": 1,
                                            "last_seen_turn": 2,
                                            "seen_count": 2,
                                            "provenance": [
                                                {
                                                    "turn": 1,
                                                    "source": "extraction",
                                                    "field": "emotions",
                                                    "evidence": "sadness",
                                                    "message_index": 2,
                                                }
                                            ],
                                        },
                                        {
                                            "id": "intervention:validation",
                                            "kind": "intervention",
                                            "label": "validation",
                                            "status": "tentative",
                                            "confidence": 0.65,
                                            "first_seen_turn": 1,
                                            "last_seen_turn": 1,
                                            "seen_count": 1,
                                        },
                                    ],
                                    "edges": [
                                        {
                                            "id": "edge:1",
                                            "source": "emotion:sadness",
                                            "target": "intervention:validation",
                                            "kind": "supports",
                                            "status": "tentative",
                                            "confidence": 0.55,
                                            "first_seen_turn": 1,
                                            "last_seen_turn": 1,
                                            "seen_count": 1,
                                        }
                                    ],
                                },
                                "conversations": {
                                    "alpha": {
                                        "created_order": 1,
                                        "updated_order": 4,
                                        "transcript": [
                                            {
                                                "index": 1,
                                                "role": "user",
                                                "text": "I'm sad today.",
                                            },
                                            {
                                                "index": 2,
                                                "role": "assistant",
                                                "text": "I hear you.",
                                                "trace_available": True,
                                            },
                                        ],
                                    }
                                },
                                "experiments": [
                                    {
                                        "id": "experiment-1",
                                        "status": "proposed",
                                        "intervention": "validation",
                                        "pattern": "minimal_disclosure",
                                        "test": "Try naming the feeling.",
                                        "try_step": "Name sadness out loud once.",
                                        "prediction": "Slightly more clarity.",
                                        "measure": "Rate usefulness 0-10.",
                                        "message_index": 2,
                                    }
                                ],
                            }
                        }
                    },
                }
            )
            db = await backend._database()

            nodes = await db.query(
                "SELECT node_id, kind, label, seen_count FROM working_node "
                "WHERE workspace_key = $workspace ORDER BY label",
                {"workspace": "default/workspace-a"},
            )
            edges = await db.query(
                "SELECT source_node_id, target_node_id, kind FROM working_edge "
                "WHERE workspace_key = $workspace",
                {"workspace": "default/workspace-a"},
            )
            messages = await db.query(
                "SELECT role, text, message_index FROM empath_message "
                "WHERE conversation_key = $conversation ORDER BY message_index",
                {"conversation": "default/workspace-a/alpha"},
            )
            experiments = await db.query(
                "SELECT experiment_id, intervention, try_step FROM coaching_experiment "
                "WHERE workspace_key = $workspace",
                {"workspace": "default/workspace-a"},
            )
            provenance = await db.query(
                "SELECT owner_id, evidence, message_index FROM working_node_provenance "
                "WHERE workspace_key = $workspace",
                {"workspace": "default/workspace-a"},
            )
            compaction = await db.query(
                "SELECT active_node_count, hidden_node_count, archived_node_count, "
                "policy_version FROM working_compaction_policy "
                "WHERE workspace_key = $workspace",
                {"workspace": "default/workspace-a"},
            )
            policy_nodes = await db.query(
                "SELECT label, active_in_map, compaction_reason, retention_action "
                "FROM working_node WHERE workspace_key = $workspace ORDER BY label",
                {"workspace": "default/workspace-a"},
            )

            self.assertEqual(
                ["sadness", "validation"], [item["label"] for item in nodes]
            )
            self.assertEqual("emotion:sadness", edges[0]["source_node_id"])
            self.assertEqual("intervention:validation", edges[0]["target_node_id"])
            self.assertEqual(["user", "assistant"], [item["role"] for item in messages])
            self.assertEqual("Name sadness out loud once.", experiments[0]["try_step"])
            self.assertEqual("sadness", provenance[0]["evidence"])
            self.assertEqual("db-compaction-v1", compaction[0]["policy_version"])
            self.assertEqual(2, compaction[0]["active_node_count"])
            self.assertEqual(0, compaction[0]["hidden_node_count"])
            self.assertEqual(
                ["protected_recurring", "active_within_budget"],
                [item["compaction_reason"] for item in policy_nodes],
            )
            await backend.close()

        anyio.run(scenario)

    def test_compaction_policy_projects_hidden_and_suppressed_nodes(self):
        async def scenario() -> None:
            backend = SurrealStateBackend(
                url="mem://",
                namespace="empath_test",
                database="compaction_policy",
                record_id="app_state:projection",
            )
            await backend.save(
                {
                    "version": 1,
                    "users": {
                        "default": {
                            "workspace-a": {
                                "memory": {
                                    "turn_count": 5,
                                    "active_node_limit": 4,
                                    "active_edge_limit": 10,
                                    "archive_after_turns": 2,
                                    "nodes": [
                                        {
                                            "id": "node:current",
                                            "kind": "emotion",
                                            "label": "current",
                                            "status": "tentative",
                                            "confidence": 0.6,
                                            "first_seen_turn": 5,
                                            "last_seen_turn": 5,
                                            "seen_count": 1,
                                        },
                                        {
                                            "id": "node:confirmed",
                                            "kind": "value",
                                            "label": "confirmed",
                                            "status": "confirmed",
                                            "confidence": 0.9,
                                            "first_seen_turn": 1,
                                            "last_seen_turn": 1,
                                            "seen_count": 1,
                                        },
                                        {
                                            "id": "node:archived",
                                            "kind": "situation",
                                            "label": "archived",
                                            "status": "archived",
                                            "confidence": 0.5,
                                            "first_seen_turn": 1,
                                            "last_seen_turn": 1,
                                            "seen_count": 1,
                                        },
                                        {
                                            "id": "node:rejected",
                                            "kind": "belief",
                                            "label": "rejected",
                                            "status": "rejected",
                                            "confidence": 0.2,
                                            "first_seen_turn": 1,
                                            "last_seen_turn": 4,
                                            "seen_count": 1,
                                        },
                                        {
                                            "id": "node:removed",
                                            "kind": "thought",
                                            "label": "removed",
                                            "status": "removed",
                                            "confidence": 0.0,
                                            "first_seen_turn": 1,
                                            "last_seen_turn": 4,
                                            "seen_count": 1,
                                        },
                                    ],
                                    "edges": [
                                        {
                                            "id": "edge:active",
                                            "source": "node:current",
                                            "target": "node:confirmed",
                                            "kind": "supports_hypothesis",
                                            "status": "tentative",
                                            "confidence": 0.6,
                                            "first_seen_turn": 5,
                                            "last_seen_turn": 5,
                                            "seen_count": 1,
                                        },
                                        {
                                            "id": "edge:hidden",
                                            "source": "node:archived",
                                            "target": "node:current",
                                            "kind": "supports_hypothesis",
                                            "status": "tentative",
                                            "confidence": 0.6,
                                            "first_seen_turn": 5,
                                            "last_seen_turn": 5,
                                            "seen_count": 1,
                                        },
                                    ],
                                },
                                "conversations": {},
                                "experiments": [],
                            }
                        }
                    },
                }
            )
            db = await backend._database()
            nodes = await db.query(
                "SELECT label, active_in_map, protected_by_policy, compaction_reason, "
                "retention_action FROM working_node "
                "WHERE workspace_key = $workspace ORDER BY label",
                {"workspace": "default/workspace-a"},
            )
            edges = await db.query(
                "SELECT edge_id, active_in_map, compaction_reason FROM working_edge "
                "WHERE workspace_key = $workspace ORDER BY edge_id",
                {"workspace": "default/workspace-a"},
            )
            summary = await backend.compaction_summary(
                user_id="default",
                workspace_id="workspace-a",
            )

            by_label = {item["label"]: item for item in nodes}
            self.assertEqual(
                "stale_singleton", by_label["archived"]["compaction_reason"]
            )
            self.assertEqual(
                "protected_confirmed", by_label["confirmed"]["compaction_reason"]
            )
            self.assertEqual("current_turn", by_label["current"]["compaction_reason"])
            self.assertEqual("user_rejected", by_label["rejected"]["compaction_reason"])
            self.assertEqual("user_removed", by_label["removed"]["compaction_reason"])
            self.assertTrue(by_label["confirmed"]["protected_by_policy"])
            self.assertFalse(by_label["archived"]["active_in_map"])
            self.assertEqual(
                ["active", "hidden_endpoint"],
                [item["compaction_reason"] for item in edges],
            )
            self.assertEqual(2, summary["active_node_count"])
            self.assertEqual(3, summary["hidden_node_count"])
            self.assertEqual(1, summary["stale_singleton_count"])
            self.assertEqual(2, summary["suppressed_node_count"])
            self.assertTrue(summary["hidden_examples"])
            await backend.close()

        anyio.run(scenario)

    def test_projection_rebuild_removes_stale_records(self):
        async def scenario() -> None:
            backend = SurrealStateBackend(
                url="mem://",
                namespace="empath_test",
                database="projection_rebuild",
                record_id="app_state:projection",
            )
            await backend.save(
                {
                    "version": 1,
                    "users": {
                        "default": {
                            "workspace-a": {
                                "memory": {
                                    "nodes": [
                                        {
                                            "id": "emotion:sadness",
                                            "kind": "emotion",
                                            "label": "sadness",
                                        }
                                    ],
                                    "edges": [],
                                },
                                "conversations": {},
                                "experiments": [],
                            }
                        }
                    },
                }
            )
            await backend.save(
                {
                    "version": 1,
                    "users": {
                        "default": {
                            "workspace-a": {
                                "memory": {
                                    "nodes": [
                                        {
                                            "id": "emotion:anxiety",
                                            "kind": "emotion",
                                            "label": "anxiety",
                                        }
                                    ],
                                    "edges": [],
                                },
                                "conversations": {},
                                "experiments": [],
                            }
                        }
                    },
                }
            )
            db = await backend._database()
            nodes = await db.query(
                "SELECT label FROM working_node WHERE workspace_key = $workspace",
                {"workspace": "default/workspace-a"},
            )

            self.assertEqual(["anxiety"], [item["label"] for item in nodes])
            await backend.close()

        anyio.run(scenario)

    def test_file_backend_persists_across_backend_instances(self):
        async def scenario() -> None:
            with TemporaryDirectory() as tmpdir:
                url = f"file://{Path(tmpdir) / 'empath-surreal-test.db'}"
                first = SurrealStateBackend(
                    url=url,
                    namespace="empath_test",
                    database="storage",
                    record_id="app_state:file",
                )
                await first.save(
                    {
                        "version": 1,
                        "users": {
                            "default": {
                                "workspace": {"activity_counter": 3},
                            },
                        },
                    }
                )
                await first.close()

                second = SurrealStateBackend(
                    url=url,
                    namespace="empath_test",
                    database="storage",
                    record_id="app_state:file",
                )
                self.assertEqual(
                    {
                        "version": 1,
                        "users": {
                            "default": {
                                "workspace": {"activity_counter": 3},
                            },
                        },
                    },
                    await second.load(),
                )
                await second.close()

        anyio.run(scenario)
