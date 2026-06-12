import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest

import anyio
from starlette.testclient import TestClient

from empath.api import ChatMessage, _local_conversation_context, create_app as create_empath_app
from empath.chat import DeterministicKernelGuidedCoach
from empath.storage import SurrealStateBackend


def create_app(*args: Any, **kwargs: Any):
    if (
        "store_backend" not in kwargs
        and "state_backend" not in kwargs
        and "state_file" not in kwargs
    ):
        kwargs["store_backend"] = "memory"
    return create_empath_app(*args, **kwargs)


class DictStateBackend:
    description = "dict-test"

    def __init__(self) -> None:
        self.data: dict[str, Any] | None = None
        self.load_count = 0
        self.save_count = 0

    async def load(self) -> dict[str, Any] | None:
        self.load_count += 1
        return json.loads(json.dumps(self.data)) if self.data is not None else None

    async def save(self, data: Mapping[str, Any]) -> None:
        self.save_count += 1
        self.data = json.loads(json.dumps(data))


class ApiSurfaceTests(unittest.TestCase):
    def test_health_and_chat_page(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        health = client.get("/api/health")
        page = client.get("/")

        self.assertEqual(200, health.status_code)
        self.assertTrue(health.json()["ok"])
        self.assertEqual("sse", health.json()["transport"])
        self.assertEqual("memory", health.json()["state_backend"])
        self.assertEqual(200, page.status_code)
        self.assertIn("Empath Chat", page.text)
        self.assertIn("EventSource", page.text)
        self.assertIn("loadSession", page.text)
        self.assertIn("newChat", page.text)
        self.assertIn("workspaceSelect", page.text)
        self.assertIn("renameWorkspace", page.text)
        self.assertIn("deleteWorkspace", page.text)
        self.assertIn("renameChat", page.text)
        self.assertIn("deleteChat", page.text)
        self.assertIn("/api/workspaces", page.text)
        self.assertIn("/api/conversations", page.text)
        self.assertIn("Why this?", page.text)
        self.assertIn("/api/chat/explain", page.text)
        self.assertIn("progressToast", page.text)
        self.assertIn("progressStages", page.text)
        self.assertIn("Reasoning through options", page.text)
        self.assertIn("Retrieving workspace memory", page.text)
        self.assertIn("messages.appendChild(progressToast)", page.text)
        self.assertIn("progressAnchor", page.text)
        self.assertIn('insertAdjacentElement("afterend", progressToast)', page.text)
        self.assertIn("prefer_existing", page.text)
        self.assertIn("loadSession({ preferExisting: true })", page.text)
        self.assertIn("height: 100dvh", page.text)
        self.assertIn("overflow: hidden", page.text)
        self.assertIn("min-height: 0", page.text)
        self.assertIn("align-content: start", page.text)
        self.assertIn("max-height: calc(100dvh - 170px)", page.text)
        self.assertIn("experiment-card", page.text)
        self.assertIn("message-actions", page.text)
        self.assertIn("mode-chip", page.text)
        self.assertIn("mode-explanation", page.text)
        self.assertIn("modeToggle", page.text)
        self.assertIn("toggleModeExplanation", page.text)
        self.assertIn("modeChipLabel", page.text)
        self.assertIn("counterfactual-chip", page.text)
        self.assertIn("counterfactual-card", page.text)
        self.assertIn("counterfactualToggle", page.text)
        self.assertIn("toggleCounterfactuals", page.text)
        self.assertIn("Why not others?", page.text)
        self.assertIn("user-turn-actions", page.text)
        self.assertIn("data-user-turn-action", page.text)
        self.assertIn("retryUserTurn", page.text)
        self.assertIn("/api/chat/retry", page.text)
        self.assertIn("data-experiment-toggle", page.text)
        self.assertIn("toggleExperiment", page.text)
        self.assertIn("/api/experiments/feedback", page.text)
        self.assertIn('source.addEventListener("experiment"', page.text)
        self.assertIn("Working Map", page.text)
        self.assertIn("Clear", page.text)
        self.assertIn("/api/formulation/clear", page.text)
        self.assertIn("clearWorkingMap", page.text)
        self.assertIn("clearMap", page.text)
        self.assertIn("Reflective listening", page.text)
        self.assertIn("What this is about", page.text)
        self.assertIn("Direction", page.text)
        self.assertIn("directionGroups", page.text)
        self.assertIn("renderDirectionOverview", page.text)
        self.assertIn("Projects", page.text)
        self.assertIn("Next actions", page.text)
        self.assertIn("Obstacles", page.text)
        self.assertIn("focus-section", page.text)
        self.assertIn("focusGroups", page.text)
        self.assertIn("renderFocusOverview", page.text)
        self.assertIn("Learning from feedback", page.text)
        self.assertIn("policySummary", page.text)
        self.assertIn("renderPolicy", page.text)
        self.assertIn('source.addEventListener("policy"', page.text)
        self.assertIn('source.addEventListener("memory"', page.text)
        self.assertIn("Concerns", page.text)
        self.assertIn("Tasks", page.text)
        self.assertIn("Challenges", page.text)
        self.assertIn("Objectives", page.text)
        self.assertIn("Stakes", page.text)
        self.assertIn("Domains", page.text)
        self.assertNotIn("Mirror back", page.text)
        self.assertIn("reflection-box", page.text)
        self.assertIn("Framework note", page.text)
        self.assertIn("info-box", page.text)
        self.assertIn("markdown-body", page.text)
        self.assertIn("renderMarkdownInto", page.text)
        self.assertIn("markdownToHtml", page.text)
        self.assertIn("escapeHtml", page.text)
        self.assertIn("sanitizeMarkdownUrl", page.text)
        self.assertIn("/api/formulation/mirror", page.text)
        self.assertIn("/api/formulation/feedback", page.text)
        self.assertIn("renderFormulation", page.text)
        self.assertIn("requestReflectiveListening", page.text)
        self.assertIn("displayRole", page.text)
        self.assertIn('source.addEventListener("formulation"', page.text)
        self.assertNotIn("position: fixed", page.text)
        self.assertLess(page.text.index('id="mapTab"'), page.text.index('id="traceTab"'))
        self.assertIn(
            '<button class="side-tab active" id="mapTab" type="button" role="tab" aria-selected="true"',
            page.text,
        )
        self.assertIn(
            '<button class="side-tab" id="traceTab" type="button" role="tab" aria-selected="false"',
            page.text,
        )
        self.assertIn(
            '<section class="side-panel" id="tracePanel" role="tabpanel" aria-labelledby="traceTab" hidden>',
            page.text,
        )

    def test_post_chat_returns_response_and_trace(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={
                "message": "I keep avoiding the prototype because I am a failure.",
                "trace": True,
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertIn("session_id", payload)
        self.assertIn("response", payload)
        self.assertEqual(["user", "assistant"], [item["role"] for item in payload["messages"]])
        self.assertFalse(payload["messages"][0]["trace_available"])
        self.assertTrue(payload["messages"][1]["trace_available"])
        self.assertEqual("coaching_turn", payload["messages"][1]["mode"])
        self.assertEqual("Coaching", payload["messages"][1]["mode_label"])
        self.assertIn("standard coaching", payload["messages"][1]["mode_explanation"])
        self.assertIn("counterfactuals", payload["messages"][1])
        self.assertGreaterEqual(len(payload["messages"][1]["counterfactuals"]), 2)
        self.assertEqual("selected", payload["messages"][1]["counterfactuals"][0]["role"])
        self.assertIn("expected_shift", payload["messages"][1]["counterfactuals"][0])
        self.assertNotIn("explanation", payload["messages"][1])
        self.assertIn("trace", payload)
        self.assertEqual("coaching_turn", payload["trace"]["route"]["mode"])
        self.assertIn("counterfactuals", payload["trace"])
        self.assertEqual(
            [
                "structured_extraction",
                "therapeutic_kernel",
                "response_plan",
                "renderer",
            ],
            payload["trace"]["pipeline"],
        )
        self.assertIn("kernel", payload["trace"])
        self.assertIn("formulation", payload)
        self.assertIn("formulation_delta", payload)
        self.assertTrue(payload["formulation"]["nodes"])
        self.assertIn("experiment", payload)
        self.assertEqual(payload["experiment"]["id"], payload["messages"][1]["experiment"]["id"])
        self.assertIn("prediction", payload["experiment"])
        self.assertIn("experiments", payload)

    def test_trace_explanation_is_generated_on_demand(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        chat = client.post(
            "/api/chat",
            json={
                "session_id": "explain-session",
                "message": "They will think I am incompetent.",
            },
        )
        before = client.get(
            "/api/chat/session",
            params={"session_id": "explain-session"},
        ).json()["messages"]
        explanation = client.get(
            "/api/chat/explain",
            params={
                "session_id": "explain-session",
                "message_index": 2,
            },
        )
        after = client.get(
            "/api/chat/session",
            params={"session_id": "explain-session"},
        ).json()["messages"]

        self.assertEqual(200, chat.status_code)
        self.assertNotIn("explanation", before[1])
        self.assertEqual(200, explanation.status_code)
        text = explanation.json()["explanation"]
        self.assertIn("Why this: evidence check", text)
        self.assertIn("Core tentative hypotheses", text)
        self.assertIn("Relevant differential formulation", text)
        self.assertIn("Backward check", text)
        self.assertIn("Not chosen now", text)
        self.assertIn("mind reading", text)
        self.assertIn("not a diagnosis", text)
        self.assertLessEqual(len(text.splitlines()), 14)
        self.assertIn("explanation", after[1])

    def test_trace_explanation_uses_surreal_graph_evidence_when_available(self):
        backend = SurrealStateBackend(
            url="mem://",
            namespace="empath_api_test",
            database="evidence",
            record_id="app_state:explain",
        )
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
            state_backend=backend,
        )
        client = TestClient(app)
        try:
            chat = client.post(
                "/api/chat",
                json={
                    "workspace_id": "graph-explain",
                    "conversation_id": "alpha",
                    "message": "They will think I am incompetent.",
                },
            )
            explanation = client.get(
                "/api/chat/explain",
                params={
                    "workspace_id": "graph-explain",
                    "conversation_id": "alpha",
                    "message_index": 2,
                },
            )
        finally:
            anyio.run(backend.close)

        self.assertEqual(200, chat.status_code)
        self.assertEqual(200, explanation.status_code)
        text = explanation.json()["explanation"]
        self.assertIn("Graph evidence", text)
        self.assertIn("Surreal working map", text)
        self.assertIn("Relevant transcript record", text)
        self.assertIn("They will think I am incompetent", text)
        self.assertIn("Working-map support", text)
        self.assertIn("Support path", text)
        self.assertLessEqual(len(text.splitlines()), 14)

    def test_response_trace_uses_surreal_memory_packet(self):
        backend = SurrealStateBackend(
            url="mem://",
            namespace="empath_api_test",
            database="memory_packet",
            record_id="app_state:memory",
        )
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
            state_backend=backend,
        )
        client = TestClient(app)
        try:
            first = client.post(
                "/api/chat",
                json={
                    "workspace_id": "memory-workspace",
                    "conversation_id": "alpha",
                    "message": (
                        "I keep avoiding the investor update because they will "
                        "think I am incompetent."
                    ),
                },
            )
            second = client.post(
                "/api/chat",
                json={
                    "workspace_id": "memory-workspace",
                    "conversation_id": "alpha",
                    "message": "I'm still avoiding the investor update.",
                    "trace": True,
                },
            )
            explanation = client.get(
                "/api/chat/explain",
                params={
                    "workspace_id": "memory-workspace",
                    "conversation_id": "alpha",
                    "message_index": 4,
                },
            )
        finally:
            anyio.run(backend.close)

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)
        payload = second.json()
        memory = payload["trace"]["memory"]
        self.assertEqual("surreal_projection", memory["source"])
        self.assertIn("memory_retrieval", payload["trace"]["pipeline"])
        self.assertIn("Retrieved workspace memory packet", memory["context"])
        self.assertTrue(memory["active_focus"])
        self.assertGreater(memory["counts"]["active_focus"], 0)
        self.assertEqual(200, explanation.status_code)
        text = explanation.json()["explanation"]
        self.assertIn("Memory used", text)
        self.assertIn("Memory source: Surreal projection", text)
        self.assertLessEqual(len(text.splitlines()), 16)

    def test_trace_explanation_requires_assistant_message(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "session_id": "bad-explain-session",
                "message": "They will think I am incompetent.",
            },
        )

        response = client.get(
            "/api/chat/explain",
            params={
                "session_id": "bad-explain-session",
                "message_index": 1,
            },
        )

        self.assertEqual(422, response.status_code)
        self.assertIn("assistant messages", response.json()["detail"])

    def test_session_transcript_accumulates_across_json_turns(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        first = client.post(
            "/api/chat",
            json={
                "session_id": "json-session",
                "message": "I'm feeling sad today.",
            },
        )
        second = client.post(
            "/api/chat",
            json={
                "session_id": "json-session",
                "message": "I also keep avoiding the prototype.",
            },
        )
        session = client.get(
            "/api/chat/session",
            params={"session_id": "json-session"},
        )

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)
        self.assertEqual(200, session.status_code)
        messages = session.json()["messages"]
        self.assertEqual(["user", "assistant", "user", "assistant"], [item["role"] for item in messages])
        self.assertIn("sad", messages[0]["text"])
        self.assertIn("prototype", messages[2]["text"])

    def test_retry_latest_user_turn_replaces_assistant_response(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        first = client.post(
            "/api/chat",
            json={
                "session_id": "retry-session",
                "message": "They will think I am incompetent.",
                "trace": True,
            },
        )

        retry = client.post(
            "/api/chat/retry",
            json={
                "session_id": "retry-session",
                "message_index": 1,
                "message": "I'm feeling sad today.",
                "trace": True,
            },
        )

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, retry.status_code)
        payload = retry.json()
        self.assertEqual(["user", "assistant"], [item["role"] for item in payload["messages"]])
        self.assertEqual("I'm feeling sad today.", payload["messages"][0]["text"])
        self.assertTrue(payload["messages"][1]["trace_available"])
        self.assertIn("trace", payload)
        labels = {node["label"] for node in payload["formulation"]["nodes"]}
        self.assertIn("sadness", labels)
        self.assertNotIn("cbt: mind_reading", labels)

    def test_retry_rejects_non_latest_user_turn(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "session_id": "retry-stale-session",
                "message": "I'm feeling sad today.",
            },
        )
        client.post(
            "/api/chat",
            json={
                "session_id": "retry-stale-session",
                "message": "They will think I am incompetent.",
            },
        )

        retry = client.post(
            "/api/chat/retry",
            json={
                "session_id": "retry-stale-session",
                "message_index": 1,
                "message": "Edit an older turn.",
            },
        )

        self.assertEqual(409, retry.status_code)
        self.assertIn("most recent user turn", retry.json()["detail"])

    def test_workspace_map_is_shared_across_conversations(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        first = client.post(
            "/api/chat",
            json={
                "workspace_id": "shared-workspace",
                "conversation_id": "conversation-a",
                "message": "I'm feeling sad today.",
            },
        )
        second_session = client.get(
            "/api/chat/session",
            params={
                "workspace_id": "shared-workspace",
                "conversation_id": "conversation-b",
            },
        )
        second = client.post(
            "/api/chat",
            json={
                "workspace_id": "shared-workspace",
                "conversation_id": "conversation-b",
                "message": "They will think I am incompetent.",
                "trace": True,
            },
        )
        first_session = client.get(
            "/api/chat/session",
            params={
                "workspace_id": "shared-workspace",
                "conversation_id": "conversation-a",
            },
        )

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)
        self.assertEqual("conversation-a", first.json()["conversation_id"])
        self.assertEqual("conversation-b", second.json()["conversation_id"])
        self.assertEqual([], second_session.json()["messages"])
        self.assertTrue(second_session.json()["formulation"]["nodes"])
        labels = {node["label"] for node in second.json()["formulation"]["nodes"]}
        self.assertIn("sadness", labels)
        self.assertIn("cbt: mind_reading", labels)
        self.assertEqual(
            ["user", "assistant"],
            [item["role"] for item in first_session.json()["messages"]],
        )
        self.assertEqual(
            "workspace",
            second.json()["trace"]["scope"]["working_map_scope"],
        )
        self.assertEqual(
            "bounded_last_5_user_turns",
            second.json()["trace"]["scope"]["message_history_scope"],
        )

    def test_local_conversation_context_is_bounded_to_last_five_user_turns(self):
        transcript = [
            ChatMessage(index=1, role="user", text="user turn 1"),
            ChatMessage(index=2, role="assistant", text="coach turn 1"),
            ChatMessage(index=3, role="user", text="user turn 2"),
            ChatMessage(index=4, role="info", text="framework note between turns"),
            ChatMessage(index=5, role="assistant", text="coach turn 2"),
            ChatMessage(index=6, role="reflection", text="reflective listening is out of band"),
            ChatMessage(index=7, role="user", text="user turn 3"),
            ChatMessage(index=8, role="assistant", text="coach turn 3"),
            ChatMessage(index=9, role="user", text="user turn 4"),
            ChatMessage(index=10, role="assistant", text="coach turn 4"),
            ChatMessage(index=11, role="user", text="user turn 5"),
            ChatMessage(index=12, role="assistant", text="coach turn 5"),
            ChatMessage(index=13, role="user", text="user turn 6"),
        ]

        context = _local_conversation_context(transcript)

        self.assertNotIn("user turn 1", context)
        self.assertIn("user: user turn 2", context)
        self.assertIn("info: framework note between turns", context)
        self.assertIn("coach: coach turn 5", context)
        self.assertIn("user: user turn 6", context)
        self.assertNotIn("reflective listening", context)

    def test_session_response_lists_workspace_conversations(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "workspace_id": "conversation-list-workspace",
                "conversation_id": "alpha",
                "message": "I'm feeling sad today.",
            },
        )
        client.post(
            "/api/chat",
            json={
                "workspace_id": "conversation-list-workspace",
                "conversation_id": "beta",
                "message": "They will think I am incompetent.",
            },
        )

        response = client.get(
            "/api/chat/session",
            params={
                "workspace_id": "conversation-list-workspace",
                "conversation_id": "alpha",
            },
        )

        self.assertEqual(200, response.status_code)
        conversations = response.json()["conversations"]
        ids = {item["conversation_id"] for item in conversations}
        self.assertEqual({"alpha", "beta"}, ids)
        active = [item for item in conversations if item["active"]]
        self.assertEqual(["alpha"], [item["conversation_id"] for item in active])
        self.assertIn("sad", active[0]["title"])
        workspaces = response.json()["workspaces"]
        self.assertEqual(
            ["conversation-list-workspace"],
            [item["workspace_id"] for item in workspaces],
        )

    def test_workspace_crud_endpoints_return_refreshed_session(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        created = client.post(
            "/api/workspaces",
            json={"title": "Founder coaching"},
        )

        self.assertEqual(200, created.status_code)
        payload = created.json()
        workspace_id = payload["workspace_id"]
        self.assertTrue(workspace_id.startswith("workspace-"))
        active = [item for item in payload["workspaces"] if item["active"]]
        self.assertEqual(["Founder coaching"], [item["title"] for item in active])
        self.assertEqual([], payload["messages"])
        self.assertEqual(1, len(payload["conversations"]))

        renamed = client.patch(
            "/api/workspaces",
            json={
                "workspace_id": workspace_id,
                "conversation_id": payload["conversation_id"],
                "title": "Leadership lab",
            },
        )

        self.assertEqual(200, renamed.status_code)
        renamed_active = [item for item in renamed.json()["workspaces"] if item["active"]]
        self.assertEqual(["Leadership lab"], [item["title"] for item in renamed_active])

        deleted = client.request(
            "DELETE",
            "/api/workspaces",
            json={"workspace_id": workspace_id},
        )

        self.assertEqual(200, deleted.status_code)
        deleted_payload = deleted.json()
        self.assertNotEqual(workspace_id, deleted_payload["workspace_id"])
        self.assertNotIn(
            workspace_id,
            {item["workspace_id"] for item in deleted_payload["workspaces"]},
        )

    def test_conversation_crud_endpoints_return_refreshed_session(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        created = client.post(
            "/api/conversations",
            json={
                "workspace_id": "crud-workspace",
                "title": "Investor update",
            },
        )

        self.assertEqual(200, created.status_code)
        payload = created.json()
        conversation_id = payload["conversation_id"]
        self.assertTrue(conversation_id.startswith("conversation-"))
        active = [item for item in payload["conversations"] if item["active"]]
        self.assertEqual(["Investor update"], [item["title"] for item in active])

        renamed = client.patch(
            "/api/conversations",
            json={
                "workspace_id": "crud-workspace",
                "conversation_id": conversation_id,
                "title": "Board prep",
            },
        )

        self.assertEqual(200, renamed.status_code)
        renamed_active = [item for item in renamed.json()["conversations"] if item["active"]]
        self.assertEqual(["Board prep"], [item["title"] for item in renamed_active])

        deleted = client.request(
            "DELETE",
            "/api/conversations",
            json={
                "workspace_id": "crud-workspace",
                "conversation_id": conversation_id,
            },
        )

        self.assertEqual(200, deleted.status_code)
        deleted_payload = deleted.json()
        self.assertNotEqual(conversation_id, deleted_payload["conversation_id"])
        self.assertNotIn(
            conversation_id,
            {item["conversation_id"] for item in deleted_payload["conversations"]},
        )

    def test_session_without_conversation_prefers_existing_workspace_conversation(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "workspace_id": "refresh-workspace",
                "conversation_id": "alpha",
                "message": "I'm feeling sad today.",
            },
        )
        client.post(
            "/api/chat",
            json={
                "workspace_id": "refresh-workspace",
                "conversation_id": "beta",
                "message": "They will think I am incompetent.",
            },
        )

        response = client.get(
            "/api/chat/session",
            params={
                "workspace_id": "refresh-workspace",
                "prefer_existing": "1",
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("beta", payload["conversation_id"])
        self.assertEqual(["user", "assistant"], [item["role"] for item in payload["messages"]])
        self.assertTrue(payload["formulation"]["nodes"])
        self.assertEqual(
            {"alpha", "beta"},
            {item["conversation_id"] for item in payload["conversations"]},
        )

    def test_session_with_stale_conversation_id_prefers_existing_when_requested(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "workspace_id": "stale-refresh-workspace",
                "conversation_id": "alpha",
                "message": "I'm feeling sad today.",
            },
        )

        response = client.get(
            "/api/chat/session",
            params={
                "workspace_id": "stale-refresh-workspace",
                "conversation_id": "stale-local-id",
                "prefer_existing": "1",
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("alpha", payload["conversation_id"])
        self.assertEqual(
            ["alpha"],
            [item["conversation_id"] for item in payload["conversations"]],
        )

    def test_stale_conversation_id_does_not_rehydrate_empty_reset_workspace(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        response = client.get(
            "/api/chat/session",
            params={
                "workspace_id": "reset-workspace",
                "conversation_id": "stale-local-id",
                "prefer_existing": "1",
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertNotEqual("stale-local-id", payload["conversation_id"])
        self.assertEqual([], payload["messages"])
        self.assertEqual(
            [payload["conversation_id"]],
            [item["conversation_id"] for item in payload["conversations"]],
        )

    def test_session_prefers_non_empty_conversation_over_empty_placeholder(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.get(
            "/api/chat/session",
            params={
                "workspace_id": "empty-placeholder-workspace",
                "conversation_id": "empty-local-id",
            },
        )
        client.post(
            "/api/chat",
            json={
                "workspace_id": "empty-placeholder-workspace",
                "conversation_id": "real-conversation",
                "message": "I'm feeling sad today.",
            },
        )

        response = client.get(
            "/api/chat/session",
            params={
                "workspace_id": "empty-placeholder-workspace",
                "conversation_id": "empty-local-id",
                "prefer_existing": "1",
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("real-conversation", payload["conversation_id"])
        self.assertEqual(["user", "assistant"], [item["role"] for item in payload["messages"]])

    def test_state_file_restores_conversations_and_workspace_map(self):
        with TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "empath-state.json"
            first_app = create_app(
                coach_factory=DeterministicKernelGuidedCoach,
                dry_run=True,
                state_file=state_file,
            )
            first_client = TestClient(first_app)
            first_client.post(
                "/api/chat",
                json={
                    "workspace_id": "persisted-workspace",
                    "conversation_id": "alpha",
                    "message": "I'm feeling sad today.",
                },
            )
            first_client.post(
                "/api/chat",
                json={
                    "workspace_id": "persisted-workspace",
                    "conversation_id": "beta",
                    "message": "They will think I am incompetent.",
                },
            )

            second_app = create_app(
                coach_factory=DeterministicKernelGuidedCoach,
                dry_run=True,
                state_file=state_file,
            )
            second_client = TestClient(second_app)
            response = second_client.get(
                "/api/chat/session",
                params={
                    "workspace_id": "persisted-workspace",
                    "conversation_id": "stale-local-id",
                    "prefer_existing": "1",
                },
            )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("beta", payload["conversation_id"])
        self.assertEqual(["user", "assistant"], [item["role"] for item in payload["messages"]])
        self.assertEqual(
            {"alpha", "beta"},
            {item["conversation_id"] for item in payload["conversations"]},
        )
        labels = {node["label"] for node in payload["formulation"]["nodes"]}
        self.assertIn("sadness", labels)
        self.assertIn("cbt: mind_reading", labels)

    def test_custom_state_backend_restores_conversations_and_workspace_map(self):
        backend = DictStateBackend()
        first_app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
            state_backend=backend,
        )
        first_client = TestClient(first_app)
        first_client.post(
            "/api/chat",
            json={
                "workspace_id": "backend-workspace",
                "conversation_id": "alpha",
                "message": "I'm feeling sad today.",
            },
        )

        second_app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
            state_backend=backend,
        )
        second_client = TestClient(second_app)
        response = second_client.get(
            "/api/chat/session",
            params={
                "workspace_id": "backend-workspace",
                "conversation_id": "alpha",
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("alpha", payload["conversation_id"])
        self.assertEqual(["user", "assistant"], [item["role"] for item in payload["messages"]])
        self.assertTrue(payload["formulation"]["nodes"])
        self.assertGreaterEqual(backend.load_count, 2)
        self.assertGreaterEqual(backend.save_count, 1)

    def test_workspace_maps_are_isolated(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        client.post(
            "/api/chat",
            json={
                "workspace_id": "workspace-one",
                "conversation_id": "conversation-one",
                "message": "I'm feeling sad today.",
            },
        )
        client.post(
            "/api/chat",
            json={
                "workspace_id": "workspace-two",
                "conversation_id": "conversation-two",
                "message": "I'm angry about the boundary conversation.",
            },
        )

        first = client.get(
            "/api/formulation",
            params={"workspace_id": "workspace-one"},
        )
        second = client.get(
            "/api/formulation",
            params={"workspace_id": "workspace-two"},
        )

        first_labels = {node["label"] for node in first.json()["nodes"]}
        second_labels = {node["label"] for node in second.json()["nodes"]}
        self.assertIn("sadness", first_labels)
        self.assertNotIn("anger", first_labels)
        self.assertIn("anger", second_labels)
        self.assertNotIn("sadness", second_labels)

    def test_session_endpoint_creates_empty_transcript(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        response = client.get(
            "/api/chat/session",
            params={"session_id": "new-session"},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("new-session", response.json()["session_id"])
        self.assertEqual([], response.json()["messages"])

    def test_framework_question_returns_info_message_without_kernel_artifacts(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={
                "session_id": "info-session",
                "message": "Can you explain ACT and CBT?",
                "trace": True,
            },
        )
        session = client.get(
            "/api/chat/session",
            params={"session_id": "info-session"},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(["user", "info"], [item["role"] for item in payload["messages"]])
        self.assertEqual("framework_note", payload["messages"][1]["mode"])
        self.assertEqual("Framework note", payload["messages"][1]["mode_label"])
        self.assertIn("ACT", payload["response"])
        self.assertIn("CBT", payload["response"])
        self.assertNotIn("trace", payload)
        self.assertNotIn("experiment", payload)
        self.assertNotIn("formulation_delta", payload)
        self.assertEqual([], payload["formulation"]["nodes"])
        self.assertEqual("info", session.json()["messages"][-1]["role"])
        self.assertNotIn("experiment", session.json()["messages"][-1])

    def test_consultative_question_skips_working_map_and_experiment(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={
                "session_id": "consultative-session",
                "message": "What is a checksum?",
                "trace": True,
            },
        )
        session = client.get(
            "/api/chat/session",
            params={"session_id": "consultative-session"},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(["user", "assistant"], [item["role"] for item in payload["messages"]])
        self.assertEqual("consultative_turn", payload["messages"][1]["mode"])
        self.assertEqual("Consultative", payload["messages"][1]["mode_label"])
        self.assertIn("information", payload["messages"][1]["mode_explanation"])
        self.assertEqual("consultative_turn", payload["trace"]["route"]["mode"])
        self.assertIn("A checksum is a short value", payload["response"])
        self.assertNotIn("neutral factual question", payload["response"])
        self.assertNotIn("In live mode", payload["response"])
        self.assertEqual(
            "concise_factual_answer",
            payload["trace"]["selection"]["intervention"],
        )
        self.assertEqual(
            "consultative_facilitation",
            payload["trace"]["kernel"]["operating_mode"]["mode"],
        )
        self.assertEqual("skipped", payload["trace"]["working_map"]["action"])
        self.assertNotIn("formulation_delta", payload)
        self.assertEqual([], payload["formulation"]["nodes"])
        self.assertNotIn("experiment", payload)
        self.assertNotIn("experiment", payload["messages"][1])
        self.assertNotIn("counterfactuals", payload["messages"][1])
        self.assertNotIn("experiment", session.json()["messages"][-1])

    def test_mbsr_framework_question_returns_info_message(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={
                "session_id": "mbsr-info-session",
                "message": "Can you explain MBSR?",
                "trace": True,
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(["user", "info"], [item["role"] for item in payload["messages"]])
        self.assertIn("MBSR", payload["response"])
        self.assertIn("mindfulness-based stress management", payload["response"])
        self.assertNotIn("trace", payload)
        self.assertNotIn("experiment", payload)

    def test_sse_chat_stream_emits_pipeline_events(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        with client.stream(
            "GET",
            "/api/chat/stream",
            params={
                "session_id": "test-session",
                "message": "They will think I am incompetent.",
                "trace": "1",
            },
        ) as response:
            body = response.read().decode("utf-8")

        events = _parse_sse(body)
        event_names = [event for event, _data in events]

        self.assertEqual(200, response.status_code)
        self.assertIn("session", event_names)
        self.assertIn("message", event_names)
        self.assertIn("extraction", event_names)
        self.assertIn("kernel", event_names)
        self.assertIn("plan", event_names)
        self.assertIn("formulation", event_names)
        self.assertIn("experiment", event_names)
        self.assertIn("response", event_names)
        self.assertIn("trace", event_names)
        self.assertIn("done", event_names)

        response_event = next(data for event, data in events if event == "response")
        kernel_event = next(data for event, data in events if event == "kernel")
        formulation_event = next(data for event, data in events if event == "formulation")
        experiment_event = next(data for event, data in events if event == "experiment")
        self.assertIn("text", response_event)
        self.assertEqual("assistant", response_event["message"]["role"])
        self.assertTrue(response_event["message"]["trace_available"])
        self.assertNotIn("explanation", response_event["message"])
        self.assertEqual(experiment_event["id"], response_event["experiment"]["id"])
        self.assertEqual(experiment_event["id"], response_event["message"]["experiment"]["id"])
        self.assertIn("counterfactuals", response_event["message"])
        self.assertGreaterEqual(len(response_event["message"]["counterfactuals"]), 2)
        self.assertIn("candidates", kernel_event)
        self.assertIn("graph", formulation_event)
        self.assertTrue(formulation_event["graph"]["nodes"])

    def test_sse_framework_question_emits_info_response_only(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        with client.stream(
            "GET",
            "/api/chat/stream",
            params={
                "session_id": "info-stream-session",
                "message": "What is DBT?",
                "trace": "1",
            },
        ) as response:
            body = response.read().decode("utf-8")

        events = _parse_sse(body)
        event_names = [event for event, _data in events]
        response_event = next(data for event, data in events if event == "response")

        self.assertEqual(200, response.status_code)
        self.assertIn("message", event_names)
        self.assertIn("response", event_names)
        self.assertIn("done", event_names)
        self.assertNotIn("extraction", event_names)
        self.assertNotIn("kernel", event_names)
        self.assertNotIn("formulation", event_names)
        self.assertNotIn("experiment", event_names)
        self.assertEqual("info", response_event["message"]["role"])
        self.assertEqual("framework_note", response_event["message"]["mode"])
        self.assertIn("DBT", response_event["text"])
        self.assertNotIn("experiment", response_event["message"])

    def test_session_and_formulation_endpoint_return_working_map(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "session_id": "map-session",
                "message": "They will think I am incompetent.",
            },
        )

        session = client.get(
            "/api/chat/session",
            params={"session_id": "map-session"},
        )
        graph = client.get(
            "/api/formulation",
            params={"session_id": "map-session"},
        )

        self.assertEqual(200, session.status_code)
        self.assertEqual(200, graph.status_code)
        self.assertTrue(session.json()["formulation"]["nodes"])
        self.assertEqual(
            session.json()["formulation"]["nodes"],
            graph.json()["nodes"],
        )

    def test_api_working_map_includes_focus_context_nodes(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={
                "session_id": "focus-map-session",
                "message": (
                    "I keep procrastinating on the investor presentation because "
                    "they'll judge me and decide I'm not cut out."
                ),
            },
        )

        self.assertEqual(200, response.status_code)
        graph = response.json()["formulation"]
        labels_by_kind = {}
        for node in graph["nodes"]:
            labels_by_kind.setdefault(node["kind"], set()).add(node["label"])
        edge_kinds = {edge["kind"] for edge in graph["edges"]}
        self.assertIn("investor presentation", labels_by_kind["concern"])
        self.assertIn("prepare investor presentation", labels_by_kind["task"])
        self.assertIn("deliver investor presentation", labels_by_kind["objective"])
        self.assertIn("investor judgment", labels_by_kind["stake"])
        self.assertIn("work", labels_by_kind["domain"])
        self.assertIn("identity", labels_by_kind["domain"])
        self.assertIn("aims_at", edge_kinds)
        self.assertIn("blocks_or_complicates", edge_kinds)

    def test_formulation_compaction_endpoint_returns_database_policy(self):
        backend = SurrealStateBackend(
            url="mem://",
            namespace="empath_api_test",
            database="compaction",
            record_id="app_state:compaction",
        )
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
            state_backend=backend,
        )
        client = TestClient(app)
        try:
            client.post(
                "/api/chat",
                json={
                    "workspace_id": "compaction-workspace",
                    "conversation_id": "alpha",
                    "message": "I'm sad today.",
                },
            )
            response = client.get(
                "/api/formulation/compaction",
                params={"workspace_id": "compaction-workspace"},
            )
        finally:
            anyio.run(backend.close)

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("db-compaction-v1", payload["policy_version"])
        self.assertEqual("default/compaction-workspace", payload["workspace_key"])
        self.assertGreater(payload["active_node_count"], 0)
        self.assertIn("active_examples", payload)
        self.assertIn("hidden_examples", payload)

    def test_multi_turn_patterns_show_in_working_map_and_trace(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        first = client.post(
            "/api/chat",
            json={
                "session_id": "multi-turn-session",
                "message": "I feel anxious and keep avoiding the investor update.",
            },
        )
        second = client.post(
            "/api/chat",
            json={
                "session_id": "multi-turn-session",
                "message": "I'm anxious again and procrastinating on the update.",
                "trace": True,
            },
        )

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)
        payload = second.json()
        graph = payload["formulation"]
        labels = {
            node["label"]
            for node in graph["nodes"]
            if node["kind"] == "longitudinal_pattern"
        }

        self.assertIn("recurring anxiety avoidance loop", labels)
        self.assertIn("longitudinal", payload["trace"])
        self.assertIn(
            "recurring anxiety avoidance loop",
            {
                item["label"]
                for item in payload["trace"]["longitudinal"]
            },
        )

    def test_formulation_feedback_updates_node_status(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "session_id": "feedback-session",
                "message": "I'm feeling sad today.",
            },
        )
        graph = client.get(
            "/api/formulation",
            params={"session_id": "feedback-session"},
        ).json()
        node_id = graph["nodes"][0]["id"]

        response = client.post(
            "/api/formulation/feedback",
            json={
                "session_id": "feedback-session",
                "node_id": node_id,
                "action": "confirm",
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("confirmed", response.json()["result"]["node"]["status"])
        self.assertIn("policy", response.json())
        self.assertEqual(1, response.json()["policy"]["counts"]["map_corrections"])

    def test_formulation_clear_resets_working_map_but_keeps_transcript(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "session_id": "clear-map-session",
                "message": "They will think I am incompetent.",
            },
        )

        response = client.post(
            "/api/formulation/clear",
            json={"session_id": "clear-map-session"},
        )
        session = client.get(
            "/api/chat/session",
            params={"session_id": "clear-map-session"},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual([], response.json()["formulation"]["nodes"])
        self.assertEqual([], response.json()["formulation"]["edges"])
        self.assertEqual(0, response.json()["formulation"]["turn_count"])
        self.assertEqual(["user", "assistant"], [item["role"] for item in session.json()["messages"]])
        self.assertEqual([], session.json()["formulation"]["nodes"])

    def test_formulation_mirror_reflects_working_map(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        client.post(
            "/api/chat",
            json={
                "session_id": "mirror-session",
                "message": "I keep avoiding the prototype because I am not cut out for this.",
            },
        )

        response = client.get(
            "/api/formulation/mirror",
            params={"session_id": "mirror-session"},
        )

        self.assertEqual(200, response.status_code)
        mirror = response.json()["mirror"]
        self.assertEqual(1, mirror["graph_turn"])
        self.assertIn("working hypothesis", mirror["text"])
        self.assertIn("correct the map", mirror["text"])
        self.assertTrue(mirror["node_ids"])
        self.assertEqual("reflection", response.json()["message"]["role"])
        self.assertFalse(response.json()["message"]["trace_available"])
        self.assertNotIn("experiment", response.json()["message"])

        session = client.get(
            "/api/chat/session",
            params={"session_id": "mirror-session"},
        )
        self.assertEqual("reflection", session.json()["messages"][-1]["role"])
        self.assertIn("working hypothesis", session.json()["messages"][-1]["text"])

    def test_sse_turn_appends_to_session_transcript(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        with client.stream(
            "GET",
            "/api/chat/stream",
            params={
                "session_id": "stream-session",
                "message": "They will think I am incompetent.",
            },
        ) as response:
            response.read()
        transcript = client.get(
            "/api/chat/session",
            params={"session_id": "stream-session"},
        ).json()["messages"]

        self.assertEqual(200, response.status_code)
        self.assertEqual(["user", "assistant"], [item["role"] for item in transcript])
        self.assertIn("incompetent", transcript[0]["text"])
        self.assertIn("experiment", transcript[1])

    def test_experiment_feedback_updates_experiment_status(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)
        chat = client.post(
            "/api/chat",
            json={
                "session_id": "experiment-session",
                "message": "I keep avoiding the prototype because they will think I am incompetent.",
            },
        )
        experiment_id = chat.json()["experiment"]["id"]

        response = client.post(
            "/api/experiments/feedback",
            json={
                "session_id": "experiment-session",
                "experiment_id": experiment_id,
                "action": "helped",
            },
        )
        listing = client.get(
            "/api/experiments",
            params={"session_id": "experiment-session"},
        )
        session = client.get(
            "/api/chat/session",
            params={"session_id": "experiment-session"},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("reviewed", response.json()["result"]["experiment"]["status"])
        self.assertEqual("helped", response.json()["result"]["experiment"]["outcome"])
        self.assertIn("useful evidence", response.json()["result"]["learning"])
        self.assertIn("policy", response.json())
        self.assertEqual(1, response.json()["policy"]["counts"]["experiment_outcomes"])
        self.assertTrue(response.json()["policy"]["helpful"])
        self.assertEqual("helped", listing.json()["experiments"][0]["outcome"])
        self.assertEqual(
            "helped",
            session.json()["messages"][1]["experiment"]["outcome"],
        )

    def test_sse_requires_message(self):
        app = create_app(
            coach_factory=DeterministicKernelGuidedCoach,
            dry_run=True,
        )
        client = TestClient(app)

        response = client.get("/api/chat/stream")

        self.assertEqual(400, response.status_code)
        self.assertIn("Missing message", response.json()["detail"])


def _parse_sse(body: str):
    events = []
    event_name = None
    data_lines = []
    for line in body.splitlines():
        if not line:
            if event_name and data_lines:
                events.append((event_name, json.loads("\n".join(data_lines))))
            event_name = None
            data_lines = []
            continue
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))
    if event_name and data_lines:
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


if __name__ == "__main__":
    unittest.main()
