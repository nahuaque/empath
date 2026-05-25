"""Persistence backends for workspace and working-map state.

The chat app keeps rich Python objects in memory while a request is active.
Storage backends persist a serialized snapshot of that state. This keeps the
therapeutic kernel independent from the database choice and lets us add a
graph-native backend incrementally.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Protocol

import anyio


class StateBackend(Protocol):
    """Durable snapshot backend for users, workspaces, and conversations."""

    @property
    def description(self) -> str:
        """Human-readable backend label for health checks and debugging."""

    async def load(self) -> dict[str, Any] | None:
        """Load a previously saved application snapshot, if one exists."""

    async def save(self, data: Mapping[str, Any]) -> None:
        """Persist one application snapshot."""


class JsonFileStateBackend:
    """JSON-file state backend matching the original app persistence format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def description(self) -> str:
        return f"json:{self.path}"

    async def load(self) -> dict[str, Any] | None:
        return await anyio.to_thread.run_sync(self._load_sync)

    async def save(self, data: Mapping[str, Any]) -> None:
        snapshot = _json_snapshot(data)
        await anyio.to_thread.run_sync(lambda: self._save_sync(snapshot))

    def _load_sync(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _save_sync(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.path)


class SurrealStateBackend:
    """SurrealDB snapshot backend.

    This is intentionally a document-shaped first step. SurrealDB gives us a
    maintained embedded-capable backend now; graph-native `working_node` and
    `working_edge` projections can be layered on top without changing API code.
    """

    def __init__(
        self,
        *,
        url: str,
        namespace: str = "coach",
        database: str = "coach",
        record_id: str = "app_state:default",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.url = url
        self.namespace = namespace
        self.database = database
        self.record_id = record_id
        self.username = username
        self.password = password
        self._lock = asyncio.Lock()
        self._context: _SurrealClientContext | None = None
        self._db: Any = None

    @property
    def description(self) -> str:
        return f"surreal:{self.url}/{self.namespace}/{self.database}/{self.record_id}"

    async def load(self) -> dict[str, Any] | None:
        async with self._lock:
            db = await self._database()
            record = await db.select(self.record_id)
        return _record_payload(record)

    async def save(self, data: Mapping[str, Any]) -> None:
        snapshot = _json_snapshot(data)
        async with self._lock:
            db = await self._database()
            if hasattr(db, "upsert"):
                await db.upsert(self.record_id, {"payload": snapshot})
            else:  # pragma: no cover - compatibility for older SDKs
                await db.update(self.record_id, {"payload": snapshot})
            await _project_snapshot(db, snapshot, app_record=self.record_id)

    async def explanation_evidence(
        self,
        *,
        user_id: str,
        workspace_id: str,
        conversation_id: str,
        message_index: int,
        trace: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return graph evidence for a trace explanation, if projection data exists."""

        async with self._lock:
            db = await self._database()
            return await _explanation_evidence(
                db,
                app_record=self.record_id,
                user_id=user_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                message_index=message_index,
                trace=trace,
            )

    async def compaction_summary(
        self,
        *,
        user_id: str,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        """Return projected database compaction policy for one workspace."""

        async with self._lock:
            db = await self._database()
            return await _compaction_summary(
                db,
                app_record=self.record_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )

    async def close(self) -> None:
        """Close the underlying SDK connection if one was opened."""

        async with self._lock:
            if self._context is not None:
                await self._context.__aexit__(None, None, None)
            self._context = None
            self._db = None

    async def _database(self) -> Any:
        if self._db is None:
            self._context = _SurrealClientContext(
                url=self.url,
                namespace=self.namespace,
                database=self.database,
                username=self.username,
                password=self.password,
            )
            self._db = await self._context.__aenter__()
        return self._db

    def _client(self) -> "_SurrealClientContext":
        return _SurrealClientContext(
            url=self.url,
            namespace=self.namespace,
            database=self.database,
            username=self.username,
            password=self.password,
        )


class _SurrealClientContext:
    """Small async context wrapper around the optional SurrealDB SDK."""

    def __init__(
        self,
        *,
        url: str,
        namespace: str,
        database: str,
        username: str | None,
        password: str | None,
    ) -> None:
        self.url = url
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        self._client: Any = None

    async def __aenter__(self) -> Any:
        client_class = _surreal_client_class()
        self._client = client_class(self.url)
        if hasattr(self._client, "__aenter__"):
            db = await self._client.__aenter__()
        else:  # pragma: no cover - defensive SDK compatibility
            db = self._client
            if hasattr(db, "connect"):
                await db.connect(self.url)
        if self.username and self.password and hasattr(db, "signin"):
            await _signin(db, self.username, self.password)
        if hasattr(db, "use"):
            await db.use(self.namespace, self.database)
        return db

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._client is None:
            return
        if hasattr(self._client, "__aexit__"):
            await self._client.__aexit__(exc_type, exc, tb)
        elif hasattr(self._client, "close"):  # pragma: no cover
            result = self._client.close()
            if hasattr(result, "__await__"):
                await result


def _surreal_client_class() -> type[Any]:
    try:
        module = importlib.import_module("surrealdb")
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "SurrealDB storage requires the surrealdb package. Run uv sync or "
            "install surrealdb>=2.0.0."
        ) from exc
    client_class = getattr(module, "AsyncSurreal", None) or getattr(
        module,
        "AsyncSurrealDB",
        None,
    )
    if client_class is None:  # pragma: no cover - SDK API boundary
        raise RuntimeError(
            "The installed surrealdb package does not expose AsyncSurreal."
        )
    return client_class


async def _signin(db: Any, username: str, password: str) -> None:
    try:
        await db.signin({"username": username, "password": password})
    except TypeError:  # pragma: no cover - SDK API boundary
        await db.signin({"user": username, "pass": password})


def _record_payload(record: Any) -> dict[str, Any] | None:
    if isinstance(record, list):
        if not record:
            return None
        record = record[0]
    if not isinstance(record, dict):
        return None
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else None


_PROJECTION_TABLES = (
    "coach_user",
    "coach_workspace",
    "coach_conversation",
    "coach_message",
    "working_node",
    "working_edge",
    "working_node_provenance",
    "working_edge_provenance",
    "coaching_experiment",
    "working_compaction_policy",
)


async def _project_snapshot(
    db: Any,
    snapshot: Mapping[str, Any],
    *,
    app_record: str,
) -> None:
    """Project the snapshot into queryable SurrealDB records."""

    await _clear_projection(db, app_record=app_record)
    users = snapshot.get("users")
    if not isinstance(users, Mapping):
        return

    for user_id, workspaces in users.items():
        user_id = str(user_id)
        if not isinstance(workspaces, Mapping):
            continue
        await db.upsert(
            _projection_record_id("coach_user", app_record, user_id),
            {
                "app_record": app_record,
                "user_id": user_id,
                "workspace_count": len(workspaces),
            },
        )
        for workspace_id, workspace in workspaces.items():
            if not isinstance(workspace, Mapping):
                continue
            await _project_workspace(
                db,
                app_record=app_record,
                user_id=user_id,
                workspace_id=str(workspace_id),
                workspace=workspace,
            )


async def _explanation_evidence(
    db: Any,
    *,
    app_record: str,
    user_id: str,
    workspace_id: str,
    conversation_id: str,
    message_index: int,
    trace: Mapping[str, Any],
) -> dict[str, Any] | None:
    workspace_key = _workspace_key(user_id, workspace_id)
    conversation_key = _conversation_key(user_id, workspace_id, conversation_id)
    selected_intervention = _selected_intervention(trace)
    selected_patterns = _selected_hypothesis_labels(trace)

    messages = await db.query(
        "SELECT role, text, message_index FROM coach_message "
        "WHERE app_record = $app_record "
        "AND conversation_key = $conversation_key "
        "AND message_index <= $message_index "
        "ORDER BY message_index",
        {
            "app_record": app_record,
            "conversation_key": conversation_key,
            "message_index": message_index,
        },
    )
    messages = [item for item in messages if isinstance(item, Mapping)][-4:]

    nodes = await db.query(
        "SELECT node_id, kind, label, status, confidence, seen_count, last_seen_turn "
        "FROM working_node "
        "WHERE app_record = $app_record AND workspace_key = $workspace_key",
        {"app_record": app_record, "workspace_key": workspace_key},
    )
    nodes = [item for item in nodes if isinstance(item, Mapping)]
    nodes_by_id = {
        str(item.get("node_id")): item
        for item in nodes
        if item.get("node_id") is not None
    }

    provenance = await db.query(
        "SELECT owner_id, owner_kind, source, field, evidence, message_index, turn "
        "FROM working_node_provenance "
        "WHERE app_record = $app_record "
        "AND workspace_key = $workspace_key "
        "AND message_index = $message_index",
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "message_index": message_index,
        },
    )
    provenance = [item for item in provenance if isinstance(item, Mapping)]

    selected_node_ids: set[str] = set()
    for item in provenance:
        if owner_id := _string_or_empty(item.get("owner_id")):
            selected_node_ids.add(owner_id)
    for node in nodes:
        kind = _string_or_empty(node.get("kind"))
        label = _string_or_empty(node.get("label"))
        node_id = _string_or_empty(node.get("node_id"))
        if not node_id:
            continue
        if selected_intervention and kind == "intervention" and label == selected_intervention:
            selected_node_ids.add(node_id)
        if kind == "hypothesis" and label in selected_patterns:
            selected_node_ids.add(node_id)

    edges = await db.query(
        "SELECT edge_id, source_node_id, target_node_id, kind, confidence, seen_count "
        "FROM working_edge "
        "WHERE app_record = $app_record AND workspace_key = $workspace_key",
        {"app_record": app_record, "workspace_key": workspace_key},
    )
    edges = [item for item in edges if isinstance(item, Mapping)]

    support_edges = []
    for edge in edges:
        source_id = _string_or_empty(edge.get("source_node_id"))
        target_id = _string_or_empty(edge.get("target_node_id"))
        kind = _string_or_empty(edge.get("kind"))
        if kind not in {"supports_hypothesis", "supports_intervention"}:
            continue
        if source_id in selected_node_ids or target_id in selected_node_ids:
            support_edges.append(
                {
                    **dict(edge),
                    "source_label": _node_label(nodes_by_id.get(source_id)),
                    "target_label": _node_label(nodes_by_id.get(target_id)),
                }
            )
            selected_node_ids.update(item for item in (source_id, target_id) if item)

    experiments = await db.query(
        "SELECT experiment_id, intervention, pattern, try_step, prediction, measure, status "
        "FROM coaching_experiment "
        "WHERE app_record = $app_record "
        "AND workspace_key = $workspace_key "
        "AND message_index = $message_index",
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "message_index": message_index,
        },
    )
    experiments = [item for item in experiments if isinstance(item, Mapping)]

    supporting_nodes = [
        nodes_by_id[node_id]
        for node_id in selected_node_ids
        if node_id in nodes_by_id
    ]
    supporting_nodes.sort(
        key=lambda item: (
            _node_priority(_string_or_empty(item.get("kind"))),
            _string_or_empty(item.get("label")).casefold(),
        )
    )
    support_edges.sort(
        key=lambda item: (
            _string_or_empty(item.get("kind")),
            _string_or_empty(item.get("source_label")).casefold(),
            _string_or_empty(item.get("target_label")).casefold(),
        )
    )

    if not (messages or supporting_nodes or support_edges or provenance or experiments):
        return None
    return {
        "source": "surreal_projection",
        "workspace_key": workspace_key,
        "conversation_key": conversation_key,
        "messages": messages,
        "supporting_nodes": supporting_nodes[:12],
        "support_edges": support_edges[:10],
        "provenance": provenance[:12],
        "experiments": experiments[:3],
    }


async def _compaction_summary(
    db: Any,
    *,
    app_record: str,
    user_id: str,
    workspace_id: str,
) -> dict[str, Any] | None:
    workspace_key = _workspace_key(user_id, workspace_id)
    summaries = await db.query(
        "SELECT * FROM working_compaction_policy "
        "WHERE app_record = $app_record AND workspace_key = $workspace_key",
        {"app_record": app_record, "workspace_key": workspace_key},
    )
    summaries = [item for item in summaries if isinstance(item, Mapping)]
    if not summaries:
        return None
    summary = dict(summaries[0])
    summary.pop("id", None)
    hidden_nodes = await db.query(
        "SELECT node_id, kind, label, status, compaction_reason, retention_action, age, priority_score "
        "FROM working_node "
        "WHERE app_record = $app_record "
        "AND workspace_key = $workspace_key "
        "AND active_in_map = false "
        "ORDER BY priority_score DESC LIMIT 12",
        {"app_record": app_record, "workspace_key": workspace_key},
    )
    active_nodes = await db.query(
        "SELECT node_id, kind, label, status, compaction_reason, retention_action, age, priority_score "
        "FROM working_node "
        "WHERE app_record = $app_record "
        "AND workspace_key = $workspace_key "
        "AND active_in_map = true "
        "ORDER BY priority_score DESC LIMIT 12",
        {"app_record": app_record, "workspace_key": workspace_key},
    )
    return {
        **summary,
        "active_examples": [dict(item) for item in active_nodes if isinstance(item, Mapping)],
        "hidden_examples": [dict(item) for item in hidden_nodes if isinstance(item, Mapping)],
    }


def _selected_intervention(trace: Mapping[str, Any]) -> str:
    selection = trace.get("selection") if isinstance(trace.get("selection"), Mapping) else {}
    matched = (
        selection.get("matched_candidate")
        if isinstance(selection.get("matched_candidate"), Mapping)
        else {}
    )
    return _string_or_empty(
        selection.get("intervention") or matched.get("intervention")
    )


def _selected_hypothesis_labels(trace: Mapping[str, Any]) -> set[str]:
    selection = trace.get("selection") if isinstance(trace.get("selection"), Mapping) else {}
    matched = (
        selection.get("matched_candidate")
        if isinstance(selection.get("matched_candidate"), Mapping)
        else {}
    )
    hypotheses = matched.get("hypotheses")
    if not hypotheses:
        kernel = trace.get("kernel") if isinstance(trace.get("kernel"), Mapping) else {}
        hypotheses = kernel.get("hypotheses")
    labels: set[str] = set()
    for item in hypotheses or ():
        if not isinstance(item, Mapping):
            continue
        source = _string_or_empty(item.get("source"))
        pattern = _string_or_empty(item.get("pattern"))
        if source and pattern:
            labels.add(f"{source}: {pattern}")
    return labels


def _node_label(node: Mapping[str, Any] | None) -> str:
    if not node:
        return ""
    return _string_or_empty(node.get("label")) or _string_or_empty(node.get("node_id"))


def _node_priority(kind: str) -> int:
    order = {
        "thought": 0,
        "belief": 1,
        "emotion": 2,
        "behavior": 3,
        "challenge": 4,
        "obstacle": 5,
        "hypothesis": 6,
        "intervention": 7,
    }
    return order.get(kind, 20)


async def _clear_projection(db: Any, *, app_record: str) -> None:
    for table in _PROJECTION_TABLES:
        await db.query(
            f"DELETE {table} WHERE app_record = $app_record",
            {"app_record": app_record},
        )


async def _project_workspace(
    db: Any,
    *,
    app_record: str,
    user_id: str,
    workspace_id: str,
    workspace: Mapping[str, Any],
) -> None:
    workspace_key = _workspace_key(user_id, workspace_id)
    memory = workspace.get("memory") if isinstance(workspace.get("memory"), Mapping) else {}
    conversations = (
        workspace.get("conversations") if isinstance(workspace.get("conversations"), Mapping) else {}
    )
    experiments = workspace.get("experiments") if isinstance(workspace.get("experiments"), list) else []
    nodes = memory.get("nodes") if isinstance(memory, Mapping) and isinstance(memory.get("nodes"), list) else []
    edges = memory.get("edges") if isinstance(memory, Mapping) and isinstance(memory.get("edges"), list) else []
    turn_count = _int_or_zero(memory.get("turn_count") if isinstance(memory, Mapping) else None)
    policy_config = {
        "active_node_limit": _int_or_zero(memory.get("active_node_limit") if isinstance(memory, Mapping) else None),
        "active_edge_limit": _int_or_zero(memory.get("active_edge_limit") if isinstance(memory, Mapping) else None),
        "archive_after_turns": _int_or_zero(memory.get("archive_after_turns") if isinstance(memory, Mapping) else None),
    }
    node_policies = _node_compaction_policies(
        nodes,
        turn_count=turn_count,
        archive_after_turns=policy_config["archive_after_turns"],
    )
    active_node_ids = {
        node_id
        for node_id, policy in node_policies.items()
        if policy["active_in_map"]
    }
    edge_policies = _edge_compaction_policies(
        edges,
        turn_count=turn_count,
        active_node_ids=active_node_ids,
    )
    compaction_summary = _workspace_compaction_policy(
        nodes=nodes,
        edges=edges,
        node_policies=node_policies,
        edge_policies=edge_policies,
        turn_count=turn_count,
        policy_config=policy_config,
    )

    await db.upsert(
        _projection_record_id("coach_workspace", app_record, workspace_key),
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "activity_counter": _int_or_zero(workspace.get("activity_counter")),
            "turn_count": turn_count,
            "conversation_count": len(conversations),
            "working_node_count": len(nodes),
            "working_edge_count": len(edges),
            "experiment_count": len(experiments),
            "active_node_count": compaction_summary["active_node_count"],
            "hidden_node_count": compaction_summary["hidden_node_count"],
            "archived_node_count": compaction_summary["archived_node_count"],
        },
    )
    await db.upsert(
        _projection_record_id("working_compaction_policy", app_record, workspace_key),
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "user_id": user_id,
            "workspace_id": workspace_id,
            **compaction_summary,
        },
    )

    for conversation_id, conversation in conversations.items():
        if isinstance(conversation, Mapping):
            await _project_conversation(
                db,
                app_record=app_record,
                user_id=user_id,
                workspace_id=workspace_id,
                workspace_key=workspace_key,
                conversation_id=str(conversation_id),
                conversation=conversation,
            )

    for node in nodes:
        if isinstance(node, Mapping):
            await _project_working_node(
                db,
                app_record=app_record,
                user_id=user_id,
                workspace_id=workspace_id,
                workspace_key=workspace_key,
                node=node,
                policy=node_policies.get(_string_or_empty(node.get("id"))) or {},
            )

    for edge in edges:
        if isinstance(edge, Mapping):
            await _project_working_edge(
                db,
                app_record=app_record,
                user_id=user_id,
                workspace_id=workspace_id,
                workspace_key=workspace_key,
                edge=edge,
                policy=edge_policies.get(_string_or_empty(edge.get("id"))) or {},
            )

    for experiment in experiments:
        if isinstance(experiment, Mapping):
            await _project_experiment(
                db,
                app_record=app_record,
                user_id=user_id,
                workspace_id=workspace_id,
                workspace_key=workspace_key,
                experiment=experiment,
            )


async def _project_conversation(
    db: Any,
    *,
    app_record: str,
    user_id: str,
    workspace_id: str,
    workspace_key: str,
    conversation_id: str,
    conversation: Mapping[str, Any],
) -> None:
    conversation_key = _conversation_key(user_id, workspace_id, conversation_id)
    transcript = conversation.get("transcript") if isinstance(conversation.get("transcript"), list) else []
    await db.upsert(
        _projection_record_id("coach_conversation", app_record, conversation_key),
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "conversation_key": conversation_key,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "created_order": _int_or_zero(conversation.get("created_order")),
            "updated_order": _int_or_zero(conversation.get("updated_order")),
            "message_count": len(transcript),
            "last_message": _last_message_text(transcript),
        },
    )
    for message in transcript:
        if not isinstance(message, Mapping):
            continue
        await db.upsert(
            _projection_record_id(
                "coach_message",
                app_record,
                conversation_key,
                str(message.get("index")),
            ),
            {
                "app_record": app_record,
                "workspace_key": workspace_key,
                "conversation_key": conversation_key,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "conversation_id": conversation_id,
                "message_index": _int_or_zero(message.get("index")),
                "role": _string_or_empty(message.get("role")),
                "text": _string_or_empty(message.get("text")),
                "trace_available": bool(message.get("trace_available")),
                "experiment_id": _nested_string(message, "experiment", "id"),
            },
        )


async def _project_working_node(
    db: Any,
    *,
    app_record: str,
    user_id: str,
    workspace_id: str,
    workspace_key: str,
    node: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    node_id = _string_or_empty(node.get("id"))
    if not node_id:
        return
    await db.upsert(
        _projection_record_id("working_node", app_record, workspace_key, node_id),
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "node_id": node_id,
            "kind": _string_or_empty(node.get("kind")),
            "label": _string_or_empty(node.get("label")),
            "status": _string_or_empty(node.get("status")),
            "confidence": _float_or_zero(node.get("confidence")),
            "first_seen_turn": _int_or_zero(node.get("first_seen_turn")),
            "last_seen_turn": _int_or_zero(node.get("last_seen_turn")),
            "seen_count": _int_or_zero(node.get("seen_count")),
            "provenance_count": len(node.get("provenance") or ()),
            "age": _int_or_zero(policy.get("age")),
            "active_in_map": bool(policy.get("active_in_map")),
            "hidden_by_policy": bool(policy.get("hidden_by_policy")),
            "protected_by_policy": bool(policy.get("protected_by_policy")),
            "compaction_reason": _string_or_empty(policy.get("compaction_reason")),
            "retention_action": _string_or_empty(policy.get("retention_action")),
            "priority_score": _float_or_zero(policy.get("priority_score")),
        },
    )
    for index, provenance in enumerate(node.get("provenance") or ()):
        if isinstance(provenance, Mapping):
            await _project_provenance(
                db,
                table="working_node_provenance",
                app_record=app_record,
                workspace_key=workspace_key,
                owner_id=node_id,
                owner_kind="node",
                index=index,
                provenance=provenance,
            )


async def _project_working_edge(
    db: Any,
    *,
    app_record: str,
    user_id: str,
    workspace_id: str,
    workspace_key: str,
    edge: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    edge_id = _string_or_empty(edge.get("id"))
    if not edge_id:
        return
    await db.upsert(
        _projection_record_id("working_edge", app_record, workspace_key, edge_id),
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "edge_id": edge_id,
            "source_node_id": _string_or_empty(edge.get("source")),
            "target_node_id": _string_or_empty(edge.get("target")),
            "kind": _string_or_empty(edge.get("kind")),
            "status": _string_or_empty(edge.get("status")),
            "confidence": _float_or_zero(edge.get("confidence")),
            "first_seen_turn": _int_or_zero(edge.get("first_seen_turn")),
            "last_seen_turn": _int_or_zero(edge.get("last_seen_turn")),
            "seen_count": _int_or_zero(edge.get("seen_count")),
            "provenance_count": len(edge.get("provenance") or ()),
            "age": _int_or_zero(policy.get("age")),
            "active_in_map": bool(policy.get("active_in_map")),
            "hidden_by_policy": bool(policy.get("hidden_by_policy")),
            "compaction_reason": _string_or_empty(policy.get("compaction_reason")),
            "retention_action": _string_or_empty(policy.get("retention_action")),
            "priority_score": _float_or_zero(policy.get("priority_score")),
        },
    )
    for index, provenance in enumerate(edge.get("provenance") or ()):
        if isinstance(provenance, Mapping):
            await _project_provenance(
                db,
                table="working_edge_provenance",
                app_record=app_record,
                workspace_key=workspace_key,
                owner_id=edge_id,
                owner_kind="edge",
                index=index,
                provenance=provenance,
            )


async def _project_provenance(
    db: Any,
    *,
    table: str,
    app_record: str,
    workspace_key: str,
    owner_id: str,
    owner_kind: str,
    index: int,
    provenance: Mapping[str, Any],
) -> None:
    await db.upsert(
        _projection_record_id(table, app_record, workspace_key, owner_id, str(index)),
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "owner_id": owner_id,
            "owner_kind": owner_kind,
            "provenance_index": index,
            "turn": _int_or_zero(provenance.get("turn")),
            "source": _string_or_empty(provenance.get("source")),
            "field": _string_or_empty(provenance.get("field")),
            "evidence": _string_or_empty(provenance.get("evidence")),
            "message_index": _optional_int(provenance.get("message_index")),
        },
    )


async def _project_experiment(
    db: Any,
    *,
    app_record: str,
    user_id: str,
    workspace_id: str,
    workspace_key: str,
    experiment: Mapping[str, Any],
) -> None:
    experiment_id = _string_or_empty(experiment.get("id"))
    if not experiment_id:
        return
    await db.upsert(
        _projection_record_id("coaching_experiment", app_record, workspace_key, experiment_id),
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "experiment_id": experiment_id,
            "status": _string_or_empty(experiment.get("status")),
            "intervention": _string_or_empty(experiment.get("intervention")),
            "pattern": _string_or_empty(experiment.get("pattern")),
            "test": _string_or_empty(experiment.get("test")),
            "try_step": _string_or_empty(experiment.get("try_step")),
            "prediction": _string_or_empty(experiment.get("prediction")),
            "measure": _string_or_empty(experiment.get("measure")),
            "message_index": _optional_int(experiment.get("message_index")),
        },
    )


def _node_compaction_policies(
    nodes: list[Any],
    *,
    turn_count: int,
    archive_after_turns: int,
) -> dict[str, dict[str, Any]]:
    archive_window = archive_after_turns or 12
    policies: dict[str, dict[str, Any]] = {}
    for item in nodes:
        if isinstance(item, Mapping):
            node_id = _string_or_empty(item.get("id"))
            if node_id:
                policies[node_id] = _node_compaction_policy(
                    item,
                    turn_count=turn_count,
                    archive_after_turns=archive_window,
                )
    return policies


def _edge_compaction_policies(
    edges: list[Any],
    *,
    turn_count: int,
    active_node_ids: set[str],
) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for item in edges:
        if isinstance(item, Mapping):
            edge_id = _string_or_empty(item.get("id"))
            if edge_id:
                policies[edge_id] = _edge_compaction_policy(
                    item,
                    turn_count=turn_count,
                    active_node_ids=active_node_ids,
                )
    return policies


def _node_compaction_policy(
    node: Mapping[str, Any],
    *,
    turn_count: int,
    archive_after_turns: int,
) -> dict[str, Any]:
    status = _string_or_empty(node.get("status")) or "tentative"
    kind = _string_or_empty(node.get("kind"))
    last_seen_turn = _int_or_zero(node.get("last_seen_turn"))
    seen_count = _int_or_zero(node.get("seen_count"))
    age = max(turn_count - last_seen_turn, 0)
    active = status not in {"archived", "rejected", "removed"}
    protected = active and (
        status == "confirmed"
        or kind == "longitudinal_pattern"
        or seen_count >= 2
        or last_seen_turn == turn_count
    )
    stale_singleton = (
        status in {"tentative", "archived"}
        and not protected
        and age >= archive_after_turns
        and seen_count <= 1
    )
    if status == "removed":
        reason = "user_removed"
        retention = "suppress"
    elif status == "rejected":
        reason = "user_rejected"
        retention = "suppress_reintroduction"
    elif status == "archived" and stale_singleton:
        reason = "stale_singleton"
        retention = "archive"
    elif status == "archived":
        reason = "bounded_low_priority"
        retention = "archive"
    elif status == "confirmed":
        reason = "protected_confirmed"
        retention = "keep_active"
    elif kind == "longitudinal_pattern":
        reason = "protected_longitudinal"
        retention = "keep_active"
    elif seen_count >= 2:
        reason = "protected_recurring"
        retention = "keep_active"
    elif last_seen_turn == turn_count:
        reason = "current_turn"
        retention = "keep_active"
    else:
        reason = "active_within_budget"
        retention = "keep_active"
    return {
        "age": age,
        "active_in_map": active,
        "hidden_by_policy": not active,
        "protected_by_policy": protected,
        "stale_singleton": stale_singleton,
        "compaction_reason": reason,
        "retention_action": retention,
        "priority_score": _node_policy_priority(
            kind=kind,
            confidence=_float_or_zero(node.get("confidence")),
            seen_count=seen_count,
            age=age,
            archive_after_turns=archive_after_turns,
        ),
    }


def _edge_compaction_policy(
    edge: Mapping[str, Any],
    *,
    turn_count: int,
    active_node_ids: set[str],
) -> dict[str, Any]:
    status = _string_or_empty(edge.get("status")) or "tentative"
    source = _string_or_empty(edge.get("source"))
    target = _string_or_empty(edge.get("target"))
    last_seen_turn = _int_or_zero(edge.get("last_seen_turn"))
    age = max(turn_count - last_seen_turn, 0)
    endpoints_active = source in active_node_ids and target in active_node_ids
    active = status not in {"archived", "rejected", "removed"} and endpoints_active
    if status == "removed":
        reason = "removed"
        retention = "suppress"
    elif status == "rejected":
        reason = "rejected"
        retention = "suppress_reintroduction"
    elif not endpoints_active:
        reason = "hidden_endpoint"
        retention = "archive"
    elif status == "archived":
        reason = "bounded_low_priority"
        retention = "archive"
    else:
        reason = "active"
        retention = "keep_active"
    return {
        "age": age,
        "active_in_map": active,
        "hidden_by_policy": not active,
        "compaction_reason": reason,
        "retention_action": retention,
        "priority_score": _edge_policy_priority(
            kind=_string_or_empty(edge.get("kind")),
            confidence=_float_or_zero(edge.get("confidence")),
            seen_count=_int_or_zero(edge.get("seen_count")),
            age=age,
        ),
    }


def _workspace_compaction_policy(
    *,
    nodes: list[Any],
    edges: list[Any],
    node_policies: Mapping[str, Mapping[str, Any]],
    edge_policies: Mapping[str, Mapping[str, Any]],
    turn_count: int,
    policy_config: Mapping[str, int],
) -> dict[str, Any]:
    node_statuses = [
        _string_or_empty(node.get("status")) or "tentative"
        for node in nodes
        if isinstance(node, Mapping)
    ]
    edge_statuses = [
        _string_or_empty(edge.get("status")) or "tentative"
        for edge in edges
        if isinstance(edge, Mapping)
    ]
    node_reasons = [
        _string_or_empty(policy.get("compaction_reason"))
        for policy in node_policies.values()
    ]
    return {
        "policy_version": "db-compaction-v1",
        "policy_source": "projected_from_case_memory",
        "turn_count": turn_count,
        "active_node_limit": _int_or_zero(policy_config.get("active_node_limit")),
        "active_edge_limit": _int_or_zero(policy_config.get("active_edge_limit")),
        "archive_after_turns": _int_or_zero(policy_config.get("archive_after_turns")),
        "node_count": len(node_statuses),
        "edge_count": len(edge_statuses),
        "active_node_count": sum(
            1 for policy in node_policies.values() if policy.get("active_in_map")
        ),
        "hidden_node_count": sum(
            1 for policy in node_policies.values() if policy.get("hidden_by_policy")
        ),
        "archived_node_count": node_statuses.count("archived"),
        "rejected_node_count": node_statuses.count("rejected"),
        "removed_node_count": node_statuses.count("removed"),
        "active_edge_count": sum(
            1 for policy in edge_policies.values() if policy.get("active_in_map")
        ),
        "hidden_edge_count": sum(
            1 for policy in edge_policies.values() if policy.get("hidden_by_policy")
        ),
        "stale_singleton_count": node_reasons.count("stale_singleton"),
        "bounded_low_priority_count": node_reasons.count("bounded_low_priority"),
        "protected_node_count": sum(
            1 for policy in node_policies.values() if policy.get("protected_by_policy")
        ),
        "suppressed_node_count": node_reasons.count("user_rejected")
        + node_reasons.count("user_removed"),
    }


def _node_policy_priority(
    *,
    kind: str,
    confidence: float,
    seen_count: int,
    age: int,
    archive_after_turns: int,
) -> float:
    recency = max(archive_after_turns - age, 0)
    return round(
        _map_kind_priority(kind)
        + 8.0 * confidence
        + 4.0 * min(seen_count, 4)
        + 0.4 * recency,
        3,
    )


def _edge_policy_priority(
    *,
    kind: str,
    confidence: float,
    seen_count: int,
    age: int,
) -> float:
    recency = max(12 - age, 0)
    return round(
        _map_edge_kind_priority(kind)
        + 7.0 * confidence
        + 3.0 * min(seen_count, 4)
        + 0.35 * recency,
        3,
    )


def _map_kind_priority(kind: str) -> float:
    return {
        "objective": 50.0,
        "project": 48.0,
        "next_action": 46.0,
        "obstacle": 44.0,
        "belief": 42.0,
        "hypothesis": 40.0,
        "challenge": 39.0,
        "behavior": 38.0,
        "concern": 37.0,
        "emotion": 36.0,
        "value": 35.0,
        "goal": 35.0,
        "success_measure": 34.0,
        "implementation_intention": 33.0,
        "thought": 32.0,
        "time_horizon": 31.5,
        "stake": 31.0,
        "waiting_for": 30.5,
        "intervention": 30.0,
        "urge": 28.0,
        "domain": 26.0,
        "feature": 24.0,
        "situation": 20.0,
        "consequence": 18.0,
    }.get(kind, 16.0)


def _map_edge_kind_priority(kind: str) -> float:
    return {
        "supports_longitudinal_pattern": 50.0,
        "supports_intervention": 42.0,
        "supports_hypothesis": 40.0,
        "blocks_or_complicates": 38.0,
        "aims_at": 36.0,
        "advances_project": 36.0,
        "advances_task": 36.0,
        "measures_objective": 35.5,
        "raises_stakes_for": 35.0,
        "may_block": 34.0,
        "plans_for": 33.0,
        "context_for": 32.0,
        "may_trigger": 30.0,
        "serves_direction": 28.0,
        "can_lead_to": 28.0,
        "shows_up_as": 26.0,
        "orients": 25.0,
        "about": 25.0,
        "has_feature": 24.0,
        "serves_goal": 24.0,
        "refines_goal": 23.0,
        "leads_to": 22.0,
        "involves_task": 22.0,
        "involves_project": 22.0,
        "waiting_for": 22.0,
        "has_horizon": 21.0,
        "domain_of": 20.0,
    }.get(kind, 18.0)


def _projection_record_id(table: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{table}:{digest}"


def _workspace_key(user_id: str, workspace_id: str) -> str:
    return f"{user_id}/{workspace_id}"


def _conversation_key(user_id: str, workspace_id: str, conversation_id: str) -> str:
    return f"{user_id}/{workspace_id}/{conversation_id}"


def _last_message_text(transcript: list[Any]) -> str | None:
    for item in reversed(transcript):
        if isinstance(item, Mapping):
            text = _string_or_empty(item.get("text"))
            if text:
                return text
    return None


def _nested_string(data: Mapping[str, Any], *keys: str) -> str | None:
    item: Any = data
    for key in keys:
        if not isinstance(item, Mapping):
            return None
        item = item.get(key)
    value = _string_or_empty(item)
    return value or None


def _string_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _int_or_zero(value: Any) -> int:
    return _optional_int(value) or 0


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_snapshot(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible deep copy of a snapshot mapping."""

    snapshot = json.loads(json.dumps(data))
    return snapshot if isinstance(snapshot, dict) else {}
