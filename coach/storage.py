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

    await db.upsert(
        _projection_record_id("coach_workspace", app_record, workspace_key),
        {
            "app_record": app_record,
            "workspace_key": workspace_key,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "activity_counter": _int_or_zero(workspace.get("activity_counter")),
            "turn_count": _int_or_zero(memory.get("turn_count") if isinstance(memory, Mapping) else None),
            "conversation_count": len(conversations),
            "working_node_count": len(nodes),
            "working_edge_count": len(edges),
            "experiment_count": len(experiments),
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
