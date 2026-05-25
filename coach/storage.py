"""Persistence backends for workspace and working-map state.

The chat app keeps rich Python objects in memory while a request is active.
Storage backends persist a serialized snapshot of that state. This keeps the
therapeutic kernel independent from the database choice and lets us add a
graph-native backend incrementally.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
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


def _json_snapshot(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible deep copy of a snapshot mapping."""

    snapshot = json.loads(json.dumps(data))
    return snapshot if isinstance(snapshot, dict) else {}
