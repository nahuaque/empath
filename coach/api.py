"""Starlette API and SSE chat app for the coaching kernel."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Literal

import anyio
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai.messages import ModelMessage
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from .chat import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_MODEL,
    ChatTurnResult,
    DeterministicKernelGuidedCoach,
    KernelGuidedCoach,
    PreparedTurn,
    build_response_prompt,
    build_turn_trace,
    read_api_key,
)
from .experiments import (
    CoachingExperiment,
    ExperimentFeedbackAction,
    ExperimentFeedbackResult,
    ExperimentStore,
)
from .formulation import (
    CaseMemory,
    FeedbackAction,
    FormulationDelta,
    FormulationFeedbackResult,
    FormulationGraph,
    FormulationMirror,
)
from .policy import PolicyMemory
from .storage import JsonFileStateBackend, StateBackend, SurrealStateBackend
from .therapeutic_kernel import TherapeuticReasoningKernel


CoachFactory = Callable[[], Any]
DEFAULT_USER_ID = "default"
DEFAULT_WORKSPACE_ID = "default"
DEFAULT_STATE_FILE = ".coach_chat_state.json"
DEFAULT_SURREAL_FILE = ".coach_surreal.db"


class ChatRequest(BaseModel):
    """Request body for the non-streaming chat endpoint."""

    message: str = Field(min_length=1, max_length=6000)
    user_id: str = Field(default=DEFAULT_USER_ID, min_length=1, max_length=128)
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    trace: bool = False


class ChatMessage(BaseModel):
    """One visible transcript message."""

    index: int
    role: Literal["user", "assistant", "reflection", "info"]
    text: str
    trace_available: bool = False
    explanation: str | None = None
    experiment: CoachingExperiment | None = None


class ConversationSummary(BaseModel):
    """One conversation in a workspace."""

    conversation_id: str
    session_id: str
    title: str
    message_count: int = 0
    last_message: str | None = None
    active: bool = False


class ChatSessionResponse(BaseModel):
    """Visible transcript state for one chat session."""

    session_id: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID
    conversation_id: str
    conversations: tuple[ConversationSummary, ...] = Field(default_factory=tuple)
    messages: tuple[ChatMessage, ...]
    formulation: FormulationGraph = Field(default_factory=FormulationGraph)
    experiments: tuple[CoachingExperiment, ...] = Field(default_factory=tuple)
    policy: dict[str, Any] = Field(default_factory=dict)


class TraceExplanationResponse(BaseModel):
    """Lazy human-readable explanation for one assistant turn."""

    session_id: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID
    conversation_id: str
    message_index: int
    explanation: str


class ChatResponse(BaseModel):
    """JSON response for a completed chat turn."""

    session_id: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID
    conversation_id: str
    response: str
    conversations: tuple[ConversationSummary, ...] = Field(default_factory=tuple)
    messages: tuple[ChatMessage, ...] = Field(default_factory=tuple)
    trace: dict[str, Any] | None = None
    formulation_delta: FormulationDelta | None = None
    formulation: FormulationGraph = Field(default_factory=FormulationGraph)
    experiment: CoachingExperiment | None = None
    experiments: tuple[CoachingExperiment, ...] = Field(default_factory=tuple)
    policy: dict[str, Any] = Field(default_factory=dict)


class FormulationFeedbackRequest(BaseModel):
    """User correction for a working-map node."""

    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    user_id: str = Field(default=DEFAULT_USER_ID, min_length=1, max_length=128)
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    node_id: str = Field(min_length=1, max_length=128)
    action: FeedbackAction
    note: str | None = Field(default=None, max_length=1000)


class FormulationFeedbackResponse(BaseModel):
    """API response after user correction."""

    session_id: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID
    conversation_id: str | None = None
    result: FormulationFeedbackResult
    policy: dict[str, Any] = Field(default_factory=dict)


class FormulationMirrorResponse(BaseModel):
    """Reflective playback of the working formulation."""

    session_id: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID
    conversation_id: str
    mirror: FormulationMirror
    message: ChatMessage | None = None


class ExperimentFeedbackRequest(BaseModel):
    """Outcome feedback for one proposed coaching experiment."""

    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    user_id: str = Field(default=DEFAULT_USER_ID, min_length=1, max_length=128)
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    experiment_id: str = Field(min_length=1, max_length=128)
    action: ExperimentFeedbackAction
    note: str | None = Field(default=None, max_length=1000)
    friction_before: int | None = Field(default=None, ge=0, le=10)
    friction_after: int | None = Field(default=None, ge=0, le=10)


class ExperimentFeedbackResponse(BaseModel):
    """API response after closing the loop on one experiment."""

    session_id: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID
    conversation_id: str | None = None
    result: ExperimentFeedbackResult
    policy: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ChatConversation:
    """Conversation-scoped chat state."""

    coach: Any
    created_order: int = 0
    updated_order: int = 0
    history: list[ModelMessage] | None = None
    transcript: list[ChatMessage] = field(default_factory=list)
    turn_traces: dict[int, dict[str, Any]] = field(default_factory=dict)
    explanations: dict[int, str] = field(default_factory=dict)


@dataclass
class ChatWorkspace:
    """Workspace-scoped memory shared by its conversations."""

    memory: CaseMemory = field(default_factory=CaseMemory)
    experiments: ExperimentStore = field(default_factory=ExperimentStore)
    policy: PolicyMemory = field(default_factory=PolicyMemory)
    conversations: dict[str, ChatConversation] = field(default_factory=dict)
    activity_counter: int = 0
    lock: asyncio.Lock | None = None

    def __post_init__(self) -> None:
        if self.lock is None:
            self.lock = asyncio.Lock()


@dataclass(frozen=True)
class ChatScope:
    """Resolved user/workspace/conversation objects for one request."""

    user_id: str
    workspace_id: str
    conversation_id: str
    workspace: ChatWorkspace
    conversation: ChatConversation

    @property
    def session_id(self) -> str:
        """Backward-compatible alias for the active conversation id."""

        return self.conversation_id


@dataclass(frozen=True)
class WorkspaceScope:
    """Resolved user/workspace object for workspace-only requests."""

    user_id: str
    workspace_id: str
    workspace: ChatWorkspace


class ChatSessionStore:
    """In-memory workspace/conversation store with pluggable snapshot persistence."""

    def __init__(
        self,
        coach_factory: CoachFactory,
        *,
        state_file: str | Path | None = None,
        state_backend: StateBackend | None = None,
    ) -> None:
        self._coach_factory = coach_factory
        self._users: dict[str, dict[str, ChatWorkspace]] = {}
        if state_backend is not None and state_file is not None:
            raise ValueError("Pass either state_backend or state_file, not both.")
        self._state_backend = state_backend or (
            JsonFileStateBackend(state_file) if state_file else None
        )
        self._lock = asyncio.Lock()
        self._loaded = False

    @property
    def backend_description(self) -> str:
        if self._state_backend is None:
            return "memory"
        return self._state_backend.description

    async def get_workspace(
        self,
        *,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> WorkspaceScope:
        resolved_user_id = _resolve_scope_id(user_id, DEFAULT_USER_ID)
        resolved_workspace_id = _resolve_scope_id(workspace_id, DEFAULT_WORKSPACE_ID)
        async with self._lock:
            await self._ensure_loaded_locked()
            workspaces = self._users.setdefault(resolved_user_id, {})
            workspace = workspaces.get(resolved_workspace_id)
            if workspace is None:
                workspace = ChatWorkspace()
                workspaces[resolved_workspace_id] = workspace
            return WorkspaceScope(
                user_id=resolved_user_id,
                workspace_id=resolved_workspace_id,
                workspace=workspace,
            )

    async def get_conversation(
        self,
        *,
        user_id: str | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        session_id: str | None = None,
        prefer_existing: bool = False,
    ) -> ChatScope:
        resolved_user_id = _resolve_scope_id(user_id, DEFAULT_USER_ID)
        resolved_workspace_id = _resolve_scope_id(workspace_id, DEFAULT_WORKSPACE_ID)
        requested_conversation_id = _resolve_scope_id(conversation_id or session_id, "")
        async with self._lock:
            await self._ensure_loaded_locked()
            workspaces = self._users.setdefault(resolved_user_id, {})
            workspace = workspaces.get(resolved_workspace_id)
            if workspace is None:
                workspace = ChatWorkspace()
                workspaces[resolved_workspace_id] = workspace
            resolved_conversation_id = requested_conversation_id
            if prefer_existing:
                latest_non_empty = _latest_conversation_id(
                    workspace,
                    require_messages=True,
                )
                latest_existing = _latest_conversation_id(workspace)
                current = workspace.conversations.get(resolved_conversation_id)
                if (
                    resolved_conversation_id
                    and current is None
                    and latest_non_empty
                ):
                    resolved_conversation_id = latest_non_empty
                elif resolved_conversation_id and current is None and latest_existing:
                    resolved_conversation_id = latest_existing
                elif resolved_conversation_id and current is None:
                    resolved_conversation_id = ""
                elif current is not None and not current.transcript and latest_non_empty:
                    resolved_conversation_id = latest_non_empty
            if not resolved_conversation_id:
                resolved_conversation_id = (
                    _latest_conversation_id(workspace, require_messages=True)
                    or _latest_conversation_id(workspace)
                    or secrets.token_urlsafe(16)
                )
            conversation = workspace.conversations.get(resolved_conversation_id)
            if conversation is None:
                workspace.activity_counter += 1
                conversation = ChatConversation(
                    coach=self._coach_factory(),
                    created_order=workspace.activity_counter,
                    updated_order=workspace.activity_counter,
                )
                workspace.conversations[resolved_conversation_id] = conversation
            return ChatScope(
                user_id=resolved_user_id,
                workspace_id=resolved_workspace_id,
                conversation_id=resolved_conversation_id,
                workspace=workspace,
                conversation=conversation,
            )

    async def save(self) -> None:
        """Persist the visible app state when a backend is configured."""

        if self._state_backend is None:
            return
        async with self._lock:
            await self._ensure_loaded_locked()
            data = self._dump_state()
        await self._state_backend.save(data)

    async def explanation_evidence(
        self,
        scope: ChatScope,
        *,
        message_index: int,
        trace: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fetch graph-backed explanation evidence when the backend supports it."""

        provider = getattr(self._state_backend, "explanation_evidence", None)
        if provider is None:
            return None
        return await provider(
            user_id=scope.user_id,
            workspace_id=scope.workspace_id,
            conversation_id=scope.conversation_id,
            message_index=message_index,
            trace=trace,
        )

    async def compaction_summary(
        self,
        scope: WorkspaceScope,
    ) -> dict[str, Any] | None:
        """Fetch projected database compaction policy when the backend supports it."""

        provider = getattr(self._state_backend, "compaction_summary", None)
        if provider is None:
            return None
        return await provider(
            user_id=scope.user_id,
            workspace_id=scope.workspace_id,
        )

    async def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._state_backend is None:
            return
        data = await self._state_backend.load()
        if not isinstance(data, dict):
            return
        self._users = self._restore_users(data)

    def _restore_users(self, data: dict[str, Any]) -> dict[str, dict[str, ChatWorkspace]]:
        users = data.get("users")
        if not isinstance(users, dict):
            return {}
        loaded_users: dict[str, dict[str, ChatWorkspace]] = {}
        for user_id, workspaces_data in users.items():
            if not isinstance(workspaces_data, dict):
                continue
            loaded_workspaces: dict[str, ChatWorkspace] = {}
            for workspace_id, workspace_data in workspaces_data.items():
                if not isinstance(workspace_data, dict):
                    continue
                workspace = self._restore_workspace(workspace_data)
                loaded_workspaces[str(workspace_id)] = workspace
            if loaded_workspaces:
                loaded_users[str(user_id)] = loaded_workspaces
        return loaded_users

    def _restore_workspace(self, data: dict[str, Any]) -> ChatWorkspace:
        workspace = ChatWorkspace()
        workspace.activity_counter = int(data.get("activity_counter") or 0)
        if memory_data := data.get("memory"):
            workspace.memory.import_state(memory_data)
        workspace.experiments.import_state(data.get("experiments") or ())
        workspace.policy.import_state(data.get("policy") or {})
        conversations = data.get("conversations") or {}
        if isinstance(conversations, dict):
            for conversation_id, conversation_data in conversations.items():
                if not isinstance(conversation_data, dict):
                    continue
                workspace.conversations[str(conversation_id)] = self._restore_conversation(
                    conversation_data
                )
        return workspace

    def _restore_conversation(self, data: dict[str, Any]) -> ChatConversation:
        transcript = []
        for item in data.get("transcript") or ():
            try:
                transcript.append(ChatMessage.model_validate(item))
            except ValidationError:
                continue
        turn_traces = {}
        for key, value in (data.get("turn_traces") or {}).items():
            try:
                turn_traces[int(key)] = value
            except (TypeError, ValueError):
                continue
        explanations = {}
        for key, value in (data.get("explanations") or {}).items():
            try:
                explanations[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return ChatConversation(
            coach=self._coach_factory(),
            created_order=int(data.get("created_order") or 0),
            updated_order=int(data.get("updated_order") or 0),
            transcript=transcript,
            turn_traces=turn_traces,
            explanations=explanations,
        )

    def _dump_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "users": {
                user_id: {
                    workspace_id: self._dump_workspace(workspace)
                    for workspace_id, workspace in workspaces.items()
                }
                for user_id, workspaces in self._users.items()
            },
        }

    def _dump_workspace(self, workspace: ChatWorkspace) -> dict[str, Any]:
        return {
            "activity_counter": workspace.activity_counter,
            "memory": workspace.memory.export_state(),
            "experiments": workspace.experiments.export_state(),
            "policy": workspace.policy.export_state(),
            "conversations": {
                conversation_id: self._dump_conversation(conversation)
                for conversation_id, conversation in workspace.conversations.items()
            },
        }

    def _dump_conversation(self, conversation: ChatConversation) -> dict[str, Any]:
        return {
            "created_order": conversation.created_order,
            "updated_order": conversation.updated_order,
            "transcript": [
                item.model_dump(exclude_none=True)
                for item in conversation.transcript
            ],
            "turn_traces": {
                str(index): trace
                for index, trace in conversation.turn_traces.items()
            },
            "explanations": {
                str(index): explanation
                for index, explanation in conversation.explanations.items()
            },
        }


def _state_backend_from_config(
    *,
    store_backend: Literal["memory", "json", "surreal"] | None,
    state_file: str | Path | None,
    surreal_url: str | None,
    surreal_namespace: str,
    surreal_database: str,
    surreal_record_id: str,
    surreal_user: str | None,
    surreal_password: str | None,
) -> StateBackend | None:
    backend = (
        store_backend
        or os.environ.get("COACH_STORE_BACKEND")
        or ("json" if state_file else "surreal")
    ).strip().casefold()
    if backend == "memory":
        return None
    if backend == "json":
        if not state_file:
            state_file = os.environ.get("COACH_STATE_FILE") or DEFAULT_STATE_FILE
        return JsonFileStateBackend(state_file)
    if backend == "surreal":
        url = surreal_url or os.environ.get("COACH_SURREAL_URL") or _default_surreal_url()
        return SurrealStateBackend(
            url=url,
            namespace=os.environ.get("COACH_SURREAL_NAMESPACE", surreal_namespace),
            database=os.environ.get("COACH_SURREAL_DATABASE", surreal_database),
            record_id=os.environ.get("COACH_SURREAL_RECORD_ID", surreal_record_id),
            username=surreal_user or os.environ.get("COACH_SURREAL_USER"),
            password=surreal_password or os.environ.get("COACH_SURREAL_PASSWORD"),
        )
    raise ValueError(f"Unsupported store backend: {store_backend!r}")


def _default_surreal_url() -> str:
    return Path(DEFAULT_SURREAL_FILE).resolve().as_uri()


def create_app(
    *,
    coach_factory: CoachFactory | None = None,
    api_key_file: str | Path = DEFAULT_API_KEY_FILE,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    max_tokens: int = 900,
    dry_run: bool = False,
    state_file: str | Path | None = None,
    state_backend: StateBackend | None = None,
    store_backend: Literal["memory", "json", "surreal"] | None = None,
    surreal_url: str | None = None,
    surreal_namespace: str = "coach",
    surreal_database: str = "coach",
    surreal_record_id: str = "app_state:default",
    surreal_user: str | None = None,
    surreal_password: str | None = None,
) -> Starlette:
    """Create the ASGI app."""

    if coach_factory is None:
        coach_factory = _default_coach_factory(
            api_key_file=api_key_file,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            dry_run=dry_run,
        )

    if state_backend is None:
        state_backend = _state_backend_from_config(
            store_backend=store_backend,
            state_file=state_file,
            surreal_url=surreal_url,
            surreal_namespace=surreal_namespace,
            surreal_database=surreal_database,
            surreal_record_id=surreal_record_id,
            surreal_user=surreal_user,
            surreal_password=surreal_password,
        )

    store = ChatSessionStore(coach_factory, state_backend=state_backend)

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "model": model_name,
                "dry_run": dry_run,
                "transport": "sse",
                "state_backend": store.backend_description,
            }
        )

    async def index(_request: Request) -> HTMLResponse:
        return HTMLResponse(CHAT_APP_HTML)

    async def chat_json(request: Request) -> JSONResponse:
        try:
            payload = ChatRequest.model_validate(await request.json())
        except ValidationError as exc:
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        except json.JSONDecodeError:
            return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

        scope = await store.get_conversation(
            user_id=payload.user_id,
            workspace_id=payload.workspace_id,
            conversation_id=payload.conversation_id,
            session_id=payload.session_id,
        )
        if _is_framework_explanation_request(payload.message):
            message = await _run_info_turn(scope, payload.message)
            await store.save()
            return JSONResponse(
                _info_response_payload(
                    scope,
                    message,
                )
            )

        try:
            turn, formulation_delta, experiment = await _run_chat_turn(
                scope,
                payload.message,
            )
        except Exception as exc:  # pragma: no cover - integration/runtime boundary
            return JSONResponse({"detail": str(exc)}, status_code=500)

        await store.save()
        return JSONResponse(
            _chat_response_payload(
                scope,
                turn,
                include_trace=payload.trace,
                formulation_delta=formulation_delta,
                experiment=experiment,
            )
        )

    async def chat_session(request: Request) -> JSONResponse:
        try:
            params = _query_conversation_scope(request)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        scope = await store.get_conversation(**params)
        await store.save()
        return JSONResponse(_session_payload(scope))

    async def formulation(request: Request) -> JSONResponse:
        try:
            params = _query_workspace_scope(request)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        workspace_scope = await store.get_workspace(**params)
        async with workspace_scope.workspace.lock:  # type: ignore[arg-type]
            graph = workspace_scope.workspace.memory.snapshot()
        return JSONResponse(graph.model_dump())

    async def formulation_compaction(request: Request) -> JSONResponse:
        try:
            params = _query_workspace_scope(request)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        workspace_scope = await store.get_workspace(**params)
        await store.save()
        summary = await store.compaction_summary(workspace_scope)
        if summary is None:
            async with workspace_scope.workspace.lock:  # type: ignore[arg-type]
                summary = _fallback_compaction_summary(workspace_scope)
        return JSONResponse(summary)

    async def formulation_feedback(request: Request) -> JSONResponse:
        try:
            payload = FormulationFeedbackRequest.model_validate(await request.json())
        except ValidationError as exc:
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        except json.JSONDecodeError:
            return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

        workspace_scope = await store.get_workspace(
            user_id=payload.user_id,
            workspace_id=payload.workspace_id,
        )
        async with workspace_scope.workspace.lock:  # type: ignore[arg-type]
            try:
                result = workspace_scope.workspace.memory.apply_feedback(
                    payload.node_id,
                    payload.action,
                    note=payload.note,
                )
                workspace_scope.workspace.policy.record_formulation(
                    result.node,
                    payload.action,
                )
            except KeyError:
                return JSONResponse({"detail": "Formulation node not found."}, status_code=404)
        await store.save()
        return JSONResponse(
            FormulationFeedbackResponse(
                session_id=payload.conversation_id or payload.session_id or "",
                user_id=workspace_scope.user_id,
                workspace_id=workspace_scope.workspace_id,
                conversation_id=payload.conversation_id or payload.session_id,
                result=result,
                policy=workspace_scope.workspace.policy.summary(),
            ).model_dump()
        )

    async def formulation_mirror(request: Request) -> JSONResponse:
        try:
            params = _query_conversation_scope(request)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)

        scope = await store.get_conversation(**params)
        async with scope.workspace.lock:  # type: ignore[arg-type]
            graph = scope.workspace.memory.snapshot()
        try:
            mirror = await anyio.to_thread.run_sync(
                lambda: scope.conversation.coach.mirror_formulation(graph)
            )
        except Exception as exc:  # pragma: no cover - integration/runtime boundary
            return JSONResponse({"detail": str(exc)}, status_code=500)

        async with scope.workspace.lock:  # type: ignore[arg-type]
            message = _append_transcript_message(scope.conversation, "reflection", mirror.text)
            _touch_conversation(scope)

        await store.save()
        return JSONResponse(
            FormulationMirrorResponse(
                session_id=scope.session_id,
                user_id=scope.user_id,
                workspace_id=scope.workspace_id,
                conversation_id=scope.conversation_id,
                mirror=mirror,
                message=message,
            ).model_dump(exclude_none=True)
        )

    async def experiments(request: Request) -> JSONResponse:
        try:
            params = _query_workspace_scope(request)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        workspace_scope = await store.get_workspace(**params)
        async with workspace_scope.workspace.lock:  # type: ignore[arg-type]
            items = workspace_scope.workspace.experiments.snapshot()
        return JSONResponse(
            {
                "session_id": request.query_params.get("conversation_id")
                or request.query_params.get("session_id")
                or "",
                "user_id": workspace_scope.user_id,
                "workspace_id": workspace_scope.workspace_id,
                "conversation_id": request.query_params.get("conversation_id")
                or request.query_params.get("session_id"),
                "experiments": [item.model_dump(exclude_none=True) for item in items],
            }
        )

    async def experiment_feedback(request: Request) -> JSONResponse:
        try:
            payload = ExperimentFeedbackRequest.model_validate(await request.json())
        except ValidationError as exc:
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        except json.JSONDecodeError:
            return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

        workspace_scope = await store.get_workspace(
            user_id=payload.user_id,
            workspace_id=payload.workspace_id,
        )
        async with workspace_scope.workspace.lock:  # type: ignore[arg-type]
            try:
                result = workspace_scope.workspace.experiments.apply_feedback(
                    payload.experiment_id,
                    payload.action,
                    note=payload.note,
                    friction_before=payload.friction_before,
                    friction_after=payload.friction_after,
                )
                workspace_scope.workspace.policy.record_experiment(result.experiment)
            except KeyError:
                return JSONResponse({"detail": "Experiment not found."}, status_code=404)
        await store.save()
        return JSONResponse(
            ExperimentFeedbackResponse(
                session_id=payload.conversation_id or payload.session_id or "",
                user_id=workspace_scope.user_id,
                workspace_id=workspace_scope.workspace_id,
                conversation_id=payload.conversation_id or payload.session_id,
                result=result,
                policy=workspace_scope.workspace.policy.summary(),
            ).model_dump(exclude_none=True)
        )

    async def explain_trace(request: Request) -> JSONResponse:
        try:
            params = _query_conversation_scope(request)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        raw_index = request.query_params.get("message_index")
        try:
            message_index = int(raw_index or "")
        except ValueError:
            return JSONResponse({"detail": "Invalid message_index."}, status_code=422)

        scope = await store.get_conversation(**params)
        async with scope.workspace.lock:  # type: ignore[arg-type]
            message = _transcript_message(scope.conversation, message_index)
            if message is None:
                return JSONResponse({"detail": "Message not found."}, status_code=404)
            if message.role != "assistant":
                return JSONResponse(
                    {"detail": "Trace explanations are available for assistant messages."},
                    status_code=422,
                )
            trace = scope.conversation.turn_traces.get(message_index)
            if trace is None:
                return JSONResponse({"detail": "No trace stored for this message."}, status_code=404)
            explanation = scope.conversation.explanations.get(message_index)
            if explanation is None:
                evidence = await store.explanation_evidence(
                    scope,
                    message_index=message_index,
                    trace=trace,
                )
                explanation = explain_trace_human_readable(trace, evidence=evidence)
                scope.conversation.explanations[message_index] = explanation
                message.explanation = explanation

        await store.save()
        return JSONResponse(
            TraceExplanationResponse(
                session_id=scope.session_id,
                user_id=scope.user_id,
                workspace_id=scope.workspace_id,
                conversation_id=scope.conversation_id,
                message_index=message_index,
                explanation=explanation,
            ).model_dump()
        )

    async def chat_stream(request: Request) -> Response:
        message = (request.query_params.get("message") or "").strip()
        if not message:
            return JSONResponse({"detail": "Missing message query parameter."}, status_code=400)
        if len(message) > 6000:
            return JSONResponse({"detail": "Message is too long."}, status_code=422)

        try:
            scope_params = _query_conversation_scope(request)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        scope = await store.get_conversation(**scope_params)
        include_trace = _truthy(request.query_params.get("trace"))

        async def events():
            yield _sse(
                "session",
                {
                    "session_id": scope.session_id,
                    "user_id": scope.user_id,
                    "workspace_id": scope.workspace_id,
                    "conversation_id": scope.conversation_id,
                },
            )
            async with scope.workspace.lock:  # type: ignore[arg-type]
                try:
                    user_record = _append_transcript_message(scope.conversation, "user", message)
                    _touch_conversation(scope)
                    yield _sse("message", user_record.model_dump(exclude_none=True))
                    if _is_framework_explanation_request(message):
                        info_record = _append_transcript_message(
                            scope.conversation,
                            "info",
                            framework_explanation_response(message),
                        )
                        _touch_conversation(scope)
                        yield _sse(
                            "response",
                            {
                                "text": info_record.text,
                                "message": info_record.model_dump(exclude_none=True),
                                "policy": scope.workspace.policy.summary(),
                                "conversations": [
                                    item.model_dump()
                                    for item in _conversation_summaries(scope)
                                ],
                            },
                        )
                        yield _sse(
                            "done",
                            {
                                "ok": True,
                                "message_count": len(scope.conversation.transcript),
                            },
                        )
                        await store.save()
                        return
                    yield _sse("status", {"stage": "structured_extraction"})
                    recent_interventions = scope.workspace.memory.recent_interventions()
                    longitudinal_context = scope.workspace.memory.longitudinal_context()
                    local_context = _local_conversation_context(
                        scope.conversation.transcript
                    )
                    prepared = await anyio.to_thread.run_sync(
                        lambda: scope.conversation.coach.prepare_turn(
                            message,
                            recent_interventions=recent_interventions,
                            local_context=local_context,
                            longitudinal_context=longitudinal_context,
                        )
                    )
                    prepared = _apply_policy_to_prepared(
                        prepared,
                        scope.workspace.policy,
                    )
                    yield _sse(
                        "extraction",
                        _drop_empty(prepared.extraction.model_dump(exclude_none=True)),
                    )
                    yield _sse("status", {"stage": "therapeutic_kernel"})
                    yield _sse("kernel", prepared.kernel_snapshot)
                    if prepared.policy_report:
                        yield _sse("policy", prepared.policy_report)
                    yield _sse("status", {"stage": "response_plan"})
                    turn = await anyio.to_thread.run_sync(
                        lambda: scope.conversation.coach.complete_prepared_turn(
                            prepared,
                            message_history=None,
                        )
                    )
                    scope.conversation.history = None
                    trace = build_turn_trace(turn)
                    trace["scope"] = _trace_scope(scope)
                    assistant_record = _append_transcript_message(
                        scope.conversation,
                        "assistant",
                        turn.text,
                        trace_available=True,
                    )
                    _touch_conversation(scope)
                    scope.conversation.turn_traces[assistant_record.index] = trace
                    formulation_delta = scope.workspace.memory.apply_turn(
                        extraction=prepared.extraction,
                        kernel_snapshot=prepared.kernel_snapshot,
                        response_plan=turn.response_plan,
                        message_index=assistant_record.index,
                    )
                    if formulation_delta.longitudinal_patterns:
                        trace["longitudinal"] = [
                            item.model_dump()
                            for item in formulation_delta.longitudinal_patterns
                        ]
                        scope.conversation.turn_traces[assistant_record.index] = trace
                    experiment = scope.workspace.experiments.propose(
                        turn=turn,
                        formulation_delta=formulation_delta,
                        message_index=assistant_record.index,
                    )
                    assistant_record.experiment = experiment
                    yield _sse("plan", turn.response_plan.model_dump(exclude_none=True))
                    yield _sse("formulation", formulation_delta.model_dump(exclude_none=True))
                    yield _sse("experiment", experiment.model_dump(exclude_none=True))
                    yield _sse(
                        "response",
                        {
                            "text": turn.text,
                            "message": assistant_record.model_dump(exclude_none=True),
                            "experiment": experiment.model_dump(exclude_none=True),
                            "policy": scope.workspace.policy.summary(),
                            "conversations": [
                                item.model_dump()
                                for item in _conversation_summaries(scope)
                            ],
                        },
                    )
                    if include_trace:
                        yield _sse("trace", trace)
                    yield _sse(
                        "done",
                            {
                                "ok": True,
                                "message_count": len(scope.conversation.transcript),
                            },
                        )
                    await store.save()
                except Exception as exc:  # pragma: no cover - integration/runtime boundary
                    await store.save()
                    yield _sse("error", {"detail": str(exc)})
                    yield _sse("done", {"ok": False})

        return EventSourceResponse(events())

    return Starlette(
        debug=False,
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/api/health", health, methods=["GET"]),
            Route("/api/chat/session", chat_session, methods=["GET"]),
            Route("/api/chat/explain", explain_trace, methods=["GET"]),
            Route("/api/chat", chat_json, methods=["POST"]),
            Route("/api/chat/stream", chat_stream, methods=["GET"]),
            Route("/api/formulation", formulation, methods=["GET"]),
            Route("/api/formulation/compaction", formulation_compaction, methods=["GET"]),
            Route("/api/formulation/mirror", formulation_mirror, methods=["GET"]),
            Route("/api/formulation/feedback", formulation_feedback, methods=["POST"]),
            Route("/api/experiments", experiments, methods=["GET"]),
            Route("/api/experiments/feedback", experiment_feedback, methods=["POST"]),
        ],
    )


async def _run_chat_turn(
    scope: ChatScope,
    message: str,
) -> tuple[ChatTurnResult, FormulationDelta, CoachingExperiment]:
    async with scope.workspace.lock:  # type: ignore[arg-type]
        _append_transcript_message(scope.conversation, "user", message)
        _touch_conversation(scope)
        recent_interventions = scope.workspace.memory.recent_interventions()
        longitudinal_context = scope.workspace.memory.longitudinal_context()
        local_context = _local_conversation_context(scope.conversation.transcript)
        prepared = await anyio.to_thread.run_sync(
            lambda: scope.conversation.coach.prepare_turn(
                message,
                recent_interventions=recent_interventions,
                local_context=local_context,
                longitudinal_context=longitudinal_context,
            )
        )
        prepared = _apply_policy_to_prepared(prepared, scope.workspace.policy)
        turn = await anyio.to_thread.run_sync(
            lambda: scope.conversation.coach.complete_prepared_turn(
                prepared,
                message_history=None,
            )
        )
        scope.conversation.history = None
        trace = build_turn_trace(turn)
        trace["scope"] = _trace_scope(scope)
        assistant_message = _append_transcript_message(
            scope.conversation,
            "assistant",
            turn.text,
            trace_available=True,
        )
        _touch_conversation(scope)
        scope.conversation.turn_traces[assistant_message.index] = trace
        formulation_delta = scope.workspace.memory.apply_turn(
            extraction=turn.prepared.extraction,
            kernel_snapshot=turn.prepared.kernel_snapshot,
            response_plan=turn.response_plan,
            message_index=assistant_message.index,
        )
        if formulation_delta.longitudinal_patterns:
            trace["longitudinal"] = [
                item.model_dump() for item in formulation_delta.longitudinal_patterns
            ]
            scope.conversation.turn_traces[assistant_message.index] = trace
        experiment = scope.workspace.experiments.propose(
            turn=turn,
            formulation_delta=formulation_delta,
            message_index=assistant_message.index,
        )
        assistant_message.experiment = experiment
        return turn, formulation_delta, experiment


async def _run_info_turn(scope: ChatScope, message: str) -> ChatMessage:
    async with scope.workspace.lock:  # type: ignore[arg-type]
        _append_transcript_message(scope.conversation, "user", message)
        info_message = _append_transcript_message(
            scope.conversation,
            "info",
            framework_explanation_response(message),
        )
        _touch_conversation(scope)
        return info_message


def _chat_response_payload(
    scope: ChatScope,
    turn: ChatTurnResult,
    *,
    include_trace: bool,
    formulation_delta: FormulationDelta,
    experiment: CoachingExperiment,
) -> dict[str, Any]:
    stored_trace = None
    if include_trace and scope.conversation.transcript:
        stored_trace = scope.conversation.turn_traces.get(
            scope.conversation.transcript[-1].index
        )
    payload = ChatResponse(
        session_id=scope.session_id,
        user_id=scope.user_id,
        workspace_id=scope.workspace_id,
        conversation_id=scope.conversation_id,
        response=turn.text,
        conversations=_conversation_summaries(scope),
        messages=tuple(scope.conversation.transcript),
        trace=stored_trace if include_trace else None,
        formulation_delta=formulation_delta,
        formulation=scope.workspace.memory.snapshot(),
        experiment=experiment,
        experiments=scope.workspace.experiments.snapshot(),
        policy=scope.workspace.policy.summary(),
    ).model_dump(exclude_none=True)
    return payload


def _info_response_payload(
    scope: ChatScope,
    message: ChatMessage,
) -> dict[str, Any]:
    payload = ChatResponse(
        session_id=scope.session_id,
        user_id=scope.user_id,
        workspace_id=scope.workspace_id,
        conversation_id=scope.conversation_id,
        response=message.text,
        conversations=_conversation_summaries(scope),
        messages=tuple(scope.conversation.transcript),
        formulation=scope.workspace.memory.snapshot(),
        experiments=scope.workspace.experiments.snapshot(),
        policy=scope.workspace.policy.summary(),
    ).model_dump(exclude_none=True)
    return payload


def _session_payload(scope: ChatScope) -> dict[str, Any]:
    return ChatSessionResponse(
        session_id=scope.session_id,
        user_id=scope.user_id,
        workspace_id=scope.workspace_id,
        conversation_id=scope.conversation_id,
        conversations=_conversation_summaries(scope),
        messages=tuple(scope.conversation.transcript),
        formulation=scope.workspace.memory.snapshot(),
        experiments=scope.workspace.experiments.snapshot(),
        policy=scope.workspace.policy.summary(),
    ).model_dump(exclude_none=True)


def _apply_policy_to_prepared(
    prepared: PreparedTurn,
    policy: PolicyMemory,
) -> PreparedTurn:
    policy_context = policy.prompt_context()
    adjusted_snapshot, policy_report = policy.apply_to_kernel_snapshot(
        prepared.kernel_snapshot
    )
    has_policy_report = (
        bool(policy_report.get("adjustments"))
        or not policy_report.get("summary", {}).get("empty", True)
    )
    return PreparedTurn(
        extraction=prepared.extraction,
        state=prepared.state,
        kernel_snapshot=adjusted_snapshot,
        extraction_prompt=prepared.extraction_prompt,
        local_context=prepared.local_context,
        longitudinal_context=prepared.longitudinal_context,
        policy_context=policy_context,
        policy_report=policy_report if has_policy_report else None,
        response_prompt=build_response_prompt(
            prepared.state.utterance,
            prepared.extraction,
            adjusted_snapshot,
            local_context=prepared.local_context,
            longitudinal_context=prepared.longitudinal_context,
            policy_context=policy_context,
        ),
    )


def _append_transcript_message(
    conversation: ChatConversation,
    role: Literal["user", "assistant", "reflection", "info"],
    text: str,
    *,
    trace_available: bool = False,
) -> ChatMessage:
    message = ChatMessage(
        index=len(conversation.transcript) + 1,
        role=role,
        text=text,
        trace_available=trace_available,
    )
    conversation.transcript.append(message)
    return message


def _transcript_message(conversation: ChatConversation, message_index: int) -> ChatMessage | None:
    for message in conversation.transcript:
        if message.index == message_index:
            return message
    return None


def _touch_conversation(scope: ChatScope) -> None:
    scope.workspace.activity_counter += 1
    scope.conversation.updated_order = scope.workspace.activity_counter


def _local_conversation_context(
    transcript: list[ChatMessage] | tuple[ChatMessage, ...],
    *,
    max_user_turns: int = 5,
    per_message_limit: int = 900,
    total_limit: int = 7000,
) -> str:
    """Return bounded visible context for response planning.

    The response LLM sees this for continuity. The extractor and symbolic kernel
    still operate only on the latest user message.
    """

    if max_user_turns <= 0 or not transcript:
        return ""

    start_index = 0
    user_turns = 0
    for index in range(len(transcript) - 1, -1, -1):
        if transcript[index].role == "user":
            user_turns += 1
            if user_turns >= max_user_turns:
                start_index = index
                break

    role_labels = {
        "user": "user",
        "assistant": "coach",
        "info": "info",
    }
    lines = []
    for message in transcript[start_index:]:
        label = role_labels.get(message.role)
        if label is None:
            continue
        text = _compact_context_text(message.text, limit=per_message_limit)
        if text:
            lines.append(f"{label}: {text}")

    context = "\n".join(lines).strip()
    if len(context) <= total_limit:
        return context
    return context[-total_limit:].lstrip()


def _conversation_summaries(scope: ChatScope) -> tuple[ConversationSummary, ...]:
    summaries = [
        _conversation_summary(
            conversation_id=conversation_id,
            conversation=conversation,
            active=conversation_id == scope.conversation_id,
        )
        for conversation_id, conversation in scope.workspace.conversations.items()
    ]
    summaries.sort(
        key=lambda item: (
            not item.active,
            -scope.workspace.conversations[item.conversation_id].updated_order,
            item.title.casefold(),
            item.conversation_id,
        )
    )
    return tuple(summaries)


def _latest_conversation_id(
    workspace: ChatWorkspace,
    *,
    require_messages: bool = False,
) -> str | None:
    if not workspace.conversations:
        return None
    conversations = [
        item
        for item in workspace.conversations.items()
        if not require_messages or item[1].transcript
    ]
    if not conversations:
        return None
    return max(
        conversations,
        key=lambda item: (
            item[1].updated_order,
            item[1].created_order,
            item[0],
        ),
    )[0]


def _conversation_summary(
    *,
    conversation_id: str,
    conversation: ChatConversation,
    active: bool,
) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=conversation_id,
        session_id=conversation_id,
        title=_conversation_title(conversation_id, conversation),
        message_count=len(conversation.transcript),
        last_message=_conversation_last_message(conversation),
        active=active,
    )


def _conversation_title(conversation_id: str, conversation: ChatConversation) -> str:
    for message in conversation.transcript:
        if message.role == "user":
            return _compact_preview(message.text, limit=44)
    return f"New conversation {conversation_id[:8]}"


def _conversation_last_message(conversation: ChatConversation) -> str | None:
    if not conversation.transcript:
        return None
    return _compact_preview(conversation.transcript[-1].text, limit=80)


def _compact_preview(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)].rstrip()}..."


def _compact_context_text(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)].rstrip()}..."


def _trace_scope(scope: ChatScope) -> dict[str, str]:
    return {
        "user_id": scope.user_id,
        "workspace_id": scope.workspace_id,
        "conversation_id": scope.conversation_id,
        "session_id": scope.session_id,
        "working_map_scope": "workspace",
        "message_history_scope": "bounded_last_5_user_turns",
    }


def _fallback_compaction_summary(scope: WorkspaceScope) -> dict[str, Any]:
    graph = scope.workspace.memory.snapshot(
        include_archived=True,
        include_rejected=True,
        include_removed=True,
    )
    active_nodes = [
        node
        for node in graph.nodes
        if node.status not in {"archived", "rejected", "removed"}
    ]
    hidden_nodes = [
        node
        for node in graph.nodes
        if node.status in {"archived", "rejected", "removed"}
    ]
    return {
        "policy_version": "memory-fallback",
        "policy_source": "case_memory_snapshot",
        "workspace_key": f"{scope.user_id}/{scope.workspace_id}",
        "user_id": scope.user_id,
        "workspace_id": scope.workspace_id,
        "turn_count": graph.turn_count,
        "active_node_limit": scope.workspace.memory.active_node_limit,
        "active_edge_limit": scope.workspace.memory.active_edge_limit,
        "archive_after_turns": scope.workspace.memory.archive_after_turns,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "active_node_count": len(active_nodes),
        "hidden_node_count": len(hidden_nodes),
        "archived_node_count": graph.archived_node_count,
        "rejected_node_count": sum(1 for node in graph.nodes if node.status == "rejected"),
        "removed_node_count": sum(1 for node in graph.nodes if node.status == "removed"),
        "active_edge_count": sum(
            1 for edge in graph.edges if edge.status not in {"archived", "rejected", "removed"}
        ),
        "hidden_edge_count": sum(
            1 for edge in graph.edges if edge.status in {"archived", "rejected", "removed"}
        ),
        "active_examples": [
            {
                "node_id": node.id,
                "kind": node.kind,
                "label": node.label,
                "status": node.status,
            }
            for node in active_nodes[:12]
        ],
        "hidden_examples": [
            {
                "node_id": node.id,
                "kind": node.kind,
                "label": node.label,
                "status": node.status,
            }
            for node in hidden_nodes[:12]
        ],
    }


def _resolve_scope_id(value: str | None, default: str) -> str:
    cleaned = str(value or "").strip()
    return cleaned or default


def _query_conversation_scope(request: Request) -> dict[str, str | None | bool]:
    return {
        "user_id": _query_scope_id(request, "user_id", DEFAULT_USER_ID),
        "workspace_id": _query_scope_id(request, "workspace_id", DEFAULT_WORKSPACE_ID),
        "conversation_id": _query_scope_id(request, "conversation_id", None),
        "session_id": _query_scope_id(request, "session_id", None),
        "prefer_existing": _truthy(request.query_params.get("prefer_existing")),
    }


def _query_workspace_scope(request: Request) -> dict[str, str | None]:
    return {
        "user_id": _query_scope_id(request, "user_id", DEFAULT_USER_ID),
        "workspace_id": _query_scope_id(request, "workspace_id", DEFAULT_WORKSPACE_ID),
    }


def _query_scope_id(
    request: Request,
    name: str,
    default: str | None,
) -> str | None:
    value = request.query_params.get(name)
    if value is None:
        return default
    cleaned = value.strip()
    if not 1 <= len(cleaned) <= 128:
        raise ValueError(f"Invalid {name}.")
    return cleaned


def _format_graph_evidence(evidence: dict[str, Any] | None) -> list[str]:
    if not evidence:
        return []

    lines = [
        "Graph evidence: the Surreal working map found concrete records behind this rationale."
    ]

    messages = [
        item
        for item in evidence.get("messages") or ()
        if isinstance(item, dict)
    ]
    if messages:
        lines.append("Relevant transcript records:")
        for item in messages[-3:]:
            role = _human_label(str(item.get("role") or "message"))
            index = item.get("message_index")
            text = _compact_preview(str(item.get("text") or ""), limit=180)
            if text:
                lines.append(f"- Message {index} ({role}): {text}")

    nodes = [
        item
        for item in evidence.get("supporting_nodes") or ()
        if isinstance(item, dict)
    ]
    if nodes:
        labels = []
        for item in nodes[:8]:
            kind = _human_label(str(item.get("kind") or "node"))
            label = _human_label(str(item.get("label") or item.get("node_id") or "item"))
            seen = item.get("seen_count")
            seen_text = f", seen {seen}x" if seen else ""
            labels.append(f"{label} ({kind}{seen_text})")
        lines.append("Working-map support: " + "; ".join(labels) + ".")

    edges = [
        item
        for item in evidence.get("support_edges") or ()
        if isinstance(item, dict)
    ]
    if edges:
        lines.append("Support paths in the map:")
        for item in edges[:5]:
            source = _human_label(str(item.get("source_label") or item.get("source_node_id") or "source"))
            target = _human_label(str(item.get("target_label") or item.get("target_node_id") or "target"))
            kind = _human_label(str(item.get("kind") or "supports"))
            lines.append(f"- {source} -> {kind} -> {target}")

    provenance = [
        item
        for item in evidence.get("provenance") or ()
        if isinstance(item, dict)
    ]
    if provenance:
        examples = []
        for item in provenance[:4]:
            field = _human_label(str(item.get("field") or "field"))
            raw_evidence = _compact_preview(str(item.get("evidence") or ""), limit=90)
            if raw_evidence:
                examples.append(f"{field}: {raw_evidence}")
        if examples:
            lines.append("Provenance examples: " + "; ".join(examples) + ".")

    experiments = [
        item
        for item in evidence.get("experiments") or ()
        if isinstance(item, dict)
    ]
    if experiments:
        experiment = experiments[0]
        try_step = _compact_preview(str(experiment.get("try_step") or ""), limit=160)
        if try_step:
            lines.append(f"The proposed experiment was recorded as: {try_step}")

    return lines


def explain_trace_human_readable(
    trace: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> str:
    """Translate a stored deterministic trace into readable intervention rationale."""

    extraction = trace.get("extraction") or {}
    kernel = trace.get("kernel") or {}
    selection = trace.get("selection") or {}
    hypotheses = kernel.get("hypotheses") or []
    formulations = kernel.get("formulations") or []
    clarifying_moves = kernel.get("clarifying_moves") or []
    candidates = kernel.get("candidates") or []
    selected = selection.get("matched_candidate") or {}
    selected_intervention = (
        selection.get("intervention")
        or selected.get("intervention")
        or "the selected intervention"
    )

    lines = [
        f"The assistant used {_human_label(str(selected_intervention))} because the structured state pointed in that direction.",
    ]

    observations = _observation_summary(extraction)
    if observations:
        lines.append(f"It noticed {observations}.")

    if evidence_lines := _format_graph_evidence(evidence):
        lines.extend(evidence_lines)

    if hypotheses:
        grouped = _group_hypotheses(hypotheses)
        lines.append("The kernel treated these as tentative hypotheses, not labels:")
        for source, patterns in grouped:
            lines.append(f"- {_source_label(source)}: {_format_labels(patterns)}.")

    if formulations:
        lines.append(
            "Differential formulation: the kernel kept multiple possible maps active instead of treating one as certain."
        )
        for item in formulations[:3]:
            label = _human_label(str(item.get("label") or item.get("formulation")))
            evidence = [
                str(hypothesis.get("pattern"))
                for hypothesis in (item.get("evidence") or ())
                if hypothesis.get("pattern")
            ]
            evidence_text = (
                f" Supported by {_format_labels(evidence[:5])}."
                if evidence
                else ""
            )
            question = str(item.get("discriminating_question") or "").strip()
            question_text = f" It would check: {question}" if question else ""
            lines.append(
                f"- {label} ({item.get('score')}): {item.get('summary') or ''}"
                f"{evidence_text}{question_text}"
            )

    selected_clarifier = selection.get("clarifying_move") or (
        clarifying_moves[0] if clarifying_moves else None
    )
    if selected_clarifier:
        targets = _clean_label_list(selected_clarifier.get("target_formulations"))
        target_text = (
            f" between {_format_labels(targets)}"
            if len(targets) > 1
            else f" for {_format_labels(targets)}"
            if targets
            else ""
        )
        lines.append(
            "Active inquiry: the kernel proposed a clarifying move"
            f"{target_text}."
        )
        if rationale := str(selected_clarifier.get("rationale") or "").strip():
            lines.append(f"Clarifying rationale: {rationale}")
        if question := str(selected_clarifier.get("question") or "").strip():
            lines.append(f"Clarifying question: {question}")

    if candidates:
        top_candidates = [
            f"{_human_label(str(item.get('intervention')))}"
            + (
                f" ({item.get('score')})"
                if item.get("score") is not None
                else ""
            )
            for item in candidates[:3]
        ]
        lines.append(
            "It then ranked safe candidates: "
            f"{', '.join(top_candidates)}."
        )

    policy = trace.get("policy") or {}
    if policy:
        adjustments = policy.get("adjustments") or []
        summary = policy.get("summary") or {}
        if adjustments:
            lines.append(
                "Adaptive policy: prior feedback nudged this turn's ranking."
            )
            for item in adjustments[:4]:
                intervention = _human_label(str(item.get("intervention") or "move"))
                delta = item.get("delta")
                reasons = [
                    str(reason)
                    for reason in (item.get("reasons") or ())
                    if str(reason).strip()
                ]
                reason_text = f" {reasons[0]}" if reasons else ""
                lines.append(
                    f"- {intervention}: score {item.get('base_score')} -> "
                    f"{item.get('adjusted_score')} ({delta:+}).{reason_text}"
                )
        counts = summary.get("counts") or {}
        if counts and (
            counts.get("experiment_outcomes") or counts.get("map_corrections")
        ):
            lines.append(
                "Policy facts available: "
                f"{counts.get('experiment_outcomes', 0)} experiment outcomes and "
                f"{counts.get('map_corrections', 0)} working-map corrections."
            )

    if selected:
        reasons = selected.get("hypotheses") or []
        patterns = [
            str(item.get("pattern"))
            for item in reasons
            if item.get("pattern")
        ]
        if patterns:
            lines.append(
                f"The chosen move was supported by {_format_labels(patterns)}."
            )

    recipe = selection.get("recipe") or {}
    if recipe:
        steps = [_human_label(str(step)) for step in recipe.get("steps") or ()]
        if steps:
            lines.append(
                "The kernel also generated a recipe-level plan: "
                f"{_human_label(str(recipe.get('recipe')))} = {' -> '.join(steps)}."
            )
        if rationale := recipe.get("rationale"):
            lines.append(f"Recipe rationale: {rationale}")

    coherence = trace.get("plan_coherence") or {}
    if coherence:
        lines.append(
            "Plan coherence check: "
            f"{_human_label(str(coherence.get('status') or 'checked'))}."
        )
        for issue in (coherence.get("issues") or [])[:4]:
            detail = str(issue.get("detail") or "").strip()
            if detail:
                lines.append(
                    f"- {_human_label(str(issue.get('code') or 'issue'))}: {detail}"
                )
        for repair in (coherence.get("repairs") or [])[:4]:
            detail = str(repair.get("detail") or "").strip()
            if detail:
                lines.append(f"- Repair: {detail}")

    backward = _backward_justification_report(
        selected_intervention=str(selected_intervention),
        selection=selection,
        selected_candidate=selected,
    )
    if backward:
        lines.extend(_format_backward_justification(backward))

    if exercise := selection.get("exercise"):
        lines.append(f"The concrete exercise came from that intervention: {exercise}")

    if question := selection.get("question"):
        lines.append(f"The follow-up question was meant to keep the next step small: {question}")

    longitudinal_patterns = trace.get("longitudinal") or []
    if longitudinal_patterns:
        lines.append("Across turns, the longitudinal detector is also holding these lightly:")
        for pattern in longitudinal_patterns[:3]:
            label = _human_label(str(pattern.get("label") or pattern.get("pattern")))
            turns = pattern.get("turns") or []
            turn_text = f" across turns {', '.join(str(turn) for turn in turns)}" if turns else ""
            description = str(pattern.get("description") or "").strip()
            suffix = f": {description}" if description else ""
            lines.append(f"- {label}{turn_text}{suffix}")

    contraindications = [
        str(reason)
        for item in candidates
        for reason in (item.get("contraindications") or [])
    ]
    if contraindications:
        lines.append(
            "Safety or timing constraints filtered some options: "
            f"{_format_labels(contraindications)}."
        )

    lines.append("This is a coaching rationale for the response, not a diagnosis.")
    return "\n".join(lines)


def _backward_justification_report(
    *,
    selected_intervention: str,
    selection: dict[str, Any],
    selected_candidate: dict[str, Any],
) -> dict[str, Any] | None:
    report = selection.get("backward_justification")
    if isinstance(report, dict):
        return report
    if not selected_intervention:
        return None

    possible_patterns = TherapeuticReasoningKernel().patterns_for_intervention(
        selected_intervention
    )
    if not possible_patterns:
        return None
    satisfied_patterns = _unique_labels(
        str(item.get("pattern"))
        for item in selected_candidate.get("hypotheses", ())
        if item.get("pattern")
    )
    satisfied_set = set(satisfied_patterns)
    return {
        "intervention": selected_intervention,
        "coherent": bool(satisfied_patterns),
        "safe": not selected_candidate.get("contraindications"),
        "possible_patterns": possible_patterns,
        "satisfied_patterns": satisfied_patterns,
        "alternative_patterns": tuple(
            pattern for pattern in possible_patterns if pattern not in satisfied_set
        ),
        "satisfied_hypotheses": selected_candidate.get("hypotheses") or (),
        "contraindications": selected_candidate.get("contraindications") or (),
    }


def _format_backward_justification(report: dict[str, Any]) -> list[str]:
    intervention = _human_label(str(report.get("intervention") or "this move"))
    lines = [
        f"Backward check: the kernel also asked what would need to be true for {intervention} to make sense.",
    ]

    possible_patterns = _clean_label_list(report.get("possible_patterns"))
    if possible_patterns:
        lines.append(
            "That move is coherent for states involving "
            f"{_format_labels(possible_patterns)}."
        )

    satisfied_patterns = _clean_label_list(report.get("satisfied_patterns"))
    if satisfied_patterns:
        lines.append(
            "In this turn, the matched path was "
            f"{_format_labels(satisfied_patterns)}."
        )
    else:
        lines.append(
            "This turn did not clearly satisfy a kernel path for that move, so the explanation should be treated as low-confidence."
        )

    alternative_patterns = _clean_label_list(report.get("alternative_patterns"))
    if alternative_patterns:
        lines.append(
            "Other possible routes to the same move were not the main support here: "
            f"{_format_labels(alternative_patterns[:6])}."
        )

    contraindications = _clean_label_list(report.get("contraindications"))
    if contraindications:
        lines.append(
            "The backwards check also found blockers: "
            f"{_format_labels(contraindications)}."
        )
    elif report.get("coherent"):
        lines.append("No timing or safety blocker was found for the selected move.")

    return lines


def _is_framework_explanation_request(message: str) -> bool:
    """Detect educational framework questions that should not become coaching turns."""

    text = f" {message.strip()} "
    lowered = text.casefold()
    explanation_cue = bool(
        re.search(
            r"\b(what is|what's|explain|tell me about|how does|how do|why use|"
            r"difference between|compare|overview|walk me through|define|"
            r"framework|modality|therapeutic system|therapy system|approach)\b",
            lowered,
        )
    )
    if not explanation_cue:
        return False

    if re.search(r"\b(CBT|ACT|REBT|DBT|MBSR|GTD|OKR|OKRs|WOOP)\b", text):
        return True
    return bool(
        re.search(
            r"\b(cognitive behavioral|acceptance and commitment|"
            r"rational emotive|dialectical behavior|dialectical behavioural|"
            r"mindfulness-based stress reduction|mindfulness based stress reduction|"
            r"focusing|coaching framework|therapeutic framework|"
            r"therapeutic system|therapy system|modality|modalities|"
            r"goal direction|goal-direction|getting things done|weekly review|"
            r"personal kanban|implementation intention)\b",
            lowered,
        )
    )


def framework_explanation_response(message: str) -> str:
    """Render a concise educational explanation outside the coaching turn loop."""

    requested = _requested_frameworks(message)
    if not requested:
        requested = ("framework",)

    sections = []
    if "framework" in requested:
        sections.append(
            "Current coaching framework:\n"
            "- The model first extracts observations from what you said: situation, thoughts, emotions, urges, behavior, values, and goals.\n"
            "- The relational kernel treats ACT, CBT, REBT, DBT, MBSR, Focusing, goal-direction, focus areas, and cross-system loops as tentative lenses.\n"
            "- The coach then chooses a small next move, while avoiding moves that are too intense for the current state.\n"
            "- The Working Map, Direction section, and tiny experiments are there to make the reasoning correctable over time."
        )

    framework_notes = {
        "act": (
            "ACT:\n"
            "- Focus: psychological flexibility.\n"
            "- Useful when thoughts are sticky, identity-laden, or driving avoidance.\n"
            "- Typical moves: values clarification, cognitive defusion, acceptance, and committed action."
        ),
        "cbt": (
            "CBT:\n"
            "- Focus: the situation-thought-emotion-behavior cycle.\n"
            "- Useful when a thought is testable or a prediction may be distorted.\n"
            "- Typical moves: evidence checks, thought records, behavioral experiments, and graded action."
        ),
        "rebt": (
            "REBT:\n"
            "- Focus: rigid demands and global self-ratings.\n"
            "- Useful when the language has musts, shoulds, awfulizing, low frustration tolerance, or self-downing.\n"
            "- Typical moves: disputing the demand, replacing it with a flexible preference, and separating worth from outcomes."
        ),
        "dbt": (
            "DBT:\n"
            "- Focus: balancing acceptance and change while building practical skills.\n"
            "- Useful for high emotion, crisis urges, self-invalidation, or interpersonal/boundary situations.\n"
            "- Typical moves: mindfulness, distress tolerance, emotion regulation, and interpersonal effectiveness."
        ),
        "mbsr": (
            "MBSR:\n"
            "- Focus: mindfulness-based stress management and noticing the stress response before reacting.\n"
            "- Useful when stress load, body tension, rumination, autopilot reactivity, or vulnerability factors are prominent.\n"
            "- Typical moves: breathing space, body scan, mindful labeling, returning attention, and small self-care checks."
        ),
        "focusing": (
            "Focusing:\n"
            "- Focus: the body's felt sense of a problem before forcing words or solutions.\n"
            "- Useful when the experience is vague, stuck, or hard to name.\n"
            "- Typical moves: pause, sense the whole of it, find a handle, check whether the words fit, and let the meaning unfold gently."
        ),
        "goal_direction": (
            "Goal-direction layer:\n"
            "- Focus: translating values and objectives into executable next actions.\n"
            "- Useful when the issue is vague goals, open loops, missing next actions, predictable obstacles, or review/recommitment.\n"
            "- Typical moves: OKR-lite objective clarification, WOOP obstacle planning, GTD capture, WIP limiting, implementation intentions, and weekly review."
        ),
    }
    for key in ("act", "cbt", "rebt", "dbt", "mbsr", "focusing", "goal_direction"):
        if key in requested:
            sections.append(framework_notes[key])

    if len(sections) > 1:
        sections.append(
            "How to read these in the app: they are hypotheses and tools, not diagnoses. "
            "The useful question is which lens makes the next humane, concrete step clearer."
        )
    return "\n\n".join(sections)


def _requested_frameworks(message: str) -> tuple[str, ...]:
    text = f" {message} "
    lowered = text.casefold()
    requested = []
    if re.search(r"\b(CBT)\b", text) or "cognitive behavioral" in lowered:
        requested.append("cbt")
    if re.search(r"\b(ACT)\b", text) or "acceptance and commitment" in lowered:
        requested.append("act")
    if re.search(r"\b(REBT)\b", text) or "rational emotive" in lowered:
        requested.append("rebt")
    if re.search(r"\b(DBT)\b", text) or "dialectical behavior" in lowered or "dialectical behavioural" in lowered:
        requested.append("dbt")
    if re.search(r"\b(MBSR)\b", text) or "mindfulness-based stress reduction" in lowered or "mindfulness based stress reduction" in lowered:
        requested.append("mbsr")
    if "focusing" in lowered:
        requested.append("focusing")
    if (
        re.search(r"\b(GTD|OKR|OKRs|WOOP)\b", text)
        or "goal direction" in lowered
        or "goal-direction" in lowered
        or "getting things done" in lowered
        or "personal kanban" in lowered
        or "implementation intention" in lowered
        or "weekly review" in lowered
    ):
        requested.append("goal_direction")
    if re.search(
        r"\b(coaching framework|therapeutic framework|therapeutic system|"
        r"therapy system|modality|modalities|framework)\b",
        lowered,
    ):
        requested.insert(0, "framework")
    return tuple(dict.fromkeys(requested))


def _observation_summary(extraction: dict[str, Any]) -> str:
    parts = []
    for key, label in (
        ("emotions", "emotion"),
        ("behaviors", "behavior"),
        ("thoughts", "thought"),
        ("beliefs", "belief"),
        ("values", "value"),
        ("goals", "goal"),
        ("objectives", "objective"),
        ("projects", "project"),
        ("next_actions", "next action"),
        ("obstacles", "obstacle"),
        ("features", "state feature"),
    ):
        values = extraction.get(key) or []
        if values:
            parts.append(f"{label}s: {_format_values(values, limit=3)}")
    if extraction.get("distress") is not None:
        parts.append(f"distress around {extraction['distress']}/10")
    return "; ".join(parts)


def _group_hypotheses(items: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for item in items:
        source = str(item.get("source") or "other")
        pattern = item.get("pattern")
        if pattern:
            grouped.setdefault(source, []).append(str(pattern))
    return [(source, patterns) for source, patterns in grouped.items()]


def _format_values(values: list[Any], *, limit: int) -> str:
    rendered = [str(value) for value in values[:limit]]
    if len(values) > limit:
        rendered.append(f"{len(values) - limit} more")
    return ", ".join(rendered)


def _clean_label_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = values
    return _unique_labels(str(value) for value in raw_values if str(value).strip())


def _unique_labels(values: Any) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        label = str(value).strip()
        if label and label not in seen:
            seen.add(label)
            unique.append(label)
    return unique


def _format_labels(values: list[str]) -> str:
    unique = []
    seen = set()
    for value in values:
        label = _human_label(value)
        if label not in seen:
            seen.add(label)
            unique.append(label)
    if not unique:
        return "none"
    if len(unique) == 1:
        return unique[0]
    return f"{', '.join(unique[:-1])}, and {unique[-1]}"


def _source_label(source: str) -> str:
    return {
        "act": "ACT process",
        "cbt": "CBT pattern",
        "dbt": "DBT skill target",
        "focus": "coaching focus",
        "goal_direction": "goal-direction execution signal",
        "mbsr": "MBSR stress-management target",
        "rebt": "REBT belief pattern",
        "loop": "clinical loop",
        "emotion": "emotion",
        "policy": "policy",
    }.get(source, _human_label(source))


def _human_label(value: str) -> str:
    return value.replace("_", " ").strip()


def _default_coach_factory(
    *,
    api_key_file: str | Path,
    model_name: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> CoachFactory:
    if dry_run:
        return DeterministicKernelGuidedCoach

    def factory() -> KernelGuidedCoach:
        return KernelGuidedCoach(
            api_key=read_api_key(Path(api_key_file)),
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return factory


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data)}


def _truthy(value: str | None) -> bool:
    return value is not None and value.casefold() in {"1", "true", "yes", "on"}


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _drop_empty(item)) not in (None, {}, [], (), "")
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := _drop_empty(item)) not in (None, {}, [], (), "")
        ]
    if isinstance(value, tuple):
        return [
            cleaned
            for item in value
            if (cleaned := _drop_empty(item)) not in (None, {}, [], (), "")
        ]
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the coaching chat API and SSE app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key-file",
        default=DEFAULT_API_KEY_FILE,
        help=f"Path to the DeepSeek API key file. Default: {DEFAULT_API_KEY_FILE}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"DeepSeek model id. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without DeepSeek calls using deterministic kernel output.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help=(
            "Path for the legacy JSON backend. Setting this without "
            "--store-backend selects json. Use --store-backend memory to disable "
            "persistence."
        ),
    )
    parser.add_argument(
        "--store-backend",
        choices=("memory", "json", "surreal"),
        default=None,
        help=(
            "Persistence backend for workspace/conversation snapshots. "
            "Defaults to json when --state-file is set, otherwise surreal. "
            "Can also be set with COACH_STORE_BACKEND."
        ),
    )
    parser.add_argument(
        "--surreal-url",
        default=None,
        help=(
            "SurrealDB endpoint for --store-backend surreal, for example "
            "mem://, ws://127.0.0.1:8000/rpc, or an embedded file URL supported "
            "by the installed SurrealDB Python SDK. Default: "
            f"{Path(DEFAULT_SURREAL_FILE).resolve().as_uri()}"
        ),
    )
    parser.add_argument("--surreal-namespace", default="coach")
    parser.add_argument("--surreal-database", default="coach")
    parser.add_argument("--surreal-record-id", default="app_state:default")
    parser.add_argument("--surreal-user", default=None)
    parser.add_argument("--surreal-password", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import uvicorn

    uvicorn.run(
        create_app(
            api_key_file=args.api_key_file,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            state_file=args.state_file or None,
            store_backend=args.store_backend,
            surreal_url=args.surreal_url,
            surreal_namespace=args.surreal_namespace,
            surreal_database=args.surreal_database,
            surreal_record_id=args.surreal_record_id,
            surreal_user=args.surreal_user,
            surreal_password=args.surreal_password,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


CHAT_APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coach Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8f5;
      --panel: #ffffff;
      --ink: #1d2320;
      --muted: #5f6862;
      --line: #d8ded7;
      --accent: #166a5b;
      --accent-strong: #0f4c41;
      --soft: #eef4f0;
      --user: #e6f0ff;
      --coach: #ffffff;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html,
    body {
      height: 100%;
    }
    body {
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
    }
    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      height: 100vh;
      height: 100dvh;
      min-height: 0;
      overflow: hidden;
    }
    main {
      display: grid;
      grid-template-rows: auto auto auto;
      align-content: start;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
    }
    .header-main {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 680;
      letter-spacing: 0;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .scope-control {
      display: flex;
      align-items: center;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .workspace-input {
      width: 148px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 9px;
      color: var(--ink);
      background: #fff;
      font: inherit;
      font-size: 13px;
      outline: none;
    }
    .workspace-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(22, 106, 91, 0.12);
    }
    .conversation-select {
      width: 210px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 9px;
      color: var(--ink);
      background: #fff;
      font: inherit;
      font-size: 13px;
      outline: none;
    }
    .conversation-select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(22, 106, 91, 0.12);
    }
    button.scope-button {
      width: auto;
      min-width: 0;
      height: 34px;
      padding: 0 10px;
      font-size: 12px;
      font-weight: 720;
    }
    .messages {
      min-height: 0;
      max-height: calc(100dvh - 170px);
      overflow-y: auto;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: linear-gradient(#ffffff, #fbfcfa);
    }
    .message-wrap {
      max-width: min(760px, 86%);
      display: flex;
      flex-direction: column;
      gap: 7px;
    }
    .message-wrap.user {
      align-self: flex-end;
      align-items: flex-end;
    }
    .message-wrap.coach {
      align-self: flex-start;
      align-items: flex-start;
    }
    .message-wrap.reflection,
    .message-wrap.info {
      align-self: flex-start;
      align-items: stretch;
      max-width: min(760px, 86%);
    }
    .message {
      max-width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 15px;
    }
    .message.user {
      align-self: flex-end;
      background: var(--user);
      border-color: #bed4f5;
    }
    .message.coach {
      align-self: flex-start;
      background: var(--coach);
    }
    .reflection-box,
    .info-box {
      max-width: 100%;
      border: 1px solid #c5d4ef;
      border-left: 3px solid #4267b2;
      border-radius: 8px;
      background: #f4f7ff;
      padding: 11px 12px;
      color: #1f2a44;
      overflow-wrap: anywhere;
    }
    .reflection-label,
    .info-label {
      margin-bottom: 6px;
      color: #34528d;
      font-size: 12px;
      font-weight: 760;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    .reflection-text,
    .info-text {
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 14px;
    }
    .markdown-body {
      white-space: normal;
    }
    .markdown-body p {
      margin: 0 0 10px;
    }
    .markdown-body p:last-child,
    .markdown-body ul:last-child,
    .markdown-body ol:last-child,
    .markdown-body pre:last-child {
      margin-bottom: 0;
    }
    .markdown-body ul,
    .markdown-body ol {
      margin: 6px 0 10px;
      padding-left: 22px;
    }
    .markdown-body li {
      margin: 3px 0;
    }
    .markdown-body h1,
    .markdown-body h2,
    .markdown-body h3 {
      margin: 0 0 8px;
      color: inherit;
      font-size: 1em;
      font-weight: 760;
      letter-spacing: 0;
    }
    .markdown-body code {
      border-radius: 5px;
      background: rgba(31, 42, 68, 0.08);
      padding: 1px 4px;
      font: 0.92em/1.35 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }
    .markdown-body pre {
      margin: 6px 0 10px;
      overflow: auto;
      border-radius: 7px;
      background: rgba(31, 42, 68, 0.08);
      padding: 9px;
      white-space: pre;
    }
    .markdown-body pre code {
      background: transparent;
      padding: 0;
    }
    .markdown-body a {
      color: var(--accent-strong);
      text-decoration: underline;
      text-underline-offset: 2px;
    }
    .message-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .why-chip {
      width: auto;
      min-width: 0;
      height: 28px;
      padding: 0 10px;
      border: 1px solid #b7ccc5;
      border-radius: 14px;
      background: #f2faf6;
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 650;
    }
    .why-chip:hover { background: #e4f2ed; }
    .why-chip:disabled { opacity: 0.7; }
    .experiment-chip {
      background: #ffffff;
    }
    .trace-explanation {
      max-width: min(680px, 100%);
      border: 1px solid #cddbd5;
      border-left: 3px solid var(--accent);
      border-radius: 8px;
      background: #f7fbf8;
      padding: 11px 12px;
      color: #26312d;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .experiment-card {
      width: min(680px, 100%);
      border: 1px solid #d0ddd7;
      border-radius: 8px;
      background: #fbfdfb;
      padding: 12px;
      color: #26312d;
      font-size: 13px;
      line-height: 1.4;
    }
    .experiment-card[hidden] {
      display: none;
    }
    .experiment-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 9px;
    }
    .experiment-title {
      font-size: 13px;
      font-weight: 760;
      color: var(--ink);
    }
    .experiment-status {
      flex: 0 0 auto;
      border: 1px solid #c6d7d0;
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--accent-strong);
      background: #eef7f2;
      font-size: 11px;
      font-weight: 720;
      text-transform: uppercase;
    }
    .experiment-rows {
      display: grid;
      gap: 7px;
    }
    .experiment-row {
      display: grid;
      grid-template-columns: 78px 1fr;
      gap: 10px;
    }
    .experiment-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 760;
      text-transform: uppercase;
    }
    .experiment-value {
      overflow-wrap: anywhere;
    }
    .experiment-rationale {
      margin-top: 9px;
      color: var(--muted);
      font-size: 12px;
    }
    .experiment-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: 10px;
    }
    button.experiment-action {
      width: auto;
      min-width: 0;
      height: 30px;
      padding: 0 9px;
      border: 1px solid #bfd0c9;
      border-radius: 8px;
      background: #ffffff;
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 720;
    }
    button.experiment-action:hover {
      background: #eaf4ef;
    }
    .composer {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 16px 22px 20px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }
    textarea {
      width: 100%;
      min-height: 54px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      color: var(--ink);
      font: inherit;
      line-height: 1.4;
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(22, 106, 91, 0.12);
    }
    button {
      min-width: 86px;
      height: 54px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    button.icon-button {
      width: 38px;
      min-width: 38px;
      height: 38px;
      padding: 0;
      font-size: 22px;
      line-height: 1;
    }
    .progress-toast {
      align-self: flex-end;
      width: min(420px, 86%);
      border: 1px solid #c9d8d2;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 8px 24px rgba(20, 32, 28, 0.1);
      padding: 12px 14px;
      margin-top: -6px;
    }
    .progress-toast[hidden] {
      display: none;
    }
    .progress-copy {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 620;
      margin-bottom: 9px;
    }
    .progress-percent {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      font-variant-numeric: tabular-nums;
    }
    .progress-track {
      width: 100%;
      height: 7px;
      overflow: hidden;
      border-radius: 999px;
      background: #e4ebe6;
    }
    .progress-bar {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
      transition: width 220ms ease;
    }
    aside {
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-width: 0;
      min-height: 0;
      background: var(--soft);
    }
    .side-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 18px;
      border-bottom: 1px solid var(--line);
      background: #f4f7f2;
    }
    .side-head h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .session-id {
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .side-tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      border-bottom: 1px solid var(--line);
      background: #f8faf6;
    }
    button.side-tab {
      min-width: 0;
      width: 100%;
      height: 40px;
      border-radius: 0;
      border-bottom: 2px solid transparent;
      background: transparent;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }
    button.side-tab:hover {
      background: #eef4f0;
      color: var(--accent-strong);
    }
    button.side-tab.active {
      color: var(--accent-strong);
      border-bottom-color: var(--accent);
      background: #ffffff;
    }
    .side-panel {
      min-height: 0;
      overflow: auto;
    }
    .side-panel[hidden] {
      display: none;
    }
    #tracePanel {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
    }
    #tracePanel[hidden] {
      display: none;
    }
    .controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      color: var(--muted);
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      user-select: none;
    }
    input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }
    pre {
      min-height: 0;
      margin: 0;
      padding: 16px 18px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      color: #26312d;
    }
    .map-panel {
      padding: 14px;
    }
    .map-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .map-head h3 {
      margin: 0;
      font-size: 14px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .map-summary {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    button.mirror-button {
      width: auto;
      min-width: 92px;
      height: 34px;
      padding: 0 10px;
      background: var(--accent);
      color: white;
      font-size: 12px;
      font-weight: 750;
    }
    .map-empty {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      padding: 10px 0;
    }
    .map-section {
      margin-bottom: 16px;
    }
    .map-section h4 {
      margin: 0 0 8px;
      color: #3d4742;
      font-size: 12px;
      font-weight: 760;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    .focus-section {
      border: 1px solid #c8dbd4;
      border-radius: 8px;
      background: #f8fcfa;
      padding: 10px;
    }
    .direction-section {
      border: 1px solid #cbd8e7;
      border-radius: 8px;
      background: #f7faff;
      padding: 10px;
    }
    .focus-grid {
      display: grid;
      gap: 10px;
    }
    .focus-row {
      display: grid;
      gap: 5px;
    }
    .focus-kind {
      color: var(--muted);
      font-size: 11px;
      font-weight: 760;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    .focus-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .focus-chip {
      border: 1px solid #bfd0c9;
      border-radius: 8px;
      background: #ffffff;
      padding: 5px 7px;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .policy-section {
      border: 1px solid #c7d9e9;
      border-radius: 8px;
      background: #f7fbff;
      padding: 10px;
      margin-bottom: 16px;
    }
    .policy-section[hidden] {
      display: none;
    }
    .policy-grid {
      display: grid;
      gap: 9px;
    }
    .policy-row {
      display: grid;
      gap: 5px;
    }
    .policy-kind {
      color: #506579;
      font-size: 11px;
      font-weight: 760;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    .policy-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .policy-chip {
      border: 1px solid #bfd0df;
      border-radius: 8px;
      background: #ffffff;
      padding: 5px 7px;
      color: #1f2f3c;
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .map-items {
      display: grid;
      gap: 8px;
    }
    .map-item {
      border: 1px solid #d3ded8;
      border-radius: 8px;
      background: #ffffff;
      padding: 10px;
    }
    .map-item.rejected {
      background: #fbf6f3;
      border-color: #e0c8bd;
    }
    .map-label {
      color: var(--ink);
      font-size: 13px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .map-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
    }
    .map-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 9px;
    }
    button.map-action {
      width: auto;
      min-width: 0;
      height: 28px;
      padding: 0 8px;
      border: 1px solid #bfd0c9;
      border-radius: 8px;
      background: #f8fbf8;
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 700;
    }
    button.map-action:hover {
      background: #e8f3ee;
    }
    button.map-action.reject {
      color: #80513e;
      border-color: #d8bdae;
      background: #fff8f4;
    }
    button.map-action.remove {
      color: #7f3434;
      border-color: #d7b4b4;
      background: #fff7f7;
    }
    @media (max-width: 860px) {
      .shell {
        grid-template-columns: 1fr;
        grid-template-rows: minmax(0, 1fr) minmax(220px, 30vh);
      }
      main { border-right: 0; min-height: 70vh; }
      aside { min-height: 30vh; border-top: 1px solid var(--line); }
      .messages { max-height: calc(70dvh - 170px); }
      .message-wrap { max-width: 94%; }
      .composer { grid-template-columns: 1fr; }
      button { width: 100%; }
      button.icon-button { width: 38px; }
      button.scope-button { width: auto; }
      .workspace-input { width: 130px; }
      .conversation-select { width: 180px; }
      .why-chip { width: auto; }
      button.experiment-action { width: auto; }
      .progress-toast { width: 94%; }
      .experiment-row { grid-template-columns: 1fr; gap: 2px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <main>
      <header>
        <div class="header-main">
          <h1>Coach Chat</h1>
          <div class="status" id="status">Ready</div>
        </div>
        <div class="header-actions">
          <label class="scope-control" for="workspaceInput">
            <span>Workspace</span>
            <input class="workspace-input" id="workspaceInput" type="text" maxlength="128" autocomplete="off">
          </label>
          <label class="scope-control" for="conversationSelect">
            <span>Conversation</span>
            <select class="conversation-select" id="conversationSelect"></select>
          </label>
          <button class="scope-button" id="newWorkspace" type="button">New workspace</button>
          <button class="icon-button" id="newChat" type="button" title="New conversation" aria-label="New conversation">+</button>
        </div>
      </header>
      <section class="messages" id="messages" aria-live="polite"></section>
      <form class="composer" id="composer">
        <textarea id="message" name="message" placeholder="Type a message..." autocomplete="off" required></textarea>
        <button id="send" type="submit">Send</button>
      </form>
    </main>
    <aside>
      <div class="side-head">
        <h2>Inspector</h2>
        <span class="session-id" id="session"></span>
      </div>
      <div class="side-tabs" role="tablist" aria-label="Inspector views">
        <button class="side-tab active" id="traceTab" type="button" role="tab" aria-selected="true" aria-controls="tracePanel">Trace</button>
        <button class="side-tab" id="mapTab" type="button" role="tab" aria-selected="false" aria-controls="mapPanel">Working Map</button>
      </div>
      <section class="side-panel" id="tracePanel" role="tabpanel" aria-labelledby="traceTab">
        <div class="controls">
          <label class="toggle"><input type="checkbox" id="trace" checked> Show trace</label>
        </div>
        <pre id="traceOut">{}</pre>
      </section>
      <section class="side-panel map-panel" id="mapPanel" role="tabpanel" aria-labelledby="mapTab" hidden>
        <div class="map-head">
          <div>
            <h3>Working Map</h3>
            <span class="map-summary" id="mapSummary">Empty</span>
          </div>
          <button class="mirror-button" id="mirrorMap" type="button" disabled>Reflective listening</button>
        </div>
        <section class="map-section policy-section" id="policySummary" hidden></section>
        <div class="map-list" id="formulationMap"></div>
      </section>
    </aside>
  </div>
  <div class="progress-toast" id="progressToast" role="status" aria-live="polite" hidden>
    <div class="progress-copy">
      <span id="progressLabel">Generating response</span>
      <span class="progress-percent" id="progressPercent">0%</span>
    </div>
    <div class="progress-track" aria-hidden="true">
      <div class="progress-bar" id="progressBar"></div>
    </div>
  </div>
  <script>
    const userKey = "coach.userId";
    const workspaceKey = "coach.workspaceId";
    const conversationKey = "coach.conversationId";
    const legacySessionKey = "coach.sessionId";
    let userId = localStorage.getItem(userKey) || "default";
    let workspaceId = localStorage.getItem(workspaceKey) || "default";
    let sessionId = localStorage.getItem(conversationKey)
      || localStorage.getItem(legacySessionKey)
      || "";
    localStorage.setItem(userKey, userId);
    localStorage.setItem(workspaceKey, workspaceId);
    if (sessionId) {
      localStorage.setItem(conversationKey, sessionId);
      localStorage.setItem(legacySessionKey, sessionId);
    }
    let activeSource = null;

    const messages = document.getElementById("messages");
    const composer = document.getElementById("composer");
    const input = document.getElementById("message");
    const send = document.getElementById("send");
    const newChat = document.getElementById("newChat");
    const newWorkspace = document.getElementById("newWorkspace");
    const workspaceInput = document.getElementById("workspaceInput");
    const conversationSelect = document.getElementById("conversationSelect");
    const statusEl = document.getElementById("status");
    const traceToggle = document.getElementById("trace");
    const traceOut = document.getElementById("traceOut");
    const sessionLabel = document.getElementById("session");
    const traceTab = document.getElementById("traceTab");
    const mapTab = document.getElementById("mapTab");
    const tracePanel = document.getElementById("tracePanel");
    const mapPanel = document.getElementById("mapPanel");
    const formulationMap = document.getElementById("formulationMap");
    const policySummary = document.getElementById("policySummary");
    const mapSummary = document.getElementById("mapSummary");
    const mirrorMap = document.getElementById("mirrorMap");
    const progressToast = document.getElementById("progressToast");
    const progressLabel = document.getElementById("progressLabel");
    const progressPercent = document.getElementById("progressPercent");
    const progressBar = document.getElementById("progressBar");
    let progressHideTimer = null;
    let progressAnchor = null;
    let formulationGraph = { turn_count: 0, nodes: [], edges: [] };
    let policyMemory = { empty: true };

    const progressStages = {
      connecting: { label: "Connecting", value: 8 },
      structured_extraction: { label: "Reading message", value: 28 },
      therapeutic_kernel: { label: "Reasoning through options", value: 55 },
      response_plan: { label: "Planning response", value: 76 },
      rendering: { label: "Preparing response", value: 92 },
      complete: { label: "Complete", value: 100 },
      error: { label: "Generation interrupted", value: 100 },
    };

    function persistSession() {
      localStorage.setItem(userKey, userId);
      localStorage.setItem(workspaceKey, workspaceId);
      if (sessionId) {
        localStorage.setItem(conversationKey, sessionId);
        localStorage.setItem(legacySessionKey, sessionId);
      } else {
        localStorage.removeItem(conversationKey);
        localStorage.removeItem(legacySessionKey);
      }
    }

    function renderSessionLabel() {
      const conversationLabel = sessionId ? sessionId.slice(0, 8) : "select";
      sessionLabel.textContent = `${workspaceId.slice(0, 16)} / ${conversationLabel}`;
      workspaceInput.value = workspaceId;
    }

    function renderConversations(conversations = []) {
      const existing = new Set();
      conversationSelect.textContent = "";
      for (const conversation of conversations) {
        const id = conversation.conversation_id || conversation.session_id;
        if (!id || existing.has(id)) continue;
        existing.add(id);
        const option = document.createElement("option");
        option.value = id;
        const count = Number(conversation.message_count || 0);
        const suffix = count ? ` (${count})` : "";
        option.textContent = `${conversation.title || `Conversation ${id.slice(0, 8)}`}${suffix}`;
        conversationSelect.appendChild(option);
      }
      if (sessionId && !existing.has(sessionId)) {
        const option = document.createElement("option");
        option.value = sessionId;
        option.textContent = `New conversation ${sessionId.slice(0, 8)}`;
        conversationSelect.appendChild(option);
      }
      if (sessionId) conversationSelect.value = sessionId;
    }

    function scopedParams(extra = {}) {
      const params = {
        user_id: userId,
        workspace_id: workspaceId,
        ...extra,
      };
      if (sessionId) {
        params.conversation_id = sessionId;
        params.session_id = sessionId;
      }
      return new URLSearchParams(params);
    }

    function scopedBody(extra = {}) {
      const body = {
        user_id: userId,
        workspace_id: workspaceId,
        ...extra,
      };
      if (sessionId) {
        body.conversation_id = sessionId;
        body.session_id = sessionId;
      }
      return body;
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function showInspector(view) {
      const showMap = view === "map";
      tracePanel.hidden = showMap;
      mapPanel.hidden = !showMap;
      traceTab.classList.toggle("active", !showMap);
      mapTab.classList.toggle("active", showMap);
      traceTab.setAttribute("aria-selected", String(!showMap));
      mapTab.setAttribute("aria-selected", String(showMap));
    }

    function updateProgress(stage) {
      const state = progressStages[stage] || progressStages.connecting;
      if (progressHideTimer) {
        clearTimeout(progressHideTimer);
        progressHideTimer = null;
      }
      progressToast.hidden = false;
      progressLabel.textContent = state.label;
      progressPercent.textContent = `${state.value}%`;
      progressBar.style.width = `${state.value}%`;
      placeProgressToast();
      messages.scrollTop = messages.scrollHeight;
    }

    function placeProgressToast() {
      if (progressAnchor && progressAnchor.parentElement === messages) {
        progressAnchor.insertAdjacentElement("afterend", progressToast);
        return;
      }
      messages.appendChild(progressToast);
    }

    function finishProgress() {
      updateProgress("complete");
      progressHideTimer = setTimeout(() => {
        progressToast.hidden = true;
        progressBar.style.width = "0%";
        progressPercent.textContent = "0%";
      }, 700);
    }

    function failProgress() {
      updateProgress("error");
      progressHideTimer = setTimeout(() => {
        progressToast.hidden = true;
      }, 1600);
    }

    function resetProgress() {
      if (progressHideTimer) {
        clearTimeout(progressHideTimer);
        progressHideTimer = null;
      }
      progressToast.hidden = true;
      progressBar.style.width = "0%";
      progressPercent.textContent = "0%";
      progressAnchor = null;
    }

    function renderMarkdownInto(node, text) {
      node.innerHTML = markdownToHtml(text);
    }

    function markdownToHtml(source) {
      const lines = String(source || "").replaceAll("\r\n", "\n").split("\n");
      const blocks = [];
      let paragraph = [];
      let listType = null;
      let listItems = [];
      let inCode = false;
      let codeLines = [];

      function flushParagraph() {
        if (!paragraph.length) return;
        blocks.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br>")}</p>`);
        paragraph = [];
      }

      function flushList() {
        if (!listType) return;
        blocks.push(`<${listType}>${listItems.join("")}</${listType}>`);
        listType = null;
        listItems = [];
      }

      function closeCodeBlock() {
        blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        inCode = false;
        codeLines = [];
      }

      for (const line of lines) {
        if (inCode) {
          if (line.trim().startsWith("```")) {
            closeCodeBlock();
          } else {
            codeLines.push(line);
          }
          continue;
        }

        if (line.trim().startsWith("```")) {
          flushParagraph();
          flushList();
          inCode = true;
          codeLines = [];
          continue;
        }

        if (!line.trim()) {
          flushParagraph();
          flushList();
          continue;
        }

        const heading = line.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
          flushParagraph();
          flushList();
          const level = heading[1].length;
          blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
          continue;
        }

        const unordered = line.match(/^\s*[-*]\s+(.+)$/);
        const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
        if (unordered || ordered) {
          flushParagraph();
          const type = unordered ? "ul" : "ol";
          if (listType && listType !== type) flushList();
          listType = type;
          listItems.push(`<li>${renderInlineMarkdown((unordered || ordered)[1])}</li>`);
          continue;
        }

        flushList();
        paragraph.push(line);
      }

      flushParagraph();
      flushList();
      if (inCode) closeCodeBlock();
      return blocks.join("");
    }

    function renderInlineMarkdown(source) {
      let text = escapeHtml(source);
      const codeSpans = [];
      text = text.replace(/`([^`]+)`/g, (_match, code) => {
        codeSpans.push(`<code>${code}</code>`);
        return `\u0000CODE${codeSpans.length - 1}\u0000`;
      });
      text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label, url) => {
        const safeUrl = sanitizeMarkdownUrl(url);
        if (!safeUrl) return label;
        return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${label}</a>`;
      });
      text = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
      text = text.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
      text = text.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
      text = text.replace(/_([^_\n]+)_/g, "<em>$1</em>");
      return text.replace(/\u0000CODE(\d+)\u0000/g, (_match, index) => codeSpans[Number(index)] || "");
    }

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function sanitizeMarkdownUrl(value) {
      const url = String(value || "").trim().replaceAll("&amp;", "&");
      if (!/^(https?:|mailto:)/i.test(url)) return "";
      return escapeHtml(url);
    }

    function addMessage(role, text, meta = {}) {
      const wrap = document.createElement("div");
      wrap.className = `message-wrap ${role}`;
      wrap.experiment = meta.experiment || null;
      if (role === "reflection" || role === "info") {
        const isInfo = role === "info";
        const box = document.createElement("div");
        box.className = isInfo ? "info-box" : "reflection-box";
        const label = document.createElement("div");
        label.className = isInfo ? "info-label" : "reflection-label";
        label.textContent = isInfo ? "Framework note" : "Reflective listening";
        const body = document.createElement("div");
        body.className = `${isInfo ? "info-text" : "reflection-text"} markdown-body`;
        renderMarkdownInto(body, text);
        box.appendChild(label);
        box.appendChild(body);
        wrap.appendChild(box);
        messages.appendChild(wrap);
        messages.scrollTop = messages.scrollHeight;
        return wrap;
      }
      const node = document.createElement("div");
      node.className = `message ${role}${role === "coach" ? " markdown-body" : ""}`;
      if (role === "coach") {
        renderMarkdownInto(node, text);
      } else {
        node.textContent = text;
      }
      wrap.appendChild(node);
      if (role === "coach" && (meta.experiment || meta.trace_available)) {
        const actions = document.createElement("div");
        actions.className = "message-actions";
        if (meta.experiment) {
          const experimentChip = document.createElement("button");
          experimentChip.className = "why-chip experiment-chip";
          experimentChip.type = "button";
          experimentChip.dataset.experimentToggle = "1";
          experimentChip.textContent = experimentChipLabel(meta.experiment);
          experimentChip.addEventListener("click", () => toggleExperiment(wrap, experimentChip));
          actions.appendChild(experimentChip);
        }
        if (meta.trace_available) {
          const chip = document.createElement("button");
          chip.className = "why-chip";
          chip.type = "button";
          chip.textContent = "Why this?";
          chip.addEventListener("click", () => toggleExplanation(wrap, chip, meta));
          actions.appendChild(chip);
          if (meta.explanation) {
            renderExplanation(wrap, meta.explanation);
            chip.textContent = "Hide why";
          }
        }
        wrap.appendChild(actions);
      }
      messages.appendChild(wrap);
      messages.scrollTop = messages.scrollHeight;
      return wrap;
    }

    function experimentChipLabel(experiment) {
      if (!experiment) return "One-turn test";
      if (experiment.outcome) return `One-turn test · ${humanizeMapLabel(experiment.outcome)}`;
      if (experiment.status && experiment.status !== "proposed") return `One-turn test · ${humanizeMapLabel(experiment.status)}`;
      return "One-turn test";
    }

    function toggleExperiment(wrap, chip) {
      const existing = wrap.querySelector(".experiment-card");
      if (existing && !existing.hidden) {
        existing.hidden = true;
        chip.textContent = experimentChipLabel(wrap.experiment);
        return;
      }
      const card = renderExperimentCard(wrap, wrap.experiment);
      if (card) {
        card.hidden = false;
        chip.textContent = "Hide test";
        messages.scrollTop = messages.scrollHeight;
      }
    }

    function renderTranscript(items) {
      messages.textContent = "";
      for (const item of items) {
        addMessage(displayRole(item), item.text, item);
      }
    }

    function displayRole(item) {
      if (item.role === "assistant") return "coach";
      if (item.role === "reflection") return "reflection";
      if (item.role === "info") return "info";
      return "user";
    }

    function renderExperimentCard(wrap, experiment) {
      if (!experiment || !experiment.id) return null;
      wrap.experiment = experiment;
      let card = wrap.querySelector(".experiment-card");
      if (!card) {
        card = document.createElement("div");
        card.className = "experiment-card";
        wrap.appendChild(card);
      }
      card.dataset.experimentId = experiment.id;
      card.textContent = "";

      const head = document.createElement("div");
      head.className = "experiment-head";
      const title = document.createElement("div");
      title.className = "experiment-title";
      title.textContent = experiment.title || "Tiny experiment";
      const status = document.createElement("span");
      status.className = "experiment-status";
      status.textContent = experiment.outcome
        ? humanizeMapLabel(experiment.outcome)
        : humanizeMapLabel(experiment.status || "proposed");
      head.appendChild(title);
      head.appendChild(status);
      card.appendChild(head);

      const rows = document.createElement("div");
      rows.className = "experiment-rows";
      rows.appendChild(experimentRow("Test", experiment.hypothesis));
      rows.appendChild(experimentRow("Try", `${experiment.action} (${experiment.timebox || "10 minutes"})`));
      rows.appendChild(experimentRow("Predict", experiment.prediction));
      rows.appendChild(experimentRow("Measure", experiment.measure));
      card.appendChild(rows);

      if (experiment.rationale) {
        const rationale = document.createElement("div");
        rationale.className = "experiment-rationale";
        rationale.textContent = experiment.learning || experiment.rationale;
        card.appendChild(rationale);
      }

      const actions = document.createElement("div");
      actions.className = "experiment-actions";
      const disabled = experiment.status && experiment.status !== "proposed";
      actions.appendChild(experimentActionButton("Done", "completed", disabled));
      actions.appendChild(experimentActionButton("Helped", "helped", disabled));
      actions.appendChild(experimentActionButton("Did not help", "did_not_help", disabled));
      actions.appendChild(experimentActionButton("Too hard", "too_hard", disabled));
      actions.appendChild(experimentActionButton("Skipped", "skipped", disabled));
      card.appendChild(actions);
      return card;
    }

    function experimentRow(label, value) {
      const row = document.createElement("div");
      row.className = "experiment-row";
      const labelNode = document.createElement("div");
      labelNode.className = "experiment-label";
      labelNode.textContent = label;
      const valueNode = document.createElement("div");
      valueNode.className = "experiment-value";
      valueNode.textContent = value || "";
      row.appendChild(labelNode);
      row.appendChild(valueNode);
      return row;
    }

    function experimentActionButton(label, action, disabled) {
      const button = document.createElement("button");
      button.className = "experiment-action";
      button.type = "button";
      button.dataset.experimentAction = action;
      button.textContent = label;
      button.disabled = Boolean(disabled);
      return button;
    }

    const focusGroups = [
      ["domain", "Domains"],
      ["concern", "Concerns"],
      ["task", "Tasks"],
      ["challenge", "Challenges"],
      ["stake", "Stakes"],
    ];
    const directionGroups = [
      ["value", "Values"],
      ["goal", "Goals"],
      ["objective", "Objectives"],
      ["project", "Projects"],
      ["key_result", "Key results"],
      ["next_action", "Next actions"],
      ["obstacle", "Obstacles"],
      ["implementation_intention", "If-then plans"],
      ["waiting_for", "Waiting for"],
      ["time_horizon", "Time horizon"],
      ["success_measure", "Measures"],
    ];
    const focusKinds = new Set(focusGroups.map(([kind]) => kind));
    const directionKinds = new Set(directionGroups.map(([kind]) => kind));

    const mapGroups = [
      ["longitudinal_pattern", "Multi-turn patterns"],
      ["hypothesis", "Loops & hypotheses"],
      ["belief", "Beliefs"],
      ["thought", "Thoughts"],
      ["emotion", "Emotions"],
      ["behavior", "Behaviors"],
      ["urge", "Urges"],
      ["intervention", "Recent moves"],
      ["feature", "Signals"],
      ["situation", "Situations"],
      ["consequence", "Consequences"],
    ];

    function renderFormulation(graph) {
      formulationGraph = graph || { turn_count: 0, nodes: [], edges: [] };
      const nodes = (formulationGraph.nodes || [])
        .filter((node) => !["archived", "rejected", "removed"].includes(node.status))
        .sort((a, b) => {
          const seen = (b.seen_count || 0) - (a.seen_count || 0);
          if (seen !== 0) return seen;
          return String(a.label).localeCompare(String(b.label));
        });
      const archivedCount = Number(formulationGraph.archived_node_count || 0);
      mapSummary.textContent = nodes.length
        ? `${nodes.length} active · turn ${formulationGraph.turn_count || 0}${archivedCount ? ` · ${archivedCount} archived` : ""}`
        : "Empty";
      mirrorMap.disabled = !nodes.length;
      formulationMap.textContent = "";
      if (!nodes.length) {
        const empty = document.createElement("div");
        empty.className = "map-empty";
        empty.textContent = "The working map will fill in as the coach sees recurring observations, hypotheses, and response moves.";
        formulationMap.appendChild(empty);
        return;
      }

      const focusSection = renderFocusOverview(nodes);
      if (focusSection) {
        formulationMap.appendChild(focusSection);
      }
      const directionSection = renderDirectionOverview(nodes);
      if (directionSection) {
        formulationMap.appendChild(directionSection);
      }

      const groupedKinds = new Set([...focusKinds, ...directionKinds]);
      for (const [kind, title] of mapGroups) {
        const groupNodes = nodes.filter((node) => node.kind === kind);
        if (!groupNodes.length) continue;
        groupedKinds.add(kind);
        formulationMap.appendChild(renderMapSection(title, groupNodes));
      }
      const otherNodes = nodes.filter((node) => !groupedKinds.has(node.kind));
      if (otherNodes.length) {
        formulationMap.appendChild(renderMapSection("Other", otherNodes));
      }
    }

    function renderPolicy(policy) {
      policyMemory = policy || { empty: true };
      policySummary.textContent = "";
      const helpful = policyMemory.helpful || [];
      const costly = policyMemory.costly || [];
      const corrections = policyMemory.map_feedback || [];
      if (policyMemory.empty || (!helpful.length && !costly.length && !corrections.length)) {
        policySummary.hidden = true;
        return;
      }
      policySummary.hidden = false;
      const heading = document.createElement("h4");
      heading.textContent = "Learning from feedback";
      policySummary.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "policy-grid";
      if (helpful.length) {
        grid.appendChild(renderPolicyRow("Try more", helpful.map((item) => item.description || humanizeMapLabel(item.intervention))));
      }
      if (costly.length) {
        grid.appendChild(renderPolicyRow("Use carefully", costly.map((item) => item.description || humanizeMapLabel(item.intervention))));
      }
      if (corrections.length) {
        grid.appendChild(renderPolicyRow("Map corrections", corrections.map((item) => `${humanizeMapLabel(item.action)} ${humanizeMapLabel(item.kind)}: ${item.label}`)));
      }
      policySummary.appendChild(grid);
    }

    function renderPolicyRow(title, values) {
      const row = document.createElement("div");
      row.className = "policy-row";
      const label = document.createElement("div");
      label.className = "policy-kind";
      label.textContent = title;
      row.appendChild(label);
      const chips = document.createElement("div");
      chips.className = "policy-chips";
      for (const value of values.slice(0, 4)) {
        const chip = document.createElement("span");
        chip.className = "policy-chip";
        chip.textContent = value;
        chips.appendChild(chip);
      }
      row.appendChild(chips);
      return row;
    }

    function renderFocusOverview(nodes) {
      const rows = [];
      for (const [kind, title] of focusGroups) {
        const groupNodes = nodes.filter((node) => node.kind === kind);
        if (groupNodes.length) {
          rows.push([title, groupNodes]);
        }
      }
      if (!rows.length) return null;

      const section = document.createElement("section");
      section.className = "map-section focus-section";
      const heading = document.createElement("h4");
      heading.textContent = "What this is about";
      section.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "focus-grid";
      for (const [title, groupNodes] of rows) {
        const row = document.createElement("div");
        row.className = "focus-row";
        const label = document.createElement("div");
        label.className = "focus-kind";
        label.textContent = title;
        row.appendChild(label);

        const chips = document.createElement("div");
        chips.className = "focus-chips";
        for (const node of groupNodes.slice(0, 4)) {
          const chip = document.createElement("span");
          chip.className = "focus-chip";
          chip.textContent = humanizeMapLabel(node.label);
          chips.appendChild(chip);
        }
        row.appendChild(chips);
        grid.appendChild(row);
      }
      section.appendChild(grid);
      return section;
    }

    function renderDirectionOverview(nodes) {
      const rows = [];
      for (const [kind, title] of directionGroups) {
        const groupNodes = nodes.filter((node) => node.kind === kind);
        if (groupNodes.length) {
          rows.push([title, groupNodes]);
        }
      }
      if (!rows.length) return null;

      const section = document.createElement("section");
      section.className = "map-section focus-section direction-section";
      const heading = document.createElement("h4");
      heading.textContent = "Direction";
      section.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "focus-grid";
      for (const [title, groupNodes] of rows) {
        const row = document.createElement("div");
        row.className = "focus-row";
        const label = document.createElement("div");
        label.className = "focus-kind";
        label.textContent = title;
        row.appendChild(label);

        const chips = document.createElement("div");
        chips.className = "focus-chips";
        for (const node of groupNodes.slice(0, 4)) {
          const chip = document.createElement("span");
          chip.className = "focus-chip";
          chip.textContent = humanizeMapLabel(node.label);
          chips.appendChild(chip);
        }
        row.appendChild(chips);
        grid.appendChild(row);
      }
      section.appendChild(grid);
      return section;
    }

    function renderMapSection(title, nodes) {
      const section = document.createElement("section");
      section.className = "map-section";
      const heading = document.createElement("h4");
      heading.textContent = title;
      section.appendChild(heading);
      const list = document.createElement("div");
      list.className = "map-items";
      for (const node of nodes) {
        list.appendChild(renderMapNode(node));
      }
      section.appendChild(list);
      return section;
    }

    function renderMapNode(node) {
      const item = document.createElement("div");
      item.className = `map-item ${node.status || "tentative"}`;
      item.dataset.nodeId = node.id;

      const label = document.createElement("div");
      label.className = "map-label";
      label.textContent = humanizeMapLabel(node.label);
      item.appendChild(label);

      const meta = document.createElement("div");
      meta.className = "map-meta";
      const confidence = Math.round((node.confidence || 0) * 100);
      meta.textContent = `${humanizeMapLabel(node.status || "tentative")} · seen ${node.seen_count || 1} · ${confidence}%`;
      item.appendChild(meta);

      const actions = document.createElement("div");
      actions.className = "map-actions";
      actions.appendChild(mapActionButton("Fits", "confirm"));
      actions.appendChild(mapActionButton("Not quite", "reject", "reject"));
      actions.appendChild(mapActionButton("Remove", "remove", "remove"));
      item.appendChild(actions);
      return item;
    }

    function mapActionButton(label, action, variant = "") {
      const button = document.createElement("button");
      button.className = `map-action ${variant}`.trim();
      button.type = "button";
      button.dataset.action = action;
      button.textContent = label;
      return button;
    }

    function humanizeMapLabel(value) {
      return String(value || "").replaceAll("_", " ");
    }

    async function sendFormulationFeedback(nodeId, action) {
      setStatus("Updating map");
      const response = await fetch("/api/formulation/feedback", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(scopedBody({
          node_id: nodeId,
          action,
        })),
      });
      if (!response.ok) {
        throw new Error("Unable to update the working map.");
      }
      const data = await response.json();
      renderFormulation(data.result.graph);
      renderPolicy(data.policy || policyMemory);
      setStatus("Ready");
    }

    async function sendExperimentFeedback(card, action) {
      const experimentId = card.dataset.experimentId;
      if (!experimentId) return;
      setStatus("Recording experiment");
      const response = await fetch("/api/experiments/feedback", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(scopedBody({
          experiment_id: experimentId,
          action,
        })),
      });
      if (!response.ok) {
        throw new Error("Unable to record the experiment outcome.");
      }
      const data = await response.json();
      const wrap = card.closest(".message-wrap");
      if (wrap && data.result && data.result.experiment) {
        renderExperimentCard(wrap, data.result.experiment);
        const chip = wrap.querySelector("button[data-experiment-toggle]");
        if (chip && card.hidden) {
          chip.textContent = experimentChipLabel(data.result.experiment);
        }
      }
      renderPolicy(data.policy || policyMemory);
      setStatus("Ready");
    }

    function updateReflectionMessage(wrap, text) {
      const body = wrap.querySelector(".reflection-text");
      if (body) renderMarkdownInto(body, text);
      messages.scrollTop = messages.scrollHeight;
    }

    async function requestReflectiveListening() {
      if (mirrorMap.disabled) return;
      mirrorMap.disabled = true;
      const pending = addMessage("reflection", "Reflecting the working map...");
      setStatus("Reflective listening");
      try {
        const params = scopedParams();
        const response = await fetch(`/api/formulation/mirror?${params}`);
        if (!response.ok) throw new Error("Unable to create reflective listening.");
        const data = await response.json();
        updateReflectionMessage(pending, (data.message && data.message.text) || data.mirror.text);
        setStatus("Ready");
      } catch (error) {
        updateReflectionMessage(pending, error.message || "Unable to create reflective listening.");
        setStatus("Error");
      } finally {
        const nodes = (formulationGraph.nodes || [])
          .filter((node) => !["archived", "rejected", "removed"].includes(node.status));
        mirrorMap.disabled = !nodes.length;
      }
    }

    function renderExplanation(wrap, text) {
      let panel = wrap.querySelector(".trace-explanation");
      if (!panel) {
        panel = document.createElement("div");
        panel.className = "trace-explanation";
        wrap.appendChild(panel);
      }
      panel.textContent = text;
      panel.hidden = false;
      messages.scrollTop = messages.scrollHeight;
      return panel;
    }

    async function toggleExplanation(wrap, chip, meta) {
      const panel = wrap.querySelector(".trace-explanation");
      if (panel && !panel.hidden) {
        panel.hidden = true;
        chip.textContent = "Why this?";
        return;
      }
      if (panel && panel.hidden) {
        panel.hidden = false;
        chip.textContent = "Hide why";
        messages.scrollTop = messages.scrollHeight;
        return;
      }

      chip.disabled = true;
      chip.textContent = "Loading...";
      setStatus("Explaining");
      try {
        const params = scopedParams({
          message_index: String(meta.index),
        });
        const response = await fetch(`/api/chat/explain?${params}`);
        if (!response.ok) throw new Error("Unable to explain this turn.");
        const data = await response.json();
        meta.explanation = data.explanation;
        renderExplanation(wrap, data.explanation);
        chip.textContent = "Hide why";
        setStatus("Ready");
      } catch (error) {
        renderExplanation(wrap, error.message || "Unable to explain this turn.");
        chip.textContent = "Why this?";
        setStatus("Error");
      } finally {
        chip.disabled = false;
      }
    }

    async function loadSession(options = {}) {
      setStatus("Loading");
      const params = scopedParams(options.preferExisting ? { prefer_existing: "1" } : {});
      const response = await fetch(`/api/chat/session?${params}`);
      if (!response.ok) {
        setStatus("Error");
        return;
      }
      const data = await response.json();
      userId = data.user_id || userId;
      workspaceId = data.workspace_id || workspaceId;
      sessionId = data.conversation_id || data.session_id || sessionId;
      persistSession();
      renderSessionLabel();
      renderConversations(data.conversations || []);
      renderTranscript(data.messages || []);
      renderFormulation(data.formulation || { turn_count: 0, nodes: [], edges: [] });
      renderPolicy(data.policy || { empty: true });
      setStatus("Ready");
    }

    async function startNewChat() {
      if (activeSource) {
        activeSource.close();
        activeSource = null;
      }
      sessionId = crypto.randomUUID();
      persistSession();
      renderSessionLabel();
      renderTranscript([]);
      traceOut.textContent = "{}";
      send.disabled = false;
      resetProgress();
      renderPolicy(policyMemory);
      await loadSession();
      input.focus();
    }

    async function switchWorkspace(nextWorkspaceId) {
      const cleaned = String(nextWorkspaceId || "").trim();
      if (!cleaned || cleaned === workspaceId) {
        workspaceInput.value = workspaceId;
        return;
      }
      if (activeSource) {
        activeSource.close();
        activeSource = null;
      }
      workspaceId = cleaned.slice(0, 128);
      sessionId = "";
      persistSession();
      renderSessionLabel();
      renderTranscript([]);
      traceOut.textContent = "{}";
      resetProgress();
      renderPolicy({ empty: true });
      await loadSession({ preferExisting: true });
      input.focus();
    }

    async function startNewWorkspace() {
      await switchWorkspace(`workspace-${crypto.randomUUID().slice(0, 8)}`);
    }

    async function switchConversation(nextConversationId) {
      const cleaned = String(nextConversationId || "").trim();
      if (!cleaned || cleaned === sessionId) {
        conversationSelect.value = sessionId;
        return;
      }
      if (activeSource) {
        activeSource.close();
        activeSource = null;
      }
      sessionId = cleaned.slice(0, 128);
      persistSession();
      renderSessionLabel();
      renderTranscript([]);
      traceOut.textContent = "{}";
      resetProgress();
      renderPolicy(policyMemory);
      await loadSession();
      input.focus();
    }

    function appendTrace(eventName, data) {
      if (!traceToggle.checked) return;
      let current = {};
      try {
        current = JSON.parse(traceOut.textContent || "{}");
      } catch {
        current = {};
      }
      current[eventName] = data;
      traceOut.textContent = JSON.stringify(current, null, 2);
    }

    newChat.addEventListener("click", startNewChat);
    newWorkspace.addEventListener("click", startNewWorkspace);
    conversationSelect.addEventListener("change", () => switchConversation(conversationSelect.value));
    workspaceInput.addEventListener("change", () => switchWorkspace(workspaceInput.value));
    workspaceInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        workspaceInput.blur();
      }
    });
    traceTab.addEventListener("click", () => showInspector("trace"));
    mapTab.addEventListener("click", () => showInspector("map"));
    mirrorMap.addEventListener("click", requestReflectiveListening);
    formulationMap.addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      const item = button.closest(".map-item");
      if (!item) return;
      button.disabled = true;
      try {
        await sendFormulationFeedback(item.dataset.nodeId, button.dataset.action);
      } catch (error) {
        setStatus("Error");
      } finally {
        button.disabled = false;
      }
    });
    messages.addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-experiment-action]");
      if (!button) return;
      const card = button.closest(".experiment-card");
      if (!card) return;
      button.disabled = true;
      try {
        await sendExperimentFeedback(card, button.dataset.experimentAction);
      } catch (error) {
        setStatus("Error");
      } finally {
        button.disabled = false;
      }
    });

    composer.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;

      progressAnchor = addMessage("user", text);
      input.value = "";
      input.focus();
      send.disabled = true;
      traceOut.textContent = "{}";
      setStatus("Connecting");
      updateProgress("connecting");

      const params = scopedParams({
        message: text,
        trace: traceToggle.checked ? "1" : "0",
      });
      const source = new EventSource(`/api/chat/stream?${params}`);
      activeSource = source;

      source.addEventListener("session", (event) => {
        const data = JSON.parse(event.data);
        userId = data.user_id || userId;
        workspaceId = data.workspace_id || workspaceId;
        sessionId = data.conversation_id || data.session_id || sessionId;
        persistSession();
        renderSessionLabel();
        renderConversations([{ conversation_id: sessionId, title: `Conversation ${sessionId.slice(0, 8)}`, active: true }]);
      });
      source.addEventListener("status", (event) => {
        const data = JSON.parse(event.data);
        setStatus(data.stage.replaceAll("_", " "));
        updateProgress(data.stage);
      });
      source.addEventListener("extraction", (event) => appendTrace("extraction", JSON.parse(event.data)));
      source.addEventListener("kernel", (event) => appendTrace("kernel", JSON.parse(event.data)));
      source.addEventListener("policy", (event) => {
        const data = JSON.parse(event.data);
        appendTrace("policy", data);
        renderPolicy(data.summary || policyMemory);
      });
      source.addEventListener("plan", (event) => appendTrace("plan", JSON.parse(event.data)));
      source.addEventListener("formulation", (event) => {
        const data = JSON.parse(event.data);
        appendTrace("formulation", data);
        renderFormulation(data.graph);
      });
      source.addEventListener("experiment", (event) => appendTrace("experiment", JSON.parse(event.data)));
      source.addEventListener("trace", (event) => appendTrace("trace", JSON.parse(event.data)));
      source.addEventListener("response", (event) => {
        const data = JSON.parse(event.data);
        updateProgress("rendering");
        const message = data.message || { role: "assistant" };
        addMessage(displayRole(message), data.text, message);
        if (data.conversations) renderConversations(data.conversations);
        if (data.policy) renderPolicy(data.policy);
        setStatus("Ready");
      });
      source.addEventListener("error", (event) => {
        try {
          const data = JSON.parse(event.data);
          addMessage("coach", `Error: ${data.detail}`);
        } catch {
          addMessage("coach", "Error: connection interrupted.");
        }
        setStatus("Error");
        failProgress();
        send.disabled = false;
        source.close();
        if (activeSource === source) activeSource = null;
      });
      source.addEventListener("done", () => {
        send.disabled = false;
        source.close();
        if (activeSource === source) activeSource = null;
        if (statusEl.textContent !== "Error") {
          setStatus("Ready");
          finishProgress();
        }
      });
    });

    renderSessionLabel();
    loadSession({ preferExisting: true });
  </script>
</body>
</html>
"""


app = create_app()


if __name__ == "__main__":
    raise SystemExit(main())
