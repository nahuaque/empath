"""Simple DeepSeek-backed Empath chat CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from .formulation import FormulationGraph, FormulationMirror, mirror_formulation
from .therapeutic_kernel import (
    CoachingState,
    TherapeuticReasoningKernel,
    _infer_state_features,
)


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_KEY_FILE = ".deepseek_api_key"
DEFAULT_AGENT_RETRY_ATTEMPTS = 3
DEFAULT_AGENT_RETRY_BACKOFF_SECONDS = 0.35
CONSULTATIVE_INTERVENTIONS = frozenset(
    {
        "consultative_problem_structuring",
        "concise_factual_answer",
        "advisory_recommendation",
        "instructional_explanation",
        "analytical_synthesis",
        "creative_ideation",
        "socratic_inquiry",
        "active_listening_repair",
    }
)
CONSULTATIVE_EXERCISE_TEMPLATES = frozenset(
    {
        "Clarify the objective, constraints, options, tradeoffs, and one practical next step.",
        "Answer directly in two to four concise sentences, then note that Empath's strongest area is coaching and emotional support if that angle would be useful.",
        "Give a recommendation, the rationale, the key tradeoff, and the next implementation step.",
        "Explain the concept briefly, give one example, and name one common mistake.",
        "State the decision criteria, compare the options, and give a provisional conclusion.",
        "Generate a small set of options, cluster them, and suggest one direction to refine.",
        "Ask one question that tests the objective, assumption, evidence, or constraint.",
        "Reflect the frustration without defensiveness, keep unconditional positive regard, and offer Socratic inquiry, emotional support, or coaching expertise if desired.",
    }
)
ConversationMode = Literal[
    "coaching_turn",
    "consultative_turn",
    "framework_note",
    "reflective_listening",
    "map_correction",
    "workspace_command",
    "interaction_repair",
    "safety_support",
]


MODE_LABELS: dict[str, str] = {
    "coaching_turn": "Coaching",
    "consultative_turn": "Consultative",
    "framework_note": "Framework note",
    "reflective_listening": "Reflective listening",
    "map_correction": "Map correction",
    "workspace_command": "Workspace command",
    "interaction_repair": "Repair",
    "safety_support": "Safety",
}


EXTRACTION_INSTRUCTIONS = """\
You extract structured coaching observations from one user message.

Return only observations that are stated or strongly implied. Do not diagnose.
Use empty lists when a field is unknown. Phrase thoughts and beliefs as concise
first-person or user-centered statements.
Capture all major clauses in the message. Do not drop identity-level clauses
("I'm not cut out", "I'm worthless"), mind-reading clauses ("they'll think..."),
demandingness ("I should/must..."), or avoidance/procrastination behavior just
because another simpler observation is available.
Also extract what the emotional state is about:
- concerns: named subjects the user is emotionally reacting to
- tasks: concrete actions or work items in view
- challenges: obstacles, difficulties, blockers, or friction points
- objectives: desired outcomes, goals, or intended results
- stakes: what feels at risk or why the situation matters
- domains: broad life/work areas such as work, relationship, health, identity,
  family, money, school, creativity, or personal growth
- projects: ongoing commitments or multi-step outcomes
- key_results: measurable evidence of progress toward an objective
- next_actions: concrete visible actions the user could do next
- obstacles: predictable internal or external blockers to follow-through
- implementation_intentions: if-then plans already stated by the user
- waiting_for: items blocked by another person, event, information, or decision
- time_horizons: timing windows such as today, this week, this month, 12 weeks
- success_measures: observable signs that progress or completion happened

Canonical emotion labels:
- sadness
- anxiety
- shame
- anger
- overwhelm

Canonical behavior labels:
- avoidance
- procrastination
- withdrawal
- rumination
- inaction

Canonical feature labels for thoughts and beliefs:
- future_disaster
- awful_outcome
- mind_reading_claim
- binary_evaluation
- single_event_global_conclusion
- global_label
- demanding_rule
- unbearable_claim
- identity_global_rating
- identity_fusion
- sticky_thought
- discounting_positive_claim
- feeling_as_fact
- personal_responsibility_claim
- negative_filter
- approval_demand
- certainty_demand_claim
- failure_intolerance_claim
- control_struggle
- symbolization_need
- inner_critic_claim

Canonical state feature labels:
- high_distress
- needs_validation
- safety_risk
- trauma_content
- present_moment_disconnection
- values_unclear
- emotion_dysregulation
- crisis_urge
- interpersonal_conflict
- boundary_difficulty
- self_invalidation
- goal_setting
- motivation_block
- self_efficacy_doubt
- decision_uncertainty
- certainty_demand
- attention_environment
- setback_recovery
- integration_review
- recurring_pattern
- open_loops
- overloaded_open_loops
- wip_overload
- obstacle_planning
- scope_too_large
- review_due
- success_measure_unclear
- control_struggle
- vulnerability_factors
- approval_threat
- stress_load
- body_tension
- autopilot_reactivity
- felt_sense
- unclear_felt_sense
- hard_to_name
- symbolization_needed
- inner_critic
- felt_shift
- consultative_request
- factual_question
- advisory_request
- instructional_request
- analytical_request
- creative_ideation_request
- socratic_exploration_request
- empath_directed_aggression

Distress is an estimated 0-10 intensity when enough evidence is present. Use
null when there is not enough evidence.
Use high_distress only for distress 7-10 or clear panic/overwhelm language
such as "panicking", "can't breathe", "can't cope", or "unbearable".
"""


RESPONSE_PLAN_INSTRUCTIONS = """\
You create a structured response plan for Empath. You are not a medical
provider and you do not diagnose.

Use the supplied structured extraction and reasoning kernel output as tentative
planning context. Treat ACT/CBT/REBT/DBT/MBSR/Focusing labels as hypotheses,
not facts about the user. Do not say "you are catastrophizing" or "you are
fused"; use soft phrasing such as "this may involve..." or "one possible frame
is...".

Plan constraints:
- start with a brief validation or reflection
- use at most one or two intervention moves
- prefer concrete next steps over long explanation
- ask at most one clear follow-up question
- use concerns, tasks, challenges, objectives, stakes, and domains to keep the
  response anchored in what the user's emotion is about
- use goal-direction fields when present: objective, project, next action,
  obstacle, time horizon, and success measure
- for goal-direction moves, keep the framework lightweight: prefer one next
  action, one if-then plan, or one review question over a full productivity
  system
- when a concrete task/objective is present, make the exercise name that
  task/objective directly instead of offering a generic technique
- when stakes are present, validate the pressure lightly without amplifying
  threat or treating the stakes as objectively true
- when only minimal disclosure is present, do not invent a task, cause, or
  objective; use a gentle check-in instead
- prefer the highest-ranked non-validation kernel candidate after the opening
  validation, unless there is a clear reason to stay with validation/check-in
- when the kernel provides intervention recipes, use them as sequencing
  guidance; the selected intervention should usually be the main active step
  within the best safe recipe, not necessarily the opening validation step
- when the kernel provides differential formulations, treat them as competing
  maps; prefer a move that fits the strongest safe formulation, and use its
  discriminating question when the map is genuinely uncertain
- when the kernel provides clarifying_moves, use the top move as the next
  question if it would reduce uncertainty between plausible formulations
- when the kernel provides intervention_deliberation, use the selected
  intervention as the preferred active move and treat the alternatives as
  counterfactual checks, unless safety or plan coherence requires a change
- when the kernel provides counterfactual_simulation, treat it as a compact
  record of which moves were considered, what each was expected to change, and
  why the selected move won
- when adaptive policy memory is supplied, treat it as user-specific outcome
  evidence: lightly reuse moves that helped, shrink or soften moves that were
  too hard, and avoid over-trusting a single feedback event
- when the kernel indicates consultative facilitation, do not force a
  therapeutic lens; clarify the objective, structure the problem, give concise
  guidance, and invite a coaching/emotional-support angle only if useful
- for factual questions, answer directly and briefly without infodumping; if
  the topic is far from coaching or emotional support, mention that Empath's
  strongest area is coaching and emotional support in at most one sentence
- for advisory requests, use recommendation, rationale, tradeoff, and next step
- for instructional requests, use concept, example, common mistake, and
  application
- for analytical requests, use criteria, evidence or comparison, conclusion,
  and confidence level
- for creative requests, generate options, cluster or evaluate them, then
  suggest a direction to refine
- for Socratic exploration, ask one question that tests the objective,
  assumption, evidence, or constraint
- for verbal aggression toward Empath, do not defend, scold, or mirror
  hostility; practice active listening with unconditional positive regard, then
  offer Socratic inquiry, emotional support, or coaching expertise if desired
- in consultative mode, the exercise field may contain the concise answer,
  recommendation, explanation, or facilitation content rather than a
  therapeutic exercise
- in consultative mode, instantiate the content for the user's actual request;
  do not echo meta-instructions like "answer directly" or "state the criteria"
- do not include safety_note unless there is an actual safety concern
- if you use the question field, make the exercise an instruction, not another question
- if the kernel indicates safety risk, prioritize immediate safety/support and
  avoid cognitive disputation
- keep the response conversational, concise, and warm without being effusive
- return a structured response plan, not the final prose response
"""


MIRROR_INSTRUCTIONS = """\
You mirror back a user's working formulation from a coaching app.

Use the supplied formulation graph as tentative context, not as truth. Write in
the voice of the coach directly to the user. The psychological move is
reflective listening / accurate empathy / perspective taking.

Constraints:
- do not diagnose
- do not say the user "is" a pattern; say "I may be hearing..." or "it looks like..."
- do not introduce facts not present in the graph
- do not give advice, exercises, or a new intervention
- include a clear invitation to correct the map
- keep it warm, precise, and concise
- 2 to 5 short paragraphs
"""


class TextFeatureExtraction(BaseModel):
    """Canonical kernel features attached to one extracted text observation."""

    text: str = Field(description="Exact thought or belief text this feature set describes.")
    features: tuple[str, ...] = Field(default_factory=tuple)


class ExtractedCoachingState(BaseModel):
    """Structured observations produced by the extraction agent."""

    situations: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Specific external or internal contexts named by the user.",
    )
    concerns: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Subjects the user's emotional state appears to be about.",
    )
    tasks: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Concrete actions, work items, or responsibilities in view.",
    )
    challenges: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Obstacles, blockers, difficulties, friction, or pain points.",
    )
    objectives: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Desired outcomes, intended results, or success targets.",
    )
    projects: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ongoing commitments or multi-step outcomes named by the user.",
    )
    key_results: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Measurable evidence of progress toward an objective.",
    )
    next_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Concrete visible actions the user could do next.",
    )
    obstacles: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Predictable internal or external blockers to follow-through.",
    )
    implementation_intentions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="If-then action plans already stated by the user.",
    )
    waiting_for: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Items blocked by another person, event, information, or decision.",
    )
    time_horizons: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Timing windows such as today, this week, this month, or 12 weeks.",
    )
    success_measures: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Observable signs that progress or completion happened.",
    )
    stakes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="What feels at risk, important, or consequential.",
    )
    domains: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Broad life/work areas such as work, relationship, health, identity, family, money, school, creativity, or growth.",
    )
    thoughts: tuple[str, ...] = Field(default_factory=tuple)
    beliefs: tuple[str, ...] = Field(default_factory=tuple)
    emotions: tuple[str, ...] = Field(default_factory=tuple)
    bodily_sensations: tuple[str, ...] = Field(default_factory=tuple)
    urges: tuple[str, ...] = Field(default_factory=tuple)
    behaviors: tuple[str, ...] = Field(default_factory=tuple)
    consequences: tuple[str, ...] = Field(default_factory=tuple)
    values: tuple[str, ...] = Field(default_factory=tuple)
    goals: tuple[str, ...] = Field(default_factory=tuple)
    distress: int | None = Field(default=None, ge=0, le=10)
    features: tuple[str, ...] = Field(default_factory=tuple)
    thought_features: tuple[TextFeatureExtraction, ...] = Field(default_factory=tuple)
    belief_features: tuple[TextFeatureExtraction, ...] = Field(default_factory=tuple)


class ResponsePlan(BaseModel):
    """Structured response plan produced before final verbalization."""

    validation: str = Field(
        description="A brief validating reflection of the user's experience."
    )
    hypothesis: str | None = Field(
        default=None,
        description=(
            "A tentative, softly worded therapeutic frame. Avoid diagnostic wording."
        ),
    )
    intervention: str | None = Field(
        default=None,
        description="The primary intervention to use, preferably one kernel candidate id.",
    )
    exercise: str | None = Field(
        default=None,
        description=(
            "A small concrete exercise or next action, or consultative answer "
            "content when the selected mode is non-therapeutic."
        ),
    )
    question: str | None = Field(
        default=None,
        description="At most one follow-up question for the user.",
    )
    tone_constraints: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tone constraints for rendering the final response.",
    )
    safety_note: str | None = Field(
        default=None,
        description="Safety-focused note only when an actual safety risk is present.",
    )


class ConversationModeRoute(BaseModel):
    """First-pass route for deciding how a user turn should be handled."""

    mode: ConversationMode
    label: str
    reason: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = Field(default_factory=tuple)


class CounterfactualIntervention(BaseModel):
    """Compact simulation of an intervention path considered by the kernel."""

    intervention: str
    role: Literal["selected", "runner_up", "not_chosen"]
    score: float | None = None
    supported_by: tuple[str, ...] = Field(default_factory=tuple)
    expected_shift: str
    risk: str | None = None
    why: str
    test_signal: str
    exercise: str | None = None


class MirrorResponse(BaseModel):
    """LLM-generated reflection of the working formulation."""

    text: str = Field(
        description=(
            "A concise reflective-listening summary of the formulation, phrased "
            "tentatively and inviting correction."
        )
    )


@dataclass(frozen=True)
class PreparedTurn:
    """The extraction and deterministic reasoning passed into the response LLM."""

    route: ConversationModeRoute
    extraction: ExtractedCoachingState
    state: CoachingState
    kernel_snapshot: dict[str, Any]
    extraction_prompt: str
    response_prompt: str
    local_context: str = ""
    longitudinal_context: str = ""
    memory_context: str = ""
    memory_packet: dict[str, Any] | None = None
    policy_context: str = ""
    policy_report: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChatTurnResult:
    """A completed model turn plus inspectable deterministic context."""

    text: str
    response_plan: ResponsePlan
    prepared: PreparedTurn
    plan_coherence: dict[str, Any]
    message_history: list[ModelMessage]


def _run_agent_sync_with_retries(
    agent: Any,
    prompt: str,
    *,
    attempts: int = DEFAULT_AGENT_RETRY_ATTEMPTS,
    backoff_seconds: float = DEFAULT_AGENT_RETRY_BACKOFF_SECONDS,
    **kwargs: Any,
) -> Any:
    """Run a Pydantic AI agent with bounded retries for transient provider errors."""

    attempts = max(1, attempts)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return agent.run_sync(prompt, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            time.sleep(backoff_seconds * (2**attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Agent call failed without an exception.")


class KernelGuidedCoach:
    """Runs extraction, the therapeutic kernel, then response planning."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 900,
    ) -> None:
        self.kernel = TherapeuticReasoningKernel()
        self._turn_index = 0
        model = OpenAIChatModel(
            model_name,
            provider=DeepSeekProvider(api_key=api_key),
        )
        self.extractor_agent = Agent(
            model,
            output_type=ExtractedCoachingState,
            instructions=EXTRACTION_INSTRUCTIONS,
            model_settings={
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
        )
        self.response_agent = Agent(
            model,
            output_type=ResponsePlan,
            instructions=RESPONSE_PLAN_INSTRUCTIONS,
            model_settings={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        self.mirror_agent = Agent(
            model,
            output_type=MirrorResponse,
            instructions=MIRROR_INSTRUCTIONS,
            model_settings={
                "temperature": 0.35,
                "max_tokens": max_tokens,
            },
        )

    def prepare_turn(
        self,
        user_message: str,
        *,
        route: ConversationModeRoute | None = None,
        recent_interventions: tuple[str, ...] = (),
        local_context: str = "",
        longitudinal_context: str = "",
        memory_context: str = "",
        memory_packet: dict[str, Any] | None = None,
        policy_context: str = "",
    ) -> PreparedTurn:
        self._turn_index += 1
        route = route or route_conversation_mode(user_message)
        extraction_prompt = build_extraction_prompt(user_message)
        extraction_result = _run_agent_sync_with_retries(
            self.extractor_agent,
            extraction_prompt,
        )
        extraction = extraction_result.output
        state = state_from_extraction(
            extraction,
            user_message=user_message,
            state_id=f"turn-{self._turn_index}",
            recent_interventions=recent_interventions,
        )
        self.kernel.add_state(state)
        snapshot = with_intervention_deliberation(
            self.kernel.reasoning_snapshot(state.state_id, limit=5)
        )
        return PreparedTurn(
            route=route,
            extraction=extraction,
            state=state,
            kernel_snapshot=snapshot,
            longitudinal_context=longitudinal_context,
            extraction_prompt=extraction_prompt,
            response_prompt=build_response_prompt(
                user_message,
                extraction,
                snapshot,
                route=route,
                local_context=local_context,
                longitudinal_context=longitudinal_context,
                memory_context=memory_context,
                policy_context=policy_context,
            ),
            local_context=local_context,
            memory_context=memory_context,
            memory_packet=memory_packet,
            policy_context=policy_context,
        )

    def respond(
        self,
        user_message: str,
        *,
        message_history: list[ModelMessage] | None = None,
        recent_interventions: tuple[str, ...] = (),
        route: ConversationModeRoute | None = None,
        local_context: str = "",
        longitudinal_context: str = "",
        memory_context: str = "",
        memory_packet: dict[str, Any] | None = None,
        policy_context: str = "",
    ) -> ChatTurnResult:
        prepared = self.prepare_turn(
            user_message,
            route=route,
            recent_interventions=recent_interventions,
            local_context=local_context,
            longitudinal_context=longitudinal_context,
            memory_context=memory_context,
            memory_packet=memory_packet,
            policy_context=policy_context,
        )
        return self.complete_prepared_turn(
            prepared,
            message_history=message_history,
        )

    def complete_prepared_turn(
        self,
        prepared: PreparedTurn,
        *,
        message_history: list[ModelMessage] | None = None,
    ) -> ChatTurnResult:
        result = _run_agent_sync_with_retries(
            self.response_agent,
            prepared.response_prompt,
            message_history=message_history,
        )
        response_plan = sanitize_response_plan(
            result.output,
            kernel_snapshot=prepared.kernel_snapshot,
        )
        response_plan, plan_coherence = cohere_response_plan(
            response_plan,
            kernel_snapshot=prepared.kernel_snapshot,
            original_plan=result.output,
        )
        return ChatTurnResult(
            text=render_response_plan(response_plan),
            response_plan=response_plan,
            prepared=prepared,
            plan_coherence=plan_coherence,
            message_history=result.all_messages(),
        )

    def mirror_formulation(self, graph: FormulationGraph) -> FormulationMirror:
        draft = mirror_formulation(graph)
        result = _run_agent_sync_with_retries(
            self.mirror_agent,
            build_mirror_prompt(graph, draft),
        )
        return FormulationMirror(
            text=_clean_plan_text(result.output.text) or draft.text,
            graph_turn=graph.turn_count,
            node_ids=draft.node_ids,
        )


class DeterministicKernelGuidedCoach:
    """Kernel-guided coach implementation that never calls an LLM."""

    def __init__(self) -> None:
        self.kernel = TherapeuticReasoningKernel()
        self._turn_index = 0

    def prepare_turn(
        self,
        user_message: str,
        *,
        route: ConversationModeRoute | None = None,
        recent_interventions: tuple[str, ...] = (),
        local_context: str = "",
        longitudinal_context: str = "",
        memory_context: str = "",
        memory_packet: dict[str, Any] | None = None,
        policy_context: str = "",
    ) -> PreparedTurn:
        self._turn_index += 1
        route = route or route_conversation_mode(user_message)
        state = state_from_user_message(
            user_message,
            state_id=f"turn-{self._turn_index}",
            recent_interventions=recent_interventions,
        )
        self.kernel.add_state(state)
        extraction = extraction_from_state(state)
        snapshot = with_intervention_deliberation(
            self.kernel.reasoning_snapshot(state.state_id, limit=5)
        )
        return PreparedTurn(
            route=route,
            extraction=extraction,
            state=state,
            kernel_snapshot=snapshot,
            longitudinal_context=longitudinal_context,
            extraction_prompt=build_extraction_prompt(user_message),
            response_prompt=build_response_prompt(
                user_message,
                extraction,
                snapshot,
                route=route,
                local_context=local_context,
                longitudinal_context=longitudinal_context,
                memory_context=memory_context,
                policy_context=policy_context,
            ),
            local_context=local_context,
            memory_context=memory_context,
            memory_packet=memory_packet,
            policy_context=policy_context,
        )

    def complete_prepared_turn(
        self,
        prepared: PreparedTurn,
        *,
        message_history: list[ModelMessage] | None = None,
    ) -> ChatTurnResult:
        response_plan = draft_response_plan(
            prepared.kernel_snapshot,
            user_message=prepared.state.utterance,
        )
        response_plan, plan_coherence = cohere_response_plan(
            response_plan,
            kernel_snapshot=prepared.kernel_snapshot,
            original_plan=response_plan,
        )
        return ChatTurnResult(
            text=render_response_plan(response_plan),
            response_plan=response_plan,
            prepared=prepared,
            plan_coherence=plan_coherence,
            message_history=message_history or [],
        )

    def respond(
        self,
        user_message: str,
        *,
        message_history: list[ModelMessage] | None = None,
        recent_interventions: tuple[str, ...] = (),
        route: ConversationModeRoute | None = None,
        local_context: str = "",
        longitudinal_context: str = "",
        memory_context: str = "",
        memory_packet: dict[str, Any] | None = None,
        policy_context: str = "",
    ) -> ChatTurnResult:
        prepared = self.prepare_turn(
            user_message,
            route=route,
            recent_interventions=recent_interventions,
            local_context=local_context,
            longitudinal_context=longitudinal_context,
            memory_context=memory_context,
            memory_packet=memory_packet,
            policy_context=policy_context,
        )
        return self.complete_prepared_turn(
            prepared,
            message_history=message_history,
        )

    def mirror_formulation(self, graph: FormulationGraph) -> FormulationMirror:
        return mirror_formulation(graph)


def route_conversation_mode(user_message: str) -> ConversationModeRoute:
    """Classify the user turn before extraction and therapeutic reasoning."""

    text = _normalize_route_text(user_message).strip()
    lowered = f" {text.casefold()} "
    if _route_matches(
        lowered,
        r"\b(suicid|self-harm|self harm|hurt myself|kill myself|want to die|"
        r"end it all|not safe|can't stay safe|cant stay safe)\b",
    ):
        return ConversationModeRoute(
            mode="safety_support",
            label=MODE_LABELS["safety_support"],
            reason="The turn contains possible immediate safety-risk language.",
            confidence=0.95,
            evidence=_route_evidence(text, ("hurt myself", "kill myself", "want to die", "not safe")),
        )

    if _looks_like_framework_explanation_request(text):
        return ConversationModeRoute(
            mode="framework_note",
            label=MODE_LABELS["framework_note"],
            reason="The turn asks for an explanation of a coaching or therapeutic framework.",
            confidence=0.9,
            evidence=_route_evidence(
                text,
                ("explain", "what is", "ACT", "CBT", "REBT", "DBT", "MBSR", "GTD", "OKR", "WOOP"),
            ),
        )

    if _route_matches(
        lowered,
        r"\b(clear|reset|blank|wipe)\s+(the\s+)?(working\s+map|map|workspace)\b|"
        r"\b(new|switch|delete)\s+(conversation|workspace)\b",
    ):
        return ConversationModeRoute(
            mode="workspace_command",
            label=MODE_LABELS["workspace_command"],
            reason="The turn appears to be an app/workspace command rather than a coaching request.",
            confidence=0.82,
            evidence=_route_evidence(text, ("clear", "reset", "working map", "new conversation", "switch conversation")),
        )

    if _route_matches(
        lowered,
        r"\b(that'?s wrong|that is wrong|not true|incorrect|you got\b.*\bwrong|"
        r"remove\b.*\bfrom\s+(the\s+)?map|fix\s+(the\s+)?map|correct\s+(the\s+)?map|"
        r"map\s+is\s+wrong)\b",
    ):
        return ConversationModeRoute(
            mode="map_correction",
            label=MODE_LABELS["map_correction"],
            reason="The turn appears to correct the working map or a prior assistant inference.",
            confidence=0.82,
            evidence=_route_evidence(text, ("wrong", "not true", "incorrect", "fix the map", "remove")),
        )

    if _route_matches(
        lowered,
        r"\b(reflect back|mirror me|mirror back|reflective listening|"
        r"play back|perspective-take|perspective take)\b",
    ):
        return ConversationModeRoute(
            mode="reflective_listening",
            label=MODE_LABELS["reflective_listening"],
            reason="The turn requests a reflective playback rather than a new coaching intervention.",
            confidence=0.88,
            evidence=_route_evidence(text, ("reflect back", "mirror", "reflective listening", "play back", "perspective")),
        )

    if _route_matches(
        lowered,
        r"\b(you are useless|you're useless|youre useless|you suck|shut up|"
        r"fuck you|stupid bot|idiot|garbage assistant|terrible assistant|"
        r"worthless assistant|bad assistant|useless assistant)\b",
    ):
        return ConversationModeRoute(
            mode="interaction_repair",
            label=MODE_LABELS["interaction_repair"],
            reason="The turn directs frustration or aggression at Empath, so the response should repair the interaction without defensiveness.",
            confidence=0.86,
            evidence=_route_evidence(text, ("useless", "you suck", "shut up", "fuck you", "idiot", "bad assistant")),
        )

    features = _infer_state_features(text)
    consultative_features = {
        "consultative_request",
        "factual_question",
        "advisory_request",
        "instructional_request",
        "analytical_request",
        "creative_ideation_request",
        "socratic_exploration_request",
    }
    matched = tuple(sorted(features.intersection(consultative_features)))
    if matched:
        return ConversationModeRoute(
            mode="consultative_turn",
            label=MODE_LABELS["consultative_turn"],
            reason="The turn is asking for information, analysis, advice, or ideation rather than emotional processing.",
            confidence=0.78,
            evidence=matched,
        )

    return ConversationModeRoute(
        mode="coaching_turn",
        label=MODE_LABELS["coaching_turn"],
        reason="The turn appears to fit the standard coaching/emotional-support loop.",
        confidence=0.68,
        evidence=(),
    )


def route_for_response_plan(
    route: ConversationModeRoute,
    response_plan: ResponsePlan,
) -> ConversationModeRoute:
    """Adjust the visible mode when the final plan clearly escalates or specializes."""

    intervention = response_plan.intervention or ""
    if intervention == "safety_planning" and route.mode != "safety_support":
        return ConversationModeRoute(
            mode="safety_support",
            label=MODE_LABELS["safety_support"],
            reason="The response plan selected safety planning, which supersedes the first-pass route.",
            confidence=0.95,
            evidence=(intervention,),
        )
    if intervention == "active_listening_repair" and route.mode != "interaction_repair":
        return ConversationModeRoute(
            mode="interaction_repair",
            label=MODE_LABELS["interaction_repair"],
            reason="The response plan selected an interaction-repair move.",
            confidence=0.86,
            evidence=(intervention,),
        )
    if (
        route.mode == "coaching_turn"
        and is_consultative_intervention(intervention)
    ):
        return ConversationModeRoute(
            mode="consultative_turn",
            label=MODE_LABELS["consultative_turn"],
            reason="The kernel selected a consultative facilitation move.",
            confidence=0.78,
            evidence=(intervention,),
        )
    return route


def _route_matches(lowered_message: str, pattern: str) -> bool:
    return bool(re.search(pattern, lowered_message, flags=re.IGNORECASE))


def _normalize_route_text(message: str) -> str:
    return (
        message.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )


def _route_evidence(message: str, cues: tuple[str, ...]) -> tuple[str, ...]:
    lowered = message.casefold()
    return tuple(cue for cue in cues if cue.casefold() in lowered)[:4]


def _looks_like_framework_explanation_request(message: str) -> bool:
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


def read_api_key(path: Path) -> str:
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"DeepSeek API key file not found: {path}") from exc

    if not key:
        raise RuntimeError(f"DeepSeek API key file is empty: {path}")
    return key


def state_from_extraction(
    extraction: ExtractedCoachingState,
    *,
    user_message: str,
    state_id: str,
    recent_interventions: tuple[str, ...] = (),
) -> CoachingState:
    """Convert LLM-extracted observations into kernel facts."""

    situations = extraction.situations or (_compact_text(user_message),)
    thoughts = _with_raw_user_message(extraction.thoughts, user_message)
    values = extraction.values or _infer_values(user_message)
    goals = extraction.goals or _infer_goals(user_message)
    concerns = extraction.concerns or _infer_concerns(user_message)
    tasks = extraction.tasks or _infer_tasks(user_message)
    challenges = extraction.challenges or _infer_challenges(user_message)
    objectives = extraction.objectives or _infer_objectives(user_message, goals)
    projects = extraction.projects or _infer_projects(user_message, tasks, objectives)
    key_results = extraction.key_results or _infer_key_results(user_message)
    next_actions = extraction.next_actions or _infer_next_actions(user_message, tasks)
    obstacles = extraction.obstacles or _infer_obstacles(user_message, challenges)
    implementation_intentions = (
        extraction.implementation_intentions
        or _infer_implementation_intentions(user_message)
    )
    waiting_for = extraction.waiting_for or _infer_waiting_for(user_message)
    time_horizons = extraction.time_horizons or _infer_time_horizons(user_message)
    success_measures = extraction.success_measures or _infer_success_measures(
        user_message,
        key_results,
    )
    stakes = extraction.stakes or _infer_stakes(user_message)
    domains = extraction.domains or _infer_domains(user_message)
    return CoachingState(
        state_id=state_id,
        utterance=user_message,
        situations=_clean_tuple(situations),
        concerns=_clean_tuple(concerns),
        tasks=_clean_tuple(tasks),
        challenges=_clean_tuple(challenges),
        objectives=_clean_tuple(objectives),
        projects=_clean_tuple(projects),
        key_results=_clean_tuple(key_results),
        next_actions=_clean_tuple(next_actions),
        obstacles=_clean_tuple(obstacles),
        implementation_intentions=_clean_tuple(implementation_intentions),
        waiting_for=_clean_tuple(waiting_for),
        time_horizons=_clean_tuple(time_horizons),
        success_measures=_clean_tuple(success_measures),
        stakes=_clean_tuple(stakes),
        domains=_clean_tuple(domains),
        thoughts=_clean_tuple(thoughts),
        beliefs=_clean_tuple(extraction.beliefs),
        emotions=_clean_tuple(extraction.emotions),
        bodily_sensations=_clean_tuple(extraction.bodily_sensations),
        urges=_clean_tuple(extraction.urges),
        behaviors=_clean_tuple(extraction.behaviors),
        consequences=_clean_tuple(extraction.consequences),
        values=_clean_tuple(values),
        goals=_clean_tuple(goals),
        distress=extraction.distress,
        features=_normalized_state_features(
            extraction.features,
            distress=extraction.distress,
            user_message=user_message,
        ),
        thought_features=_feature_mapping(extraction.thought_features),
        belief_features=_feature_mapping(extraction.belief_features),
        recent_interventions=_clean_tuple(recent_interventions),
    )


def state_from_user_message(
    user_message: str,
    *,
    state_id: str,
    recent_interventions: tuple[str, ...] = (),
) -> CoachingState:
    """Deterministic fallback used for dry runs and tests."""

    emotions = _infer_emotions(user_message)
    values = _infer_values(user_message)
    goals = _infer_goals(user_message)
    distress = _infer_distress(user_message, emotions)
    tasks = _infer_tasks(user_message)
    objectives = _infer_objectives(user_message, goals)
    key_results = _infer_key_results(user_message)

    return CoachingState(
        state_id=state_id,
        utterance=user_message,
        situations=(_compact_text(user_message),),
        concerns=_infer_concerns(user_message),
        tasks=tasks,
        challenges=_infer_challenges(user_message),
        objectives=objectives,
        projects=_infer_projects(user_message, tasks, objectives),
        key_results=key_results,
        next_actions=_infer_next_actions(user_message, tasks),
        obstacles=_infer_obstacles(user_message, _infer_challenges(user_message)),
        implementation_intentions=_infer_implementation_intentions(user_message),
        waiting_for=_infer_waiting_for(user_message),
        time_horizons=_infer_time_horizons(user_message),
        success_measures=_infer_success_measures(user_message, key_results),
        stakes=_infer_stakes(user_message),
        domains=_infer_domains(user_message),
        thoughts=(user_message,),
        emotions=emotions,
        values=values,
        goals=goals,
        distress=distress,
        features=tuple(sorted(_infer_state_features(user_message))),
        recent_interventions=_clean_tuple(recent_interventions),
    )


def build_extraction_prompt(user_message: str) -> str:
    return (
        "Extract structured coaching observations from this user message:\n"
        f"{user_message}"
    )


def build_response_prompt(
    user_message: str,
    extraction: ExtractedCoachingState,
    kernel_snapshot: dict[str, Any],
    *,
    route: ConversationModeRoute | None = None,
    local_context: str = "",
    longitudinal_context: str = "",
    memory_context: str = "",
    policy_context: str = "",
) -> str:
    route = route or route_conversation_mode(user_message)
    prompt = (
        "User message:\n"
        f"{user_message}\n\n"
        "Conversation mode route:\n"
        f"{json.dumps(route.model_dump(), indent=2)}\n\n"
        "Structured extraction:\n"
        f"{json.dumps(extraction.model_dump(), indent=2)}\n\n"
        f"{_format_focus_context(extraction)}\n\n"
        "Therapeutic kernel output, including consultative fallback hypotheses, "
        "to use as tentative planning context:\n"
        f"{json.dumps(kernel_snapshot, indent=2)}\n\n"
    )
    if local_context.strip():
        prompt += (
            "Local conversation context, bounded to the last five user turns "
            "with intervening coach and info messages. Use this only for "
            "continuity; the structured extraction and kernel output are based "
            "only on the latest user message:\n"
            f"{local_context.strip()}\n\n"
        )
    if longitudinal_context.strip():
        prompt += (
            "Longitudinal session context, to use as tentative context only:\n"
            f"{longitudinal_context.strip()}\n\n"
        )
    if memory_context.strip():
        prompt += (
            "Retrieved workspace memory from the Surreal working map. Use this "
            "for continuity and user-specific learning; do not overrule the "
            "latest user message or reintroduce suppressed assumptions:\n"
            f"{memory_context.strip()}\n\n"
        )
    if policy_context.strip():
        prompt += (
            "Adaptive policy memory, to use as tentative user-specific outcome evidence:\n"
            f"{policy_context.strip()}\n\n"
        )
    return (
        prompt
        + "Create the structured response plan now. Honor the conversation mode "
        "route as the first-pass interaction contract unless safety evidence "
        "requires escalation. Choose interventions from the kernel candidates "
        "when possible."
    )


def with_intervention_deliberation(
    kernel_snapshot: dict[str, Any],
    *,
    limit: int = 4,
) -> dict[str, Any]:
    """Attach a compact counterfactual intervention deliberation to a snapshot."""

    snapshot = dict(kernel_snapshot)
    candidates = [dict(item) for item in snapshot.get("candidates") or ()]
    if not candidates:
        snapshot["intervention_deliberation"] = {
            "selected_intervention": None,
            "runner_up_intervention": None,
            "candidates_considered": [],
            "note": "No kernel candidates were available.",
        }
        snapshot["counterfactual_simulation"] = []
        return snapshot

    selected = _select_deliberated_candidate(candidates)
    runner_up = next(
        (
            candidate for candidate in candidates
            if candidate.get("intervention") != selected.get("intervention")
            and not candidate.get("contraindications")
        ),
        None,
    )
    rows = []
    for candidate in _ordered_deliberation_candidates(candidates, selected, runner_up, limit=limit):
        role = (
            "selected"
            if candidate.get("intervention") == selected.get("intervention")
            else "runner_up"
            if runner_up and candidate.get("intervention") == runner_up.get("intervention")
            else "not_chosen"
        )
        rows.append(
            _deliberation_row(
                candidate,
                selected=selected,
                role=role,
            )
        )

    snapshot["intervention_deliberation"] = {
        "selected_intervention": selected.get("intervention"),
        "runner_up_intervention": runner_up.get("intervention") if runner_up else None,
        "selected_reason": _deliberation_selected_reason(selected, candidates),
        "candidates_considered": rows,
        "note": (
            "Counterfactual intervention check: selected the preferred active move, "
            "then kept alternatives as reasons to avoid or revisit."
        ),
    }
    snapshot["counterfactual_simulation"] = [
        item.model_dump()
        for item in simulate_counterfactual_interventions(
            snapshot,
            selected_intervention=str(selected.get("intervention") or ""),
            limit=limit,
        )
    ]
    return snapshot


def simulate_counterfactual_interventions(
    kernel_snapshot: dict[str, Any],
    *,
    selected_intervention: str | None = None,
    limit: int = 4,
) -> tuple[CounterfactualIntervention, ...]:
    """Build a compact, user-facing counterfactual view over kernel candidates."""

    deliberation = kernel_snapshot.get("intervention_deliberation") or {}
    rows = [
        dict(item)
        for item in deliberation.get("candidates_considered") or ()
        if isinstance(item, dict) and item.get("intervention")
    ]
    if not rows:
        candidates = [
            dict(item)
            for item in kernel_snapshot.get("candidates") or ()
            if isinstance(item, dict) and item.get("intervention")
        ]
        selected_name = selected_intervention or (
            str(candidates[0].get("intervention")) if candidates else None
        )
        selected_candidate = next(
            (
                item for item in candidates
                if str(item.get("intervention")) == selected_name
            ),
            candidates[0] if candidates else {},
        )
        rows = []
        for index, candidate in enumerate(candidates[:limit]):
            rows.append(
                _deliberation_row(
                    candidate,
                    selected=selected_candidate,
                    role=(
                        "selected"
                        if str(candidate.get("intervention")) == selected_name
                        else "runner_up"
                        if index == 1
                        else "not_chosen"
                    ),
                )
            )

    selected_name = (
        selected_intervention
        or deliberation.get("selected_intervention")
        or (rows[0].get("intervention") if rows else None)
    )
    simulations = []
    for index, row in enumerate(rows[:limit]):
        intervention = str(row.get("intervention") or "")
        role = str(row.get("role") or "")
        if intervention == selected_name:
            role = "selected"
        elif role not in {"runner_up", "not_chosen"}:
            role = "runner_up" if index == 1 else "not_chosen"
        supported_by = _counterfactual_supported_patterns(row)
        risk = _first_text(row.get("risks"))
        if role == "selected" and not risk:
            risk = _selected_counterfactual_risk(intervention, supported_by)
        why = (
            str(row.get("decision") or "").strip()
            if role == "selected"
            else str(row.get("reason_not_chosen") or "").strip()
        ) or _default_counterfactual_reason(role, intervention)
        simulations.append(
            CounterfactualIntervention(
                intervention=intervention,
                role=role,  # type: ignore[arg-type]
                score=_optional_float(row.get("score")),
                supported_by=supported_by,
                expected_shift=_expected_shift_for_intervention(
                    intervention,
                    supported_by,
                ),
                risk=risk,
                why=why,
                test_signal=_test_signal_for_intervention(
                    intervention,
                    supported_by,
                ),
                exercise=_clean_plan_text(str(row.get("exercise") or ""))
                if row.get("exercise")
                else None,
            )
        )
    return tuple(simulations)


def _select_deliberated_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [candidate for candidate in candidates if not candidate.get("contraindications")]
    if not safe:
        return candidates[0]
    active = [
        candidate
        for candidate in safe
        if candidate.get("intervention") != "validation"
    ]
    return active[0] if active else safe[0]


def _ordered_deliberation_candidates(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
    runner_up: dict[str, Any] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = [selected]
    if runner_up:
        ordered.append(runner_up)
    for candidate in candidates:
        if candidate.get("intervention") in {
            item.get("intervention") for item in ordered
        }:
            continue
        ordered.append(candidate)
        if len(ordered) >= limit:
            break
    return ordered[:limit]


def _deliberation_row(
    candidate: dict[str, Any],
    *,
    selected: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    intervention = str(candidate.get("intervention") or "")
    possible_patterns = tuple(
        TherapeuticReasoningKernel().patterns_for_intervention(intervention)
    )
    supported_hypotheses = [
        dict(item)
        for item in candidate.get("hypotheses") or ()
        if isinstance(item, dict) and item.get("pattern")
    ]
    supported_patterns = _unique_text(
        str(item.get("pattern"))
        for item in supported_hypotheses
        if item.get("pattern")
    )
    missing_evidence = tuple(
        pattern for pattern in possible_patterns
        if pattern not in set(supported_patterns)
    )
    contraindications = tuple(str(item) for item in candidate.get("contraindications") or ())
    row = {
        "intervention": intervention,
        "role": role,
        "score": candidate.get("score"),
        "modality": tuple(candidate.get("modality") or ()),
        "would_fit_if": possible_patterns[:6],
        "supported_by": supported_hypotheses[:4],
        "missing_evidence": missing_evidence[:4],
        "risks": contraindications or _counterfactual_risks(candidate, role=role),
        "exercise": candidate.get("exercise"),
    }
    if role == "selected":
        row["decision"] = _deliberation_selected_reason(candidate, [candidate])
    else:
        row["reason_not_chosen"] = _counterfactual_not_chosen_reason(
            candidate,
            selected=selected,
            missing_evidence=missing_evidence,
            contraindications=contraindications,
        )
    return row


def _deliberation_selected_reason(
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    if selected.get("intervention") == "validation":
        return "Validation was the safest coherent move available."
    if candidates and candidates[0].get("intervention") == "validation":
        return "Chosen as the active move after an opening validation stance."
    return "Chosen as the highest-ranked safe active move."


def _counterfactual_risks(candidate: dict[str, Any], *, role: str) -> tuple[str, ...]:
    intervention = str(candidate.get("intervention") or "")
    if role == "selected":
        return ()
    if intervention == "validation":
        return ("may validate without creating enough movement",)
    if "disputation" in intervention:
        return ("may feel too sharp if shame or distress is high",)
    if "behavioral" in intervention or "committed_action" in intervention:
        return ("may jump to action before enough emotional contact",)
    if "defusion" in intervention:
        return ("may bypass practical execution if action is the main gap",)
    return ("less direct fit than the selected intervention",)


def _counterfactual_not_chosen_reason(
    candidate: dict[str, Any],
    *,
    selected: dict[str, Any],
    missing_evidence: tuple[str, ...],
    contraindications: tuple[str, ...],
) -> str:
    if contraindications:
        return "Blocked or mistimed: " + ", ".join(contraindications[:2])
    if candidate.get("intervention") == "validation" and selected.get("intervention") != "validation":
        return "Kept as the opening stance, not the main active step."
    if missing_evidence:
        return "Less complete match; missing " + _format_label_text(list(missing_evidence[:2])) + "."
    return f"Lower ranked than {str(selected.get('intervention')).replace('_', ' ')}."


def _counterfactual_supported_patterns(row: dict[str, Any]) -> tuple[str, ...]:
    patterns = []
    for item in row.get("supported_by") or ():
        if isinstance(item, dict) and item.get("pattern"):
            patterns.append(str(item.get("pattern")))
        elif item:
            patterns.append(str(item))
    return tuple(_unique_text(patterns)[:4])


def _first_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        for item in value:
            text = str(item).strip()
            if text:
                return text
    return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _selected_counterfactual_risk(
    intervention: str,
    supported_by: tuple[str, ...],
) -> str | None:
    if intervention == "validation":
        return "May not create enough movement if the user is ready for action."
    if "high_distress" in supported_by:
        return "Could still be mistimed if activation is higher than the text suggests."
    if intervention in {"committed_action", "acceptance_committed_action", "behavioral_experiment"}:
        return "Could feel premature if emotion or shame needs more contact first."
    if intervention in {"cognitive_defusion", "evidence_check", "rebt_disputation"}:
        return "Could stay too cognitive if the user mainly needs support or grounding."
    return None


def _default_counterfactual_reason(role: str, intervention: str) -> str:
    if role == "selected":
        return "Selected as the best safe fit among the coherent kernel candidates."
    if role == "runner_up":
        return f"Kept as a plausible alternative, but less direct than {intervention.replace('_', ' ')}."
    return "Considered coherent, but not the strongest current fit."


def _expected_shift_for_intervention(
    intervention: str,
    supported_by: tuple[str, ...],
) -> str:
    if intervention == "validation":
        return "The user feels understood enough to keep exploring."
    if intervention == "gentle_check_in":
        return "Minimal disclosure turns into one clearer next piece of context."
    if intervention == "present_moment_grounding":
        return "Activation drops enough for the next step to feel more workable."
    if intervention == "cognitive_defusion":
        return "The user has more distance from a sticky or identity-laden thought."
    if intervention == "evidence_check":
        return "Guesses, facts, and interpretations become easier to separate."
    if intervention == "decatastrophizing":
        return "The feared outcome becomes more specific and less totalizing."
    if intervention in {"rebt_disputation", "unconditional_self_acceptance"}:
        return "A rigid self-worth or demand belief loosens slightly."
    if intervention in {"committed_action", "acceptance_committed_action"}:
        return "Avoidance shifts into one small values-aligned action."
    if intervention == "behavioral_experiment":
        return "The belief becomes testable through a small real-world experiment."
    if intervention == "values_clarification":
        return "The user can name what matters underneath the difficulty."
    if intervention == "self_compassion":
        return "Shame softens enough to reduce self-attack."
    if intervention == "safety_planning":
        return "Immediate support and safety become clearer than problem-solving."
    if supported_by:
        return (
            "The response targets "
            f"{_format_label_text(list(supported_by[:2]))} in a more workable way."
        )
    return "The turn moves toward a clearer, more workable next step."


def _test_signal_for_intervention(
    intervention: str,
    supported_by: tuple[str, ...],
) -> str:
    if intervention == "validation":
        return "The user adds detail or says the reflection fits."
    if intervention == "gentle_check_in":
        return "The user can name what the feeling is about, even tentatively."
    if intervention == "present_moment_grounding":
        return "The user reports even a small drop in intensity or more steadiness."
    if intervention == "cognitive_defusion":
        return "The thought is described as a thought, not as settled fact."
    if intervention == "evidence_check":
        return "The user separates evidence from prediction or mind-reading."
    if intervention in {"committed_action", "acceptance_committed_action"}:
        return "The user identifies or tries one concrete next action."
    if intervention == "behavioral_experiment":
        return "The user can state a small test and what outcome would teach them."
    if intervention in {"rebt_disputation", "unconditional_self_acceptance"}:
        return "The user can weaken a must, awfulizing claim, or global self-rating."
    if intervention == "values_clarification":
        return "The user names a value, direction, or caring point."
    if intervention == "self_compassion":
        return "Self-criticism becomes less harsh or more contextual."
    if intervention == "safety_planning":
        return "The user states immediate safety status and a support option."
    if supported_by:
        return f"Look for a shift in {_format_label_text(list(supported_by[:2]))}."
    return "Look for more clarity, steadiness, or action-readiness."


def _unique_text(values: Any) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        label = str(value).strip()
        if label and label not in seen:
            seen.add(label)
            unique.append(label)
    return unique


def _format_label_text(values: list[str]) -> str:
    labels = [value.replace("_", " ") for value in _unique_text(values)]
    if not labels:
        return "none"
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _format_focus_context(extraction: ExtractedCoachingState) -> str:
    fields = (
        ("domains", extraction.domains),
        ("concerns", extraction.concerns),
        ("tasks", extraction.tasks),
        ("challenges", extraction.challenges),
        ("objectives", extraction.objectives),
        ("projects", extraction.projects),
        ("key_results", extraction.key_results),
        ("next_actions", extraction.next_actions),
        ("obstacles", extraction.obstacles),
        ("waiting_for", extraction.waiting_for),
        ("time_horizons", extraction.time_horizons),
        ("success_measures", extraction.success_measures),
        ("stakes", extraction.stakes),
    )
    lines = ["Concrete focus context:"]
    found = False
    for name, values in fields:
        cleaned = _clean_tuple(values)
        if cleaned:
            found = True
            lines.append(f"- {name}: {', '.join(cleaned)}")
    if not found:
        lines.append("- none extracted; avoid inventing a focus")
    return "\n".join(lines)


def build_mirror_prompt(
    graph: FormulationGraph,
    deterministic_draft: FormulationMirror,
) -> str:
    return (
        "Working formulation graph:\n"
        f"{json.dumps(graph.model_dump(), indent=2)}\n\n"
        "Deterministic draft to improve, if useful:\n"
        f"{deterministic_draft.text}\n\n"
        "Mirror this back to the user now."
    )


def build_llm_prompt(user_message: str, kernel_snapshot: dict[str, Any]) -> str:
    """Backward-compatible wrapper for older tests/imports."""

    extraction = ExtractedCoachingState(thoughts=(user_message,))
    return build_response_prompt(user_message, extraction, kernel_snapshot)


def format_extraction(extraction: ExtractedCoachingState) -> str:
    data = extraction.model_dump(exclude_none=True)
    lines = ["Structured extraction:"]
    for key, value in data.items():
        if value:
            lines.append(f"- {key}: {value}")
    if len(lines) == 1:
        lines.append("- none")
    return "\n".join(lines)


def format_kernel_snapshot(snapshot: dict[str, Any]) -> str:
    lines = ["Kernel hypotheses:"]
    operating_mode = snapshot.get("operating_mode") or {}
    if operating_mode.get("mode"):
        lines.append(f"- mode: {operating_mode.get('mode')}")
    hypotheses = snapshot.get("hypotheses", [])
    if hypotheses:
        for item in hypotheses:
            lines.append(f"- {item['source']}: {item['pattern']}")
    else:
        lines.append("- none")

    lines.append("Kernel differential formulations:")
    formulations = snapshot.get("formulations", [])
    if formulations:
        for item in formulations:
            lines.append(f"- {item.get('score')}: {item.get('label') or item.get('formulation')}")
            if item.get("discriminating_question"):
                lines.append(f"  discriminator: {item['discriminating_question']}")
    else:
        lines.append("- none")

    lines.append("Kernel clarifying moves:")
    clarifying_moves = snapshot.get("clarifying_moves", [])
    if clarifying_moves:
        for item in clarifying_moves:
            lines.append(f"- {item.get('priority')}: {item.get('kind')} - {item.get('question')}")
            if item.get("target_formulations"):
                lines.append(f"  targets: {', '.join(item['target_formulations'])}")
    else:
        lines.append("- none")

    deliberation = snapshot.get("intervention_deliberation") or {}
    lines.append("Kernel intervention deliberation:")
    if deliberation.get("selected_intervention"):
        lines.append(f"- selected: {deliberation.get('selected_intervention')}")
        if deliberation.get("runner_up_intervention"):
            lines.append(f"- runner_up: {deliberation.get('runner_up_intervention')}")
        for item in deliberation.get("candidates_considered") or ():
            reason = item.get("decision") or item.get("reason_not_chosen")
            suffix = f" - {reason}" if reason else ""
            lines.append(
                f"  {item.get('role')}: {item.get('intervention')} "
                f"({item.get('score')}){suffix}"
            )
    else:
        lines.append("- none")

    lines.append("Kernel-ranked candidates:")
    candidates = snapshot.get("candidates", [])
    if candidates:
        for item in candidates:
            lines.append(f"- {item['score']}: {item['intervention']}")
            if item.get("exercise"):
                lines.append(f"  exercise: {item['exercise']}")
            if item.get("contraindications"):
                lines.append(f"  contraindications: {', '.join(item['contraindications'])}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def format_response_plan(plan: ResponsePlan) -> str:
    lines = ["Response plan:"]
    for key, value in plan.model_dump(exclude_none=True).items():
        if value:
            lines.append(f"- {key}: {value}")
    if len(lines) == 1:
        lines.append("- none")
    return "\n".join(lines)


def render_response_plan(plan: ResponsePlan) -> str:
    """Render a structured plan into concise chat prose."""

    paragraphs = [plan.validation.strip()]

    if plan.safety_note:
        paragraphs.append(plan.safety_note.strip())
    if plan.hypothesis:
        paragraphs.append(plan.hypothesis.strip())

    if plan.exercise:
        paragraphs.append(plan.exercise.strip())

    if plan.question:
        question = plan.question.strip()
        paragraphs.append(
            question
            if question.endswith(("?", ".", "!")) or "?" in question
            else f"{question}?"
        )

    return "\n\n".join(part for part in paragraphs if part)


def is_consultative_intervention(intervention: str | None) -> bool:
    """Return whether an intervention is outside the therapeutic map loop."""

    return _clean_intervention_label(intervention) in CONSULTATIVE_INTERVENTIONS


def sanitize_response_plan(
    plan: ResponsePlan,
    *,
    kernel_snapshot: dict[str, Any],
) -> ResponsePlan:
    """Remove internal/debug content and enforce one-question rendering."""

    safety_note = _clean_safety_note(plan.safety_note, kernel_snapshot)
    hypothesis = _clean_hypothesis_text(plan.hypothesis, kernel_snapshot)
    exercise = _clean_plan_text(plan.exercise)
    question = _clean_plan_text(plan.question)
    intervention = _kernel_aligned_intervention(plan.intervention, kernel_snapshot)

    if question and exercise and "?" in exercise:
        exercise = None
    elif exercise:
        exercise = _limit_question_marks(exercise)
    if question:
        question = _limit_question_marks(question)

    return plan.model_copy(
        update={
            "validation": _clean_plan_text(plan.validation) or plan.validation,
            "hypothesis": hypothesis,
            "intervention": intervention,
            "exercise": exercise,
            "question": question,
            "safety_note": safety_note,
        }
    )


def cohere_response_plan(
    plan: ResponsePlan,
    *,
    kernel_snapshot: dict[str, Any],
    original_plan: ResponsePlan | None = None,
) -> tuple[ResponsePlan, dict[str, Any]]:
    """Verify and repair a response plan against kernel constraints."""

    issues: list[dict[str, str]] = []
    repairs: list[dict[str, str]] = []
    candidates = kernel_snapshot.get("candidates") or []
    selected = _candidate_for_intervention(candidates, plan.intervention)
    original_intervention = (
        _clean_intervention_label(original_plan.intervention)
        if original_plan is not None
        else None
    )
    working = plan

    def issue(code: str, severity: str, detail: str) -> None:
        issues.append({"code": code, "severity": severity, "detail": detail})

    def repair(code: str, detail: str) -> None:
        repairs.append({"code": code, "detail": detail})

    if original_intervention and original_intervention != working.intervention:
        issue(
            "intervention_realigned",
            "info",
            (
                f"The planner proposed {original_intervention}, but the kernel "
                f"selected {working.intervention} as the coherent safe move."
            ),
        )
        repair(
            "intervention_realigned",
            f"Changed intervention to {working.intervention}.",
        )

    if _kernel_has_pattern(kernel_snapshot, "safety_risk"):
        safety_candidate = _candidate_for_intervention(candidates, "safety_planning")
        if original_intervention and original_intervention != "safety_planning":
            issue(
                "safety_plan_required",
                "error",
                "Safety risk requires safety planning before coaching content.",
            )
            if working.intervention == "safety_planning":
                if not working.safety_note:
                    working = working.model_copy(
                        update={
                            "safety_note": (
                                "Because safety may be involved, pause coaching content and focus on immediate support."
                            )
                        }
                    )
                repair(
                    "safety_plan_required",
                    "Kept the kernel-aligned safety planning intervention.",
                )
        if safety_candidate and working.intervention != "safety_planning":
            if not any(item["code"] == "safety_plan_required" for item in issues):
                issue(
                    "safety_plan_required",
                    "error",
                    "Safety risk requires safety planning before coaching content.",
                )
            working = working.model_copy(
                update={
                    "intervention": "safety_planning",
                    "exercise": _candidate_exercise(safety_candidate),
                    "question": "Are you safe right now, and is there someone you can contact immediately?",
                    "safety_note": (
                        "Because safety may be involved, pause coaching content and focus on immediate support."
                    ),
                }
            )
            repair(
                "safety_plan_required",
                "Replaced the selected move with safety planning.",
            )

    selected = _candidate_for_intervention(candidates, working.intervention)
    if selected is None and candidates:
        best = _best_safe_candidate(candidates)
        if best:
            best_intervention = str(best.get("intervention"))
            issue(
                "unsupported_intervention",
                "error",
                f"The selected intervention was not generated by the kernel: {working.intervention}.",
            )
            working = working.model_copy(update={"intervention": best_intervention})
            selected = best
            repair(
                "unsupported_intervention",
                f"Changed intervention to {best_intervention}.",
            )

    if selected and selected.get("contraindications"):
        best = _best_safe_candidate(candidates)
        if best:
            best_intervention = str(best.get("intervention"))
            issue(
                "contraindicated_intervention",
                "error",
                (
                    f"{working.intervention} had contraindications: "
                    f"{', '.join(selected.get('contraindications') or [])}."
                ),
            )
            working = working.model_copy(update={"intervention": best_intervention})
            selected = best
            repair(
                "contraindicated_intervention",
                f"Changed intervention to {best_intervention}.",
            )

    if _needs_validation(kernel_snapshot) and not _substantive_validation(working.validation):
        issue(
            "validation_required",
            "error",
            "The state calls for validation before technique or challenge.",
        )
        working = working.model_copy(
            update={
                "validation": "That sounds difficult, and it makes sense to slow down with it."
            }
        )
        repair("validation_required", "Inserted a validating opening.")

    selected = _candidate_for_intervention(candidates, working.intervention)
    selected_exercise = _candidate_exercise(selected)
    if selected and _needs_exercise(working.intervention):
        original_changed = (
            bool(original_intervention)
            and original_intervention != working.intervention
        )
        if not working.exercise and selected_exercise:
            issue(
                "missing_exercise",
                "warning",
                f"{working.intervention} should include a concrete exercise.",
            )
            working = working.model_copy(update={"exercise": selected_exercise})
            repair("missing_exercise", "Filled the exercise from the kernel candidate.")
        elif original_changed and selected_exercise:
            issue(
                "exercise_rechecked_after_intervention_change",
                "info",
                "The exercise was refreshed because the intervention changed.",
            )
            working = working.model_copy(update={"exercise": selected_exercise})
            repair(
                "exercise_rechecked_after_intervention_change",
                "Replaced the exercise with the selected intervention's kernel exercise.",
            )

    if _kernel_has_pattern(kernel_snapshot, "minimal_disclosure") and not working.question:
        issue(
            "gentle_followup_missing",
            "warning",
            "Minimal disclosure usually needs one gentle follow-up question.",
        )

    clarifying_move = _top_clarifying_move(kernel_snapshot)
    if clarifying_move and not working.question and _should_apply_clarifying_move(clarifying_move):
        issue(
            "clarifying_question_missing",
            "warning",
            "The differential formulation layer proposed a question that would reduce uncertainty.",
        )
        working = working.model_copy(update={"question": clarifying_move.get("question")})
        repair(
            "clarifying_question_missing",
            "Inserted the top clarifying question from the kernel.",
        )

    selected_recipe = _recipe_for_intervention(
        kernel_snapshot.get("recipes") or (),
        working.intervention,
    )
    repaired_codes = {item["code"] for item in repairs}
    unrepaired_errors = [
        item for item in issues
        if item["severity"] == "error" and item["code"] not in repaired_codes
    ]
    status = "passed"
    if unrepaired_errors:
        status = "failed"
    elif repairs:
        status = "repaired"
    elif any(item["severity"] == "warning" for item in issues):
        status = "warning"

    report = {
        "status": status,
        "ok": not unrepaired_errors,
        "issues": issues,
        "repairs": repairs,
        "final_intervention": working.intervention,
        "matched_candidate": selected,
        "matched_recipe": selected_recipe,
    }
    return working, report


def build_turn_trace(
    turn: ChatTurnResult,
    *,
    include_prompts: bool = False,
) -> dict[str, Any]:
    return build_debug_trace(
        extraction=turn.prepared.extraction,
        state=turn.prepared.state,
        kernel_snapshot=turn.prepared.kernel_snapshot,
        response_plan=turn.response_plan,
        rendered_response=turn.text,
        extraction_prompt=turn.prepared.extraction_prompt,
        response_prompt=turn.prepared.response_prompt,
        local_context=turn.prepared.local_context,
        longitudinal_context=turn.prepared.longitudinal_context,
        memory_context=turn.prepared.memory_context,
        memory_packet=turn.prepared.memory_packet,
        policy_context=turn.prepared.policy_context,
        policy_report=turn.prepared.policy_report,
        plan_coherence=turn.plan_coherence,
        route=turn.prepared.route,
        include_prompts=include_prompts,
    )


def build_debug_trace(
    *,
    extraction: ExtractedCoachingState,
    state: CoachingState,
    kernel_snapshot: dict[str, Any],
    response_plan: ResponsePlan,
    rendered_response: str,
    extraction_prompt: str | None = None,
    response_prompt: str | None = None,
    local_context: str = "",
    longitudinal_context: str = "",
    memory_context: str = "",
    memory_packet: dict[str, Any] | None = None,
    policy_context: str = "",
    policy_report: dict[str, Any] | None = None,
    plan_coherence: dict[str, Any] | None = None,
    route: ConversationModeRoute | None = None,
    include_prompts: bool = False,
) -> dict[str, Any]:
    route = route or route_conversation_mode(state.utterance)
    candidates = kernel_snapshot.get("candidates") or []
    recipes = kernel_snapshot.get("recipes") or []
    clarifying_moves = kernel_snapshot.get("clarifying_moves") or []
    selected_candidate = _candidate_for_intervention(
        candidates,
        response_plan.intervention,
    )
    selected_recipe = _recipe_for_intervention(recipes, response_plan.intervention)
    counterfactuals = [
        item.model_dump()
        for item in simulate_counterfactual_interventions(
            kernel_snapshot,
            selected_intervention=response_plan.intervention,
        )
    ]
    backward_justification = None
    if response_plan.intervention:
        trace_kernel = TherapeuticReasoningKernel()
        trace_kernel.add_state(state)
        backward_justification = trace_kernel.intervention_requirement_report(
            state.state_id,
            response_plan.intervention,
        )
    pipeline = [
        "structured_extraction",
        "therapeutic_kernel",
    ]
    if memory_context.strip() or memory_packet:
        pipeline.append("memory_retrieval")
    if policy_context.strip() or policy_report:
        pipeline.append("adaptive_policy")
    pipeline.extend(["response_plan", "renderer"])
    trace: dict[str, Any] = {
        "pipeline": pipeline,
        "state_id": state.state_id,
        "route": route.model_dump(),
        "counterfactuals": counterfactuals,
        "extraction": _drop_empty(extraction.model_dump(exclude_none=True)),
        "kernel": {
            "operating_mode": kernel_snapshot.get("operating_mode") or {},
            "hypotheses": kernel_snapshot.get("hypotheses") or [],
            "formulations": kernel_snapshot.get("formulations") or [],
            "clarifying_moves": clarifying_moves,
            "candidates": candidates,
            "recipes": recipes,
            "intervention_deliberation": kernel_snapshot.get("intervention_deliberation")
            or {},
            "counterfactual_simulation": counterfactuals,
        },
        "selection": {
            "intervention": response_plan.intervention,
            "matched_candidate": selected_candidate,
            "recipe": selected_recipe,
            "exercise": response_plan.exercise,
            "hypothesis": response_plan.hypothesis,
            "question": response_plan.question,
            "safety_note": response_plan.safety_note,
            "tone_constraints": response_plan.tone_constraints,
            "backward_justification": backward_justification,
            "clarifying_move": _selected_clarifying_move(
                clarifying_moves,
                response_plan.question,
            ),
        },
        "rendered_response": rendered_response,
    }
    if plan_coherence:
        trace["plan_coherence"] = plan_coherence
    if state_context := longitudinal_context.strip():
        trace["longitudinal_context"] = state_context
    if conversation_context := local_context.strip():
        trace["local_context"] = conversation_context
    if memory_context.strip() or memory_packet:
        trace["memory"] = {
            "context": memory_context.strip(),
            **(memory_packet or {}),
        }
    if policy_context.strip() or policy_report:
        trace["policy"] = {
            "context": policy_context.strip(),
            **(policy_report or {}),
        }
    if include_prompts:
        trace["prompts"] = {
            "extraction": extraction_prompt,
            "response": response_prompt,
        }
    return trace


def format_turn_trace(
    turn: ChatTurnResult,
    *,
    include_prompts: bool = False,
) -> str:
    return format_debug_trace(build_turn_trace(turn, include_prompts=include_prompts))


def format_debug_trace(trace: dict[str, Any]) -> str:
    lines = ["Trace:"]
    lines.append(f"- state_id: {trace.get('state_id')}")
    lines.append(f"- pipeline: {' -> '.join(trace.get('pipeline', []))}")
    route = trace.get("route") or {}
    if route.get("mode"):
        lines.append(
            f"- mode_route: {route.get('label') or route.get('mode')} "
            f"({route.get('mode')}): {route.get('reason')}"
        )

    extraction = trace.get("extraction", {})
    lines.append("- extraction:")
    for key, value in extraction.items():
        lines.append(f"  {key}: {value}")
    if not extraction:
        lines.append("  none")

    kernel = trace.get("kernel", {})
    operating_mode = kernel.get("operating_mode") or {}
    if operating_mode.get("mode"):
        lines.append(f"- operating_mode: {operating_mode.get('mode')}")
    lines.append("- kernel hypotheses:")
    hypotheses = kernel.get("hypotheses") or []
    if hypotheses:
        for item in hypotheses:
            lines.append(f"  {item.get('source')}: {item.get('pattern')}")
    else:
        lines.append("  none")

    formulations = kernel.get("formulations") or []
    lines.append("- differential formulations:")
    if formulations:
        for item in formulations:
            label = item.get("label") or item.get("formulation")
            evidence = ", ".join(
                f"{hypothesis.get('source')}:{hypothesis.get('pattern')}"
                for hypothesis in (item.get("evidence") or [])[:4]
            )
            suffix = f" evidence={evidence}" if evidence else ""
            lines.append(f"  {item.get('score')}: {label}{suffix}")
            if item.get("discriminating_question"):
                lines.append(f"    question: {item.get('discriminating_question')}")
    else:
        lines.append("  none")

    clarifying_moves = kernel.get("clarifying_moves") or []
    lines.append("- clarifying moves:")
    if clarifying_moves:
        for item in clarifying_moves:
            targets = ", ".join(item.get("target_formulations") or ())
            suffix = f" targets={targets}" if targets else ""
            lines.append(
                f"  {item.get('priority')}: {item.get('kind')}{suffix}"
            )
            if item.get("question"):
                lines.append(f"    question: {item.get('question')}")
    else:
        lines.append("  none")

    lines.append("- kernel candidates:")
    candidates = kernel.get("candidates") or []
    if candidates:
        for item in candidates:
            contraindications = item.get("contraindications") or []
            suffix = (
                f" contraindications={contraindications}"
                if contraindications
                else ""
            )
            lines.append(
                f"  {item.get('score')}: {item.get('intervention')}{suffix}"
            )
    else:
        lines.append("  none")

    recipes = kernel.get("recipes") or []
    lines.append("- kernel recipes:")
    if recipes:
        for item in recipes:
            steps = " -> ".join(item.get("steps") or ())
            lines.append(f"  {item.get('score')}: {item.get('recipe')} ({steps})")
    else:
        lines.append("  none")

    deliberation = kernel.get("intervention_deliberation") or {}
    lines.append("- intervention deliberation:")
    if deliberation.get("selected_intervention"):
        lines.append(f"  selected: {deliberation.get('selected_intervention')}")
        if deliberation.get("runner_up_intervention"):
            lines.append(f"  runner_up: {deliberation.get('runner_up_intervention')}")
        for item in deliberation.get("candidates_considered") or ():
            reason = item.get("decision") or item.get("reason_not_chosen")
            suffix = f" - {reason}" if reason else ""
            lines.append(
                f"  {item.get('role')}: {item.get('intervention')} ({item.get('score')}){suffix}"
            )
    else:
        lines.append("  none")

    counterfactuals = trace.get("counterfactuals") or kernel.get("counterfactual_simulation") or []
    lines.append("- counterfactual simulation:")
    if counterfactuals:
        for item in counterfactuals[:4]:
            lines.append(
                f"  {item.get('role')}: {item.get('intervention')} -> "
                f"{item.get('expected_shift')}"
            )
            if item.get("why"):
                lines.append(f"    why: {item.get('why')}")
    else:
        lines.append("  none")

    selection = trace.get("selection", {})
    lines.append("- selection:")
    for key in (
        "intervention",
        "exercise",
        "hypothesis",
        "question",
        "safety_note",
        "tone_constraints",
    ):
        value = selection.get(key)
        if value:
            lines.append(f"  {key}: {value}")
    matched = selection.get("matched_candidate")
    if matched:
        lines.append(f"  matched_candidate_score: {matched.get('score')}")
    recipe = selection.get("recipe")
    if recipe:
        lines.append(f"  matched_recipe: {recipe.get('recipe')}")
    clarifying_move = selection.get("clarifying_move")
    if clarifying_move:
        lines.append(f"  clarifying_move: {clarifying_move.get('move')}")
    if len(lines) == 1:
        lines.append("  none")

    if coherence := trace.get("plan_coherence"):
        lines.append("- plan coherence:")
        lines.append(f"  status: {coherence.get('status')}")
        for item in coherence.get("issues") or ():
            lines.append(
                f"  issue[{item.get('severity')}]: {item.get('code')} - {item.get('detail')}"
            )
        for item in coherence.get("repairs") or ():
            lines.append(f"  repair: {item.get('code')} - {item.get('detail')}")

    if context := trace.get("longitudinal_context"):
        lines.append("- prior multi-turn context:")
        lines.extend(f"  {line}" for line in str(context).splitlines())

    if context := trace.get("local_context"):
        lines.append("- local conversation context:")
        lines.extend(f"  {line}" for line in str(context).splitlines())

    if memory := trace.get("memory"):
        lines.append("- retrieved workspace memory:")
        context = str(memory.get("context") or "").strip()
        if context:
            lines.extend(f"  {line}" for line in context.splitlines())
        else:
            counts = memory.get("counts") or {}
            if counts:
                lines.append(
                    "  "
                    + ", ".join(f"{key}={value}" for key, value in counts.items())
                )

    if policy := trace.get("policy"):
        lines.append("- adaptive policy:")
        for item in policy.get("adjustments") or ():
            lines.append(
                f"  {item.get('intervention')}: {item.get('base_score')} -> "
                f"{item.get('adjusted_score')} ({item.get('delta'):+})"
            )
            for reason in item.get("reasons") or ():
                lines.append(f"    reason: {reason}")
        summary = policy.get("summary") or {}
        counts = summary.get("counts") or {}
        if counts:
            lines.append(
                "  facts: "
                f"{counts.get('experiment_outcomes', 0)} experiment outcomes, "
                f"{counts.get('map_corrections', 0)} map corrections"
            )

    if longitudinal := trace.get("longitudinal"):
        lines.append("- multi-turn patterns:")
        for item in longitudinal:
            turns = item.get("turns") or []
            suffix = f" turns={tuple(turns)}" if turns else ""
            lines.append(f"  {item.get('label') or item.get('pattern')}{suffix}")

    if prompts := trace.get("prompts"):
        lines.append("- prompts:")
        for key, value in prompts.items():
            if value:
                lines.append(f"  {key}:")
                lines.extend(f"    {line}" for line in value.splitlines())

    return "\n".join(lines)


def chat_loop(
    coach: KernelGuidedCoach,
    *,
    show_extraction: bool = False,
    show_kernel: bool = False,
    show_plan: bool = False,
    show_trace: bool = False,
    trace_prompts: bool = False,
) -> int:
    print(
        "Empath chat. Type /quit or /exit to leave. "
        "Type /trace for last-turn trace, /debug to toggle trace."
    )
    history: list[ModelMessage] | None = None
    last_turn: ChatTurnResult | None = None
    trace_enabled = show_trace
    trace_prompts_enabled = trace_prompts
    while True:
        try:
            user_message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_message:
            continue
        if user_message in {"/quit", "/exit"}:
            return 0
        if user_message == "/trace":
            if last_turn is None:
                print("No completed turn to trace yet.")
            else:
                print(
                    "\n"
                    + format_turn_trace(
                        last_turn,
                        include_prompts=trace_prompts_enabled,
                    )
                )
            continue
        if user_message == "/debug":
            trace_enabled = not trace_enabled
            print(f"Trace output {'on' if trace_enabled else 'off'}.")
            continue
        if user_message == "/prompts":
            trace_prompts_enabled = not trace_prompts_enabled
            print(f"Trace prompt output {'on' if trace_prompts_enabled else 'off'}.")
            continue

        try:
            turn = coach.respond(user_message, message_history=history)
        except Exception as exc:  # pragma: no cover - exercised manually in CLI use
            print(f"error: {exc}", file=sys.stderr)
            return 1

        history = turn.message_history
        last_turn = turn
        if show_extraction:
            print("\n" + format_extraction(turn.prepared.extraction))
        if show_kernel:
            print("\n" + format_kernel_snapshot(turn.prepared.kernel_snapshot))
        if show_plan:
            print("\n" + format_response_plan(turn.response_plan))
        if trace_enabled:
            print(
                "\n"
                + format_turn_trace(
                    turn,
                    include_prompts=trace_prompts_enabled,
                )
            )
        print(f"\nempath> {turn.text}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeepSeek-backed Empath chat CLI.")
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
    parser.add_argument(
        "--once",
        help="Send one message and exit instead of starting an interactive loop.",
    )
    parser.add_argument(
        "--show-extraction",
        action="store_true",
        help="Print the LLM structured extraction for each turn.",
    )
    parser.add_argument(
        "--show-kernel",
        action="store_true",
        help="Print kernel hypotheses and intervention candidates for each turn.",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the structured response plan before rendered chat text.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print a compact end-to-end trace for each turn.",
    )
    parser.add_argument(
        "--trace-prompts",
        action="store_true",
        help="Include extraction and response prompts in trace output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the kernel-guided prompt without calling DeepSeek.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Model temperature. Default: 0.4",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Maximum response tokens. Default: 900",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        message = args.once or "I keep avoiding the prototype because if it is bad, I am a failure."
        kernel = TherapeuticReasoningKernel()
        state = state_from_user_message(message, state_id="dry-run")
        kernel.add_state(state)
        snapshot = with_intervention_deliberation(
            kernel.reasoning_snapshot(state.state_id, limit=5)
        )
        extraction = extraction_from_state(state)
        response_plan = draft_response_plan(snapshot)
        rendered_response = render_response_plan(response_plan)
        extraction_prompt = build_extraction_prompt(message)
        response_prompt = build_response_prompt(message, extraction, snapshot)
        print(format_extraction(extraction))
        print()
        print(format_kernel_snapshot(snapshot))
        print()
        print(format_response_plan(response_plan))
        if args.trace or args.trace_prompts:
            print()
            print(
                format_debug_trace(
                    build_debug_trace(
                        extraction=extraction,
                        state=state,
                        kernel_snapshot=snapshot,
                        response_plan=response_plan,
                        rendered_response=rendered_response,
                        extraction_prompt=extraction_prompt,
                        response_prompt=response_prompt,
                        include_prompts=args.trace_prompts,
                    )
                )
            )
        print("\nRendered plan preview:")
        print(rendered_response)
        print("\nPrompt preview:")
        print(response_prompt)
        return 0

    try:
        api_key = read_api_key(Path(args.api_key_file))
        coach = KernelGuidedCoach(
            api_key=api_key,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.once:
        try:
            turn = coach.respond(args.once)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.show_extraction:
            print(format_extraction(turn.prepared.extraction))
            print()
        if args.show_kernel:
            print(format_kernel_snapshot(turn.prepared.kernel_snapshot))
            print()
        if args.show_plan:
            print(format_response_plan(turn.response_plan))
            print()
        if args.trace or args.trace_prompts:
            print(format_turn_trace(turn, include_prompts=args.trace_prompts))
            print()
        print(turn.text)
        return 0

    return chat_loop(
        coach,
        show_extraction=args.show_extraction,
        show_kernel=args.show_kernel,
        show_plan=args.show_plan,
        show_trace=args.trace or args.trace_prompts,
        trace_prompts=args.trace_prompts,
    )


def draft_response_plan(
    kernel_snapshot: dict[str, Any],
    *,
    user_message: str = "",
) -> ResponsePlan:
    """Deterministic dry-run plan preview from the top kernel candidate."""

    candidates = kernel_snapshot.get("candidates") or []
    deliberation = kernel_snapshot.get("intervention_deliberation") or {}
    selected_intervention = deliberation.get("selected_intervention")
    candidate = (
        _candidate_for_intervention(candidates, selected_intervention)
        if selected_intervention
        else None
    ) or (candidates[0] if candidates else {})
    intervention = candidate.get("intervention")
    exercise = candidate.get("exercise")
    if is_consultative_intervention(str(intervention) if intervention else None):
        return sanitize_response_plan(
            _draft_consultative_response_plan(
                str(intervention),
                exercise=str(exercise) if exercise else None,
                user_message=user_message,
            ),
            kernel_snapshot=kernel_snapshot,
        )

    hypotheses = candidate.get("hypotheses") or kernel_snapshot.get("hypotheses") or []
    hypothesis_text = None
    if hypotheses:
        pattern = hypotheses[0].get("pattern")
        if pattern:
            hypothesis_text = _hypothesis_phrase(str(pattern))

    clarifying_move = _top_clarifying_move(kernel_snapshot)
    question = (
        _clean_plan_text(clarifying_move.get("question"))
        if clarifying_move
        else None
    ) or "What would be one small next step that feels workable right now?"

    return sanitize_response_plan(
        ResponsePlan(
            validation="That sounds difficult, and it makes sense to slow down with it.",
            hypothesis=hypothesis_text,
            intervention=intervention,
            exercise=exercise,
            question=question,
            tone_constraints=("brief", "tentative", "non-diagnostic"),
        ),
        kernel_snapshot=kernel_snapshot,
    )


def _draft_consultative_response_plan(
    intervention: str,
    *,
    exercise: str | None,
    user_message: str = "",
) -> ResponsePlan:
    if exercise in CONSULTATIVE_EXERCISE_TEMPLATES:
        exercise = None
    if intervention == "active_listening_repair":
        return ResponsePlan(
            validation="I hear the frustration. I am not here to argue with you.",
            hypothesis=(
                "I can stay useful here without being defensive: we can use Socratic inquiry, "
                "emotional support, or practical coaching depending on what you want."
            ),
            intervention=intervention,
            exercise=(
                exercise
                or "From here, I can ask a Socratic question, offer emotional support, or help coach through the problem you want to work on."
            ),
            question="What would be most useful from me right now?",
            tone_constraints=("friendly", "non-defensive", "supportive", "concise"),
        )
    if intervention == "concise_factual_answer":
        factual_answer = _deterministic_factual_answer(user_message)
        return ResponsePlan(
            validation=factual_answer
            or "I can answer this as a concise factual question.",
            hypothesis=None,
            intervention=intervention,
            exercise=None
            if factual_answer
            else "For arbitrary factual questions, use the live model path so Empath can answer directly instead of showing a kernel template.",
            question=None
            if factual_answer
            else "Do you want to switch to the live model for the factual answer?",
            tone_constraints=("friendly", "concise", "consultative"),
        )
    if intervention == "advisory_recommendation":
        return ResponsePlan(
            validation="I can approach this as a practical recommendation.",
            hypothesis=(
                "This looks like an advisory problem-solving request, not something that needs a therapeutic frame."
            ),
            intervention=intervention,
            exercise=exercise
            or "Give the recommendation, the rationale, the main tradeoff, and one next step.",
            question="What constraint matters most here?",
            tone_constraints=("friendly", "direct", "consultative"),
        )
    if intervention == "instructional_explanation":
        return ResponsePlan(
            validation="I can explain that briefly.",
            hypothesis=(
                "This looks like an instructional request, so the useful move is a compact explanation with one example."
            ),
            intervention=intervention,
            exercise=exercise
            or "Explain the concept briefly, give one example, and name one common mistake.",
            question="Do you want a coaching-oriented application of it?",
            tone_constraints=("friendly", "concise", "educational"),
        )
    if intervention == "analytical_synthesis":
        return ResponsePlan(
            validation="I can structure the comparison.",
            hypothesis=(
                "This looks like an analytical request, so the useful move is criteria, tradeoffs, and a provisional conclusion."
            ),
            intervention=intervention,
            exercise=exercise
            or "State the criteria, compare the options, and give a provisional conclusion.",
            question="What decision criterion should carry the most weight?",
            tone_constraints=("friendly", "analytical", "concise"),
        )
    if intervention == "creative_ideation":
        return ResponsePlan(
            validation="I can help brainstorm options.",
            hypothesis=(
                "This looks like an ideation request, so the useful move is to generate a few options and then narrow them."
            ),
            intervention=intervention,
            exercise=exercise
            or "Generate a small set of options, cluster them, and suggest one direction to refine.",
            question="Should the ideas optimize for clarity, novelty, or usefulness?",
            tone_constraints=("friendly", "creative", "practical"),
        )
    if intervention == "socratic_inquiry":
        return ResponsePlan(
            validation="I can help you think it through.",
            hypothesis=(
                "This looks like an exploration request, so one good question is more useful than a long answer."
            ),
            intervention=intervention,
            exercise=exercise
            or "Ask one question that tests the objective, assumption, evidence, or constraint.",
            question="What evidence would change your mind about the current conclusion?",
            tone_constraints=("friendly", "socratic", "concise"),
        )
    return ResponsePlan(
        validation="I can help structure this.",
        hypothesis=(
            "This looks like a consultative facilitation request rather than a therapeutic coaching turn."
        ),
        intervention=intervention,
        exercise=exercise
        or "Clarify the objective, constraints, options, tradeoffs, and one practical next step.",
        question="What outcome are you trying to move toward?",
        tone_constraints=("friendly", "open", "encouraging", "consultative"),
    )


def _deterministic_factual_answer(user_message: str) -> str | None:
    """Tiny dry-run fact base so consultative tests do not leak prompt templates."""

    lowered = user_message.casefold()
    if "checksum" in lowered:
        return (
            "A checksum is a short value computed from data to help verify integrity. "
            "If the data changes or is corrupted, recomputing the checksum will usually "
            "produce a different value, which makes mismatches easier to detect."
        )
    if re.search(r"\bSSE\b|server-sent events?|eventsource", user_message, re.IGNORECASE):
        return (
            "SSE, or Server-Sent Events, is a simple HTTP streaming pattern where a "
            "server pushes text events to a browser over one long-lived connection. "
            "It is useful for one-way updates like assistant progress, notifications, "
            "or logs."
        )
    if re.search(r"\bAPI\b|application programming interface", user_message, re.IGNORECASE):
        return (
            "An API is a defined way for software systems to communicate. It specifies "
            "what requests can be made, what inputs are expected, and what responses "
            "come back."
        )
    return None


def _hypothesis_phrase(pattern: str) -> str:
    phrases = {
        "felt_sense_contact": (
            "One possible frame is that there is a bodily felt sense here worth approaching gently."
        ),
        "unclear_felt_meaning": (
            "One possible frame is that the feeling has meaning that is not fully in words yet."
        ),
        "symbolization_needed": (
            "One possible frame is that a word, image, or phrase may need to be checked against the body sense."
        ),
        "inner_critic_presence": (
            "One possible frame is that a critical part may need a little space before the rest can be heard."
        ),
    }
    return phrases.get(
        pattern,
        f"One possible frame is that {pattern.replace('_', ' ')} may be part of what is happening.",
    )


def extraction_from_state(state: CoachingState) -> ExtractedCoachingState:
    """Create an extraction-shaped object from a deterministic fallback state."""

    return ExtractedCoachingState(
        situations=state.situations,
        concerns=state.concerns,
        tasks=state.tasks,
        challenges=state.challenges,
        objectives=state.objectives,
        projects=state.projects,
        key_results=state.key_results,
        next_actions=state.next_actions,
        obstacles=state.obstacles,
        implementation_intentions=state.implementation_intentions,
        waiting_for=state.waiting_for,
        time_horizons=state.time_horizons,
        success_measures=state.success_measures,
        stakes=state.stakes,
        domains=state.domains,
        thoughts=state.thoughts,
        beliefs=state.beliefs,
        emotions=state.emotions,
        bodily_sensations=state.bodily_sensations,
        urges=state.urges,
        behaviors=state.behaviors,
        consequences=state.consequences,
        values=state.values,
        goals=state.goals,
        distress=state.distress,
        features=state.features,
        thought_features=tuple(
            TextFeatureExtraction(text=text, features=tuple(features))
            for text, features in state.thought_features.items()
        ),
        belief_features=tuple(
            TextFeatureExtraction(text=text, features=tuple(features))
            for text, features in state.belief_features.items()
        ),
    )


def _clean_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values if value and value.strip())


def _feature_mapping(
    items: tuple[TextFeatureExtraction, ...],
) -> dict[str, tuple[str, ...]]:
    return {
        item.text.strip(): _clean_tuple(item.features)
        for item in items
        if item.text.strip() and item.features
    }


def _normalized_state_features(
    features: tuple[str, ...],
    *,
    distress: int | None,
    user_message: str,
) -> tuple[str, ...]:
    normalized = set(_clean_tuple(features))
    if "high_distress" in normalized:
        high_by_score = distress is not None and distress >= 7
        high_by_text = _text_implies_high_distress(user_message)
        if not high_by_score and not high_by_text:
            normalized.remove("high_distress")
    return tuple(sorted(normalized))


def _with_raw_user_message(
    extracted_thoughts: tuple[str, ...],
    user_message: str,
) -> tuple[str, ...]:
    thoughts = _clean_tuple(extracted_thoughts)
    message = user_message.strip()
    if not message:
        return thoughts
    if not thoughts:
        return (message,)
    normalized_message = _normalize_evidence_text(message)
    if any(_normalize_evidence_text(thought) == normalized_message for thought in thoughts):
        return thoughts
    return (*thoughts, message)


def _candidate_for_intervention(
    candidates: list[dict[str, Any]],
    intervention: str | None,
) -> dict[str, Any] | None:
    if intervention is None:
        return None
    for candidate in candidates:
        if candidate.get("intervention") == intervention:
            return candidate
    return None


def _best_safe_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in candidates:
        if not candidate.get("contraindications"):
            return candidate
    return None


def _candidate_exercise(candidate: dict[str, Any] | None) -> str | None:
    if not candidate:
        return None
    return _clean_plan_text(candidate.get("exercise"))


def _recipe_for_intervention(
    recipes: list[dict[str, Any]],
    intervention: str | None,
) -> dict[str, Any] | None:
    if intervention is None:
        return recipes[0] if recipes else None
    for recipe in recipes:
        if intervention in set(recipe.get("steps") or ()):
            return recipe
    return recipes[0] if recipes else None


def _top_clarifying_move(kernel_snapshot: dict[str, Any]) -> dict[str, Any] | None:
    moves = kernel_snapshot.get("clarifying_moves") or []
    return moves[0] if moves else None


def _selected_clarifying_move(
    moves: list[dict[str, Any]],
    question: str | None,
) -> dict[str, Any] | None:
    if not moves:
        return None
    cleaned_question = _clean_plan_text(question)
    if cleaned_question:
        normalized = cleaned_question.casefold().rstrip("?")
        for move in moves:
            move_question = _clean_plan_text(move.get("question"))
            if move_question and move_question.casefold().rstrip("?") == normalized:
                return move
    return moves[0]


def _should_apply_clarifying_move(move: dict[str, Any]) -> bool:
    question = _clean_plan_text(move.get("question"))
    if not question:
        return False
    priority = float(move.get("priority") or 0.0)
    if priority < 6.0:
        return False
    return str(move.get("kind") or "") in {
        "differential_question",
        "evidence_probe",
        "stabilization_check",
    }


def _clean_intervention_label(value: str | None) -> str | None:
    cleaned = _clean_plan_text(value)
    if not cleaned:
        return None
    return re.sub(r"[^a-z0-9_]+", "_", cleaned.casefold()).strip("_")


def _kernel_aligned_intervention(
    intervention: str | None,
    kernel_snapshot: dict[str, Any],
) -> str | None:
    candidates = kernel_snapshot.get("candidates") or []
    if not candidates:
        return intervention

    selected = _candidate_for_intervention(candidates, intervention)
    if intervention == "validation" and selected is not None:
        return intervention

    reference = [
        candidate
        for candidate in candidates
        if candidate.get("intervention") != "validation"
    ] or candidates
    best = reference[0]
    best_intervention = best.get("intervention")
    if selected is None:
        return best_intervention

    selected_score = float(selected.get("score") or 0)
    best_score = float(best.get("score") or 0)
    if best_intervention and best_score - selected_score >= 1.0:
        return best_intervention
    return intervention


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
        return tuple(
            cleaned
            for item in value
            if (cleaned := _drop_empty(item)) not in (None, {}, [], (), "")
        )
    return value


def _normalize_evidence_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _text_implies_high_distress(text: str) -> bool:
    return bool(
        re.search(
            r"\b(panic|panicking|can't breathe|cannot breathe|can't cope|"
            r"cannot cope|unbearable|overwhelmed)\b",
            text.casefold(),
        )
    )


def _clean_safety_note(
    safety_note: str | None,
    kernel_snapshot: dict[str, Any],
) -> str | None:
    cleaned = _clean_plan_text(safety_note)
    if not cleaned:
        return None
    if "no safety risk" in cleaned.casefold() or "no risk indicated" in cleaned.casefold():
        return None
    if not _kernel_has_pattern(kernel_snapshot, "safety_risk"):
        return None
    return cleaned


def _clean_hypothesis_text(
    hypothesis: str | None,
    kernel_snapshot: dict[str, Any],
) -> str | None:
    cleaned = _clean_plan_text(hypothesis)
    if not cleaned:
        return None
    normalized = _clean_intervention_label(cleaned)
    labels = set()
    for candidate in kernel_snapshot.get("candidates") or ():
        if intervention := candidate.get("intervention"):
            labels.add(str(intervention))
        for hypothesis_item in candidate.get("hypotheses") or ():
            if isinstance(hypothesis_item, dict) and hypothesis_item.get("pattern"):
                labels.add(str(hypothesis_item.get("pattern")))
    for item in kernel_snapshot.get("hypotheses") or ():
        if isinstance(item, dict) and item.get("pattern"):
            labels.add(str(item.get("pattern")))
    if normalized and normalized in labels:
        return None
    if "_" in cleaned and re.fullmatch(r"[a-z0-9_]+", cleaned.casefold()):
        return None
    return cleaned


def _clean_plan_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = " ".join(text.split())
    if cleaned in _INTERNAL_PLAN_TEXTS:
        return None
    lowered_cleaned = cleaned.casefold()
    internal_fragments = (
        "in live mode, answer",
        "answer directly in two to four concise sentences",
        "this looks like a neutral factual question",
        "rather than turning it into a coaching intervention",
        "no therapeutic hypothesis needed",
        "this is a factual request",
        "not something that needs a therapeutic frame",
        "rather than a therapeutic coaching turn",
        "this looks like an advisory problem-solving request",
        "this looks like an instructional request",
        "this looks like an analytical request",
        "this looks like an ideation request",
        "this looks like an exploration request",
    )
    if any(fragment in lowered_cleaned for fragment in internal_fragments):
        return None
    blocked_phrases = (
        "No safety risk indicated",
        "remain open in case",
    )
    for phrase in blocked_phrases:
        if phrase in cleaned:
            cleaned = cleaned.replace(phrase, "").strip(" .")
    return cleaned or None


_INTERNAL_PLAN_TEXTS = {
    "Reflect the feeling and the stakes before offering a technique.",
    "Ask one gentle question about what the feeling may be connected to.",
    *CONSULTATIVE_EXERCISE_TEMPLATES,
}


def _limit_question_marks(text: str) -> str:
    if text.count("?") <= 1:
        return text
    head, _, _tail = text.partition("?")
    return f"{head.strip()}?"


def _kernel_has_pattern(kernel_snapshot: dict[str, Any], pattern: str) -> bool:
    return any(
        item.get("pattern") == pattern
        for item in kernel_snapshot.get("hypotheses", [])
    )


def _needs_validation(kernel_snapshot: dict[str, Any]) -> bool:
    return any(
        _kernel_has_pattern(kernel_snapshot, pattern)
        for pattern in (
            "needs_validation",
            "minimal_disclosure",
            "high_distress",
            "safety_risk",
        )
    )


def _substantive_validation(text: str | None) -> bool:
    cleaned = _clean_plan_text(text)
    if not cleaned:
        return False
    words = re.findall(r"[a-zA-Z']+", cleaned)
    if len(words) < 5:
        return False
    lowered = cleaned.casefold()
    return bool(
        re.search(
            r"\b(makes sense|understandable|sounds|hear you|hear the|heavy|hard|difficult|painful|thank)\b",
            lowered,
        )
    )


def _needs_exercise(intervention: str | None) -> bool:
    if not intervention:
        return False
    if is_consultative_intervention(intervention):
        return False
    return intervention not in {
        "validation",
        "gentle_check_in",
        "needs_exploration",
        "safety_planning",
    }


def _infer_emotions(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    matches = []
    patterns = {
        "anxiety": r"\b(anxious|anxiety|worried|panic|nervous|scared|afraid)\b",
        "shame": r"\b(ashamed|shame|embarrassed|humiliated|worthless)\b",
        "sadness": r"\b(sad|grief|lonely|hopeless|down)\b",
        "anger": r"\b(angry|mad|furious|resentful|irritated)\b",
        "overwhelm": r"\b(overwhelmed|too much|can't cope|cannot cope)\b",
    }
    for emotion, pattern in patterns.items():
        if re.search(pattern, lowered):
            matches.append(emotion)
    return tuple(matches)


def _infer_values(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    values = []
    patterns = {
        "mastery": r"\b(build|learn|improve|craft|practice|prototype)\b",
        "autonomy": r"\b(startup|independent|freedom|choice|own path)\b",
        "connection": r"\b(friend|partner|team|family|relationship|belong)\b",
        "health": r"\b(sleep|body|exercise|health|recovery)\b",
        "integrity": r"\b(honest|truth|integrity|fair|right thing)\b",
    }
    for value, pattern in patterns.items():
        if re.search(pattern, lowered):
            values.append(value)
    return tuple(values)


def _infer_goals(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    goals = []
    if "prototype" in lowered:
        goals.append("work on prototype")
    if re.search(r"\b(launch|ship|publish|send|finish)\b", lowered):
        goals.append("complete a visible next step")
    if re.search(r"\b(talk|tell|ask|message)\b", lowered):
        goals.append("have a conversation")
    return tuple(goals)


def _infer_concerns(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    concerns = []
    if "investor update" in lowered:
        concerns.append("investor update")
    if "investor presentation" in lowered:
        concerns.append("investor presentation")
    elif "presentation" in lowered:
        concerns.append("presentation")
    if "prototype" in lowered:
        concerns.append("prototype")
    if re.search(r"\bnumbers?\b", lowered):
        concerns.append("numbers")
    if re.search(r"\b(company|startup|business)\b", lowered):
        concerns.append("company")
    if re.search(r"\b(hard conversation|difficult conversation|conflict)\b", lowered):
        concerns.append("conversation")
    if re.search(r"\b(boundary|say no|ask for what i need)\b", lowered):
        concerns.append("boundary")
    return _unique_texts(concerns)


def _infer_tasks(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    tasks = []
    if "investor update" in lowered:
        tasks.append("send investor update")
    if "investor presentation" in lowered:
        tasks.append("prepare investor presentation")
    elif re.search(r"\bpresentation\b", lowered):
        tasks.append("prepare presentation")
    if "prototype" in lowered:
        tasks.append("work on prototype")
    if re.search(r"\b(write|send|draft).{0,24}\b(email|message)\b", lowered):
        tasks.append("send message")
    if re.search(r"\b(hard conversation|difficult conversation)\b", lowered):
        tasks.append("have hard conversation")
    if re.search(r"\b(set|hold).{0,20}\bboundary\b", lowered):
        tasks.append("set boundary")
    if re.search(r"\b(launch|ship|publish|finish)\b", lowered):
        tasks.append("complete visible next step")
    return _unique_texts(tasks)


def _infer_challenges(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    challenges = []
    if re.search(r"\b(avoid|avoiding|procrastinat\w*|putting off|put off)\b", lowered):
        challenges.append("avoidance or procrastination")
    if re.search(r"\b(stuck|blocked|hard to start|can't get myself|cannot get myself)\b", lowered):
        challenges.append("difficulty starting")
    if re.search(r"\b(not cut out|incompetent|not capable|can't do this|cannot do this)\b", lowered):
        challenges.append("self-efficacy doubt")
    if re.search(r"\b(uncertain|don't know|do not know|can't decide|cannot decide)\b", lowered):
        challenges.append("uncertainty")
    if re.search(r"\b(distracted|can't focus|cannot focus|notifications|phone)\b", lowered):
        challenges.append("attention or environment friction")
    return _unique_texts(challenges)


def _infer_objectives(text: str, goals: tuple[str, ...] = ()) -> tuple[str, ...]:
    lowered = text.casefold()
    objectives = list(goals)
    if "investor update" in lowered:
        objectives.append("send investor update")
    if "investor presentation" in lowered:
        objectives.append("deliver investor presentation")
    elif re.search(r"\bpresentation\b", lowered):
        objectives.append("deliver presentation")
    if "prototype" in lowered:
        objectives.append("ship prototype")
    if re.search(r"\b(boundary|say no|ask for what i need)\b", lowered):
        objectives.append("communicate clearly")
    return _unique_texts(objectives)


def _infer_projects(
    text: str,
    tasks: tuple[str, ...] = (),
    objectives: tuple[str, ...] = (),
) -> tuple[str, ...]:
    lowered = text.casefold()
    projects = []
    if "investor update" in lowered:
        projects.append("investor update")
    if "investor presentation" in lowered:
        projects.append("investor presentation")
    elif re.search(r"\bpresentation\b", lowered):
        projects.append("presentation")
    if "prototype" in lowered:
        projects.append("prototype")
    if re.search(r"\b(project|initiative|launch|campaign|workflow|system)\b", lowered):
        projects.extend(tasks or objectives)
    return _unique_texts(projects)


def _infer_key_results(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    key_results = []
    if re.search(r"\b(key result|kr|metric|measure|measured by|success looks like)\b", lowered):
        key_results.append("define measurable progress")
    if re.search(r"\b(reduce|increase|complete|finish|ship|send|publish|deliver)\b", lowered):
        key_results.append("observable completion or progress")
    if re.search(r"\b(\d+%|\d+ percent|\d+ times?|\d+ days?|\d+ weeks?)\b", lowered):
        key_results.append("numeric progress target")
    return _unique_texts(key_results)


def _infer_next_actions(
    text: str,
    tasks: tuple[str, ...] = (),
) -> tuple[str, ...]:
    lowered = text.casefold()
    actions = []
    if re.search(r"\b(open|start|draft|write|send|schedule|call|email|text|review)\b", lowered):
        if "investor update" in lowered:
            actions.append("open or draft the investor update")
        elif "prototype" in lowered:
            actions.append("open the prototype work")
        elif "presentation" in lowered:
            actions.append("open the presentation draft")
        elif tasks:
            actions.append(f"start: {tasks[0]}")
    if re.search(r"\b(next action|next step|smallest step|first step)\b", lowered):
        actions.append("choose the next visible action")
    return _unique_texts(actions)


def _infer_obstacles(
    text: str,
    challenges: tuple[str, ...] = (),
) -> tuple[str, ...]:
    lowered = text.casefold()
    obstacles = list(challenges)
    if re.search(r"\b(if|because).{0,80}\b(fail|bad|weak|judge|incompetent|not cut out)\b", lowered):
        obstacles.append("fear of what the result might mean")
    if re.search(r"\b(tired|exhausted|no energy|after work|too busy|overloaded)\b", lowered):
        obstacles.append("low energy or overload")
    if re.search(r"\b(distracted|notifications|phone|environment|interruptions)\b", lowered):
        obstacles.append("attention friction")
    if re.search(r"\b(waiting for|blocked by|need .* from|until they)\b", lowered):
        obstacles.append("waiting on external input")
    return _unique_texts(obstacles)


def _infer_implementation_intentions(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    intentions = []
    match = re.search(r"\bif\b(.{1,80})\bthen\b(.{1,100})", text, re.IGNORECASE)
    if match:
        intentions.append(f"if {match.group(1).strip()} then {match.group(2).strip()}")
    if "when i" in lowered and re.search(r"\bi will\b", lowered):
        intentions.append("when obstacle appears, follow stated plan")
    return _unique_texts(intentions)


def _infer_waiting_for(text: str) -> tuple[str, ...]:
    waiting = []
    for match in re.finditer(
        r"\b(waiting for|blocked by|need)\s+(.{1,60}?)(?:[.;,]|$)",
        text,
        re.IGNORECASE,
    ):
        item = match.group(2).strip()
        if item:
            waiting.append(item)
    return _unique_texts(waiting)


def _infer_time_horizons(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    horizons = []
    patterns = {
        "today": r"\btoday\b",
        "tomorrow": r"\btomorrow\b",
        "this week": r"\bthis week\b",
        "next week": r"\bnext week\b",
        "this month": r"\bthis month\b",
        "12 weeks": r"\b(12 week|twelve week)\b",
        "deadline": r"\b(deadline|due)\b",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, lowered):
            horizons.append(label)
    return _unique_texts(horizons)


def _infer_success_measures(
    text: str,
    key_results: tuple[str, ...] = (),
) -> tuple[str, ...]:
    lowered = text.casefold()
    measures = list(key_results)
    if re.search(r"\b(done when|success means|success looks like|measure|metric)\b", lowered):
        measures.append("observable success measure")
    if "investor update" in lowered:
        measures.append("investor update sent")
    if "prototype" in lowered:
        measures.append("prototype progress is visible")
    if re.search(r"\b(send|sent|publish|published|ship|shipped|finish|finished)\b", lowered):
        measures.append("item completed or shipped")
    return _unique_texts(measures)


def _infer_stakes(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    stakes = []
    if re.search(r"\b(investor|fundraising|fundraise|board)\b", lowered):
        stakes.append("investor judgment")
    if re.search(r"\b(judge me|judged|think i.?m|think i am|disappoint|approve|approval)\b", lowered):
        stakes.append("how others see me")
    if re.search(r"\b(not cut out|worthless|failure|i am a failure|i'm a failure|incompetent)\b", lowered):
        stakes.append("self-worth or identity")
    if re.search(r"\b(fail|failure|succeed|success|weak numbers?)\b", lowered):
        stakes.append("success or failure")
    if re.search(r"\b(relationship|friend|partner|family|team|conflict)\b", lowered):
        stakes.append("relationship strain")
    return _unique_texts(stakes)


def _infer_domains(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    domains = []
    if re.search(
        r"\b(work|job|company|startup|business|project|prototype|presentation|"
        r"investor|deadline|launch|ship|team)\b",
        lowered,
    ):
        domains.append("work")
    if re.search(r"\b(investor|fundraising|fundraise|money|revenue|numbers?)\b", lowered):
        domains.append("money")
    if re.search(
        r"\b(not cut out|worthless|failure|incompetent|self-worth|identity)\b",
        lowered,
    ):
        domains.append("identity")
    if re.search(r"\b(friend|partner|family|relationship|conflict|boundary)\b", lowered):
        domains.append("relationship")
    if re.search(r"\b(sleep|body|exercise|health|recovery)\b", lowered):
        domains.append("health")
    return _unique_texts(domains)


def _infer_distress(text: str, emotions: tuple[str, ...]) -> int | None:
    lowered = text.casefold()
    if re.search(r"\b(suicid|self-harm|hurt myself|end it all)\b", lowered):
        return 9
    if re.search(r"\b(panic|can't cope|cannot cope|overwhelmed|unbearable)\b", lowered):
        return 8
    if emotions:
        return 6 if "overwhelm" in emotions else 5
    return None


def _compact_text(text: str, *, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _unique_texts(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


if __name__ == "__main__":
    raise SystemExit(main())
