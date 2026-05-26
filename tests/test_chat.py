import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coach.chat import (
    DEFAULT_MODEL,
    EXTRACTION_INSTRUCTIONS,
    ExtractedCoachingState,
    KernelGuidedCoach,
    MirrorResponse,
    RESPONSE_PLAN_INSTRUCTIONS,
    ResponsePlan,
    TextFeatureExtraction,
    build_llm_prompt,
    build_response_prompt,
    build_turn_trace,
    cohere_response_plan,
    extraction_from_state,
    format_turn_trace,
    format_response_plan,
    render_response_plan,
    sanitize_response_plan,
    format_kernel_snapshot,
    read_api_key,
    state_from_extraction,
    state_from_user_message,
)
from coach.formulation import FormulationGraph
from coach.therapeutic_kernel import TherapeuticReasoningKernel


class _FakeRunResult:
    def __init__(self, output):
        self.output = output

    def all_messages(self):
        return ["message-history"]


class _FakeAgent:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def run_sync(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return _FakeRunResult(self.output)


class _FlakyAgent(_FakeAgent):
    def __init__(self, output, *, fail_times):
        super().__init__(output)
        self.fail_times = fail_times

    def run_sync(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("transient model failure")
        return _FakeRunResult(self.output)


class ChatWorkflowTests(unittest.TestCase):
    def test_read_api_key_strips_whitespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key"
            path.write_text("  secret-key\n", encoding="utf-8")
            self.assertEqual("secret-key", read_api_key(path))

    def test_message_preparation_feeds_kernel(self):
        kernel = TherapeuticReasoningKernel()
        state = state_from_user_message(
            "I keep avoiding the prototype because if it is bad, I am a failure.",
            state_id="turn-1",
        )
        kernel.add_state(state)
        snapshot = kernel.reasoning_snapshot("turn-1")
        prompt = build_llm_prompt(state.utterance, snapshot)

        rendered = format_kernel_snapshot(snapshot)
        self.assertIn("act: experiential_avoidance", rendered)
        self.assertIn("rebt: self_downing", rendered)
        self.assertIn(DEFAULT_MODEL, "deepseek-v4-flash")
        self.assertIn("Therapeutic kernel output", prompt)
        self.assertIn("cognitive_defusion", prompt)

    def test_extraction_to_state_preserves_features_for_kernel(self):
        extraction = ExtractedCoachingState(
            situations=("working on the prototype",),
            concerns=("prototype quality",),
            tasks=("work on prototype",),
            challenges=("avoidance",),
            objectives=("ship prototype",),
            projects=("prototype",),
            next_actions=("open the prototype work",),
            obstacles=("fear of what the result might mean",),
            success_measures=("prototype progress is visible",),
            stakes=("self-worth feels at risk",),
            domains=("work", "identity"),
            thoughts=("If it is bad, I am a failure.",),
            emotions=("anxiety",),
            behaviors=("avoidance",),
            values=("mastery",),
            thought_features=(
                TextFeatureExtraction(
                    text="If it is bad, I am a failure.",
                    features=("global_label", "identity_fusion"),
                ),
            ),
        )
        state = state_from_extraction(
            extraction,
            user_message="I keep avoiding the prototype because if it is bad, I am a failure.",
            state_id="turn-1",
        )
        kernel = TherapeuticReasoningKernel(auto_features=False)
        kernel.add_state(state)

        self.assertEqual(("prototype quality",), state.concerns)
        self.assertEqual(("work on prototype",), state.tasks)
        self.assertEqual(("avoidance",), state.challenges)
        self.assertEqual(("ship prototype",), state.objectives)
        self.assertEqual(("prototype",), state.projects)
        self.assertEqual(("open the prototype work",), state.next_actions)
        self.assertEqual(("fear of what the result might mean",), state.obstacles)
        self.assertEqual(("prototype progress is visible",), state.success_measures)
        self.assertEqual(("self-worth feels at risk",), state.stakes)
        self.assertEqual(("work", "identity"), state.domains)
        rendered = format_kernel_snapshot(kernel.reasoning_snapshot("turn-1"))
        self.assertIn("act: experiential_avoidance", rendered)
        self.assertIn("cbt: global_labeling", rendered)

    def test_extraction_schema_tracks_what_emotion_is_about(self):
        extraction = ExtractedCoachingState(
            concerns=("investor presentation",),
            tasks=("prepare investor presentation",),
            challenges=("urge to procrastinate",),
            objectives=("deliver a clear investor presentation",),
            projects=("investor presentation",),
            next_actions=("open the presentation draft",),
            obstacles=("fear of investor judgment",),
            time_horizons=("this week",),
            success_measures=("presentation draft sent",),
            stakes=("investor judgment",),
            domains=("work", "fundraising"),
        )

        payload = extraction.model_dump()

        self.assertEqual(("investor presentation",), payload["concerns"])
        self.assertEqual(("prepare investor presentation",), payload["tasks"])
        self.assertEqual(("urge to procrastinate",), payload["challenges"])
        self.assertEqual(
            ("deliver a clear investor presentation",),
            payload["objectives"],
        )
        self.assertEqual(("investor presentation",), payload["projects"])
        self.assertEqual(("open the presentation draft",), payload["next_actions"])
        self.assertEqual(("fear of investor judgment",), payload["obstacles"])
        self.assertEqual(("this week",), payload["time_horizons"])
        self.assertEqual(("presentation draft sent",), payload["success_measures"])
        self.assertEqual(("investor judgment",), payload["stakes"])
        self.assertEqual(("work", "fundraising"), payload["domains"])
        for field in (
            "concerns",
            "tasks",
            "challenges",
            "objectives",
            "projects",
            "next_actions",
            "obstacles",
            "success_measures",
            "stakes",
            "domains",
        ):
            self.assertIn(field, EXTRACTION_INSTRUCTIONS)

    def test_deterministic_fallback_infers_focus_context(self):
        state = state_from_user_message(
            (
                "I keep procrastinating on the investor presentation because "
                "they'll judge me and decide I'm not cut out."
            ),
            state_id="investor-presentation",
        )
        extraction = extraction_from_state(state)

        self.assertIn("investor presentation", state.concerns)
        self.assertIn("prepare investor presentation", state.tasks)
        self.assertIn("avoidance or procrastination", state.challenges)
        self.assertIn("deliver investor presentation", state.objectives)
        self.assertIn("investor presentation", state.projects)
        self.assertIn("fear of what the result might mean", state.obstacles)
        self.assertIn("investor judgment", state.stakes)
        self.assertIn("work", state.domains)
        self.assertIn("identity", state.domains)
        self.assertEqual(state.tasks, extraction.tasks)
        self.assertEqual(state.domains, extraction.domains)

    def test_kernel_guided_coach_runs_extractor_before_response_agent(self):
        extraction = ExtractedCoachingState(
            thoughts=("If it is bad, I am a failure.",),
            behaviors=("avoidance",),
            thought_features=(
                TextFeatureExtraction(
                    text="If it is bad, I am a failure.",
                    features=("global_label", "identity_fusion"),
                ),
            ),
        )

        coach = KernelGuidedCoach.__new__(KernelGuidedCoach)
        coach.kernel = TherapeuticReasoningKernel(auto_features=False)
        coach._turn_index = 0
        coach.extractor_agent = _FakeAgent(extraction)
        coach.response_agent = _FakeAgent(
            ResponsePlan(
                validation="That sounds heavy.",
                hypothesis="One possible frame is that this thought has become sticky.",
                intervention="cognitive_defusion",
                exercise="Try saying: I am noticing the thought that this will fail.",
                question="What is one small step you can take anyway?",
                tone_constraints=("brief", "tentative"),
            )
        )

        turn = coach.respond("I keep avoiding the prototype.")

        self.assertIn("That sounds heavy", turn.text)
        self.assertIn("I am noticing the thought", turn.text)
        self.assertEqual("cognitive_defusion", turn.response_plan.intervention)
        self.assertEqual(1, len(coach.extractor_agent.calls))
        self.assertEqual(1, len(coach.response_agent.calls))
        response_prompt = coach.response_agent.calls[0][0]
        self.assertIn("Structured extraction", response_prompt)
        self.assertIn("Therapeutic kernel output", response_prompt)
        self.assertIn("cognitive_defusion", response_prompt)
        self.assertEqual(["message-history"], turn.message_history)

    def test_response_prompt_includes_extraction_and_kernel_snapshot(self):
        extraction = ExtractedCoachingState(
            concerns=("investor presentation",),
            tasks=("prepare investor presentation",),
            challenges=("avoidance or procrastination",),
            objectives=("deliver investor presentation",),
            stakes=("investor judgment",),
            domains=("work", "identity"),
            thoughts=("I am not good enough.",),
        )
        kernel = TherapeuticReasoningKernel()
        state = state_from_extraction(
            extraction,
            user_message="I am not good enough.",
            state_id="turn-1",
        )
        kernel.add_state(state)
        prompt = build_response_prompt(
            "I am not good enough.",
            extraction,
            kernel.reasoning_snapshot("turn-1"),
        )

        self.assertIn("Structured extraction", prompt)
        self.assertIn("Concrete focus context", prompt)
        self.assertIn("- tasks: prepare investor presentation", prompt)
        self.assertIn("- stakes: investor judgment", prompt)
        self.assertIn("Therapeutic kernel output", prompt)
        self.assertIn("concerns, tasks, challenges, objectives, stakes, and domains", RESPONSE_PLAN_INSTRUCTIONS)
        self.assertIn("make the exercise name that", RESPONSE_PLAN_INSTRUCTIONS)

    def test_response_plan_can_be_rendered_and_inspected(self):
        plan = ResponsePlan(
            validation="That sounds like a painful place to be.",
            hypothesis="One possible frame is that the prototype has become tied to self-worth.",
            intervention="unconditional_self_acceptance",
            exercise="Separate the outcome from your worth as a person.",
            question="What would you try if this were just practice?",
            tone_constraints=("warm", "concise", "non-diagnostic"),
        )

        rendered = render_response_plan(plan)
        formatted = format_response_plan(plan)

        self.assertIn("painful place", rendered)
        self.assertIn("Separate the outcome", rendered)
        self.assertIn("unconditional_self_acceptance", formatted)
        self.assertIn("tone_constraints", formatted)

    def test_response_plan_sanitizer_removes_internal_safety_note_and_extra_question(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            state_from_extraction(
                ExtractedCoachingState(emotions=("sad",)),
                user_message="hi, I'm feeling sad today",
                state_id="sad",
            )
        )
        snapshot = kernel.reasoning_snapshot("sad")
        plan = ResponsePlan(
            validation="Thanks for saying that.",
            hypothesis="One possible frame is that sadness may be asking for care.",
            intervention="gentle_check_in",
            exercise="Is there anything that feels especially heavy today?",
            question="What would feel supportive right now?",
            safety_note="No safety risk indicated, but remain open in case the user shares more depth.",
            tone_constraints=("warm",),
        )

        sanitized = sanitize_response_plan(plan, kernel_snapshot=snapshot)
        rendered = render_response_plan(sanitized)

        self.assertIsNone(sanitized.safety_note)
        self.assertIsNone(sanitized.exercise)
        self.assertEqual(1, rendered.count("?"))
        self.assertNotIn("No safety risk", rendered)

    def test_plan_coherence_reports_intervention_realignment(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            state_from_extraction(
                ExtractedCoachingState(
                    thoughts=("I keep putting off sending the investor update.",),
                    behaviors=("procrastination", "avoidance"),
                    features=("needs_validation",),
                ),
                user_message="I keep putting off sending the investor update.",
                state_id="investor-update",
            )
        )
        snapshot = kernel.reasoning_snapshot("investor-update")
        original = ResponsePlan(
            validation="That hesitation makes sense.",
            intervention="committed_action",
            exercise="Open the document and work for 10 minutes.",
        )
        sanitized = sanitize_response_plan(original, kernel_snapshot=snapshot)

        coherent, report = cohere_response_plan(
            sanitized,
            kernel_snapshot=snapshot,
            original_plan=original,
        )

        self.assertEqual("acceptance_committed_action", coherent.intervention)
        self.assertEqual("repaired", report["status"])
        self.assertIn(
            "intervention_realigned",
            {item["code"] for item in report["issues"]},
        )

    def test_plan_coherence_fills_missing_exercise_from_selected_candidate(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            state_from_extraction(
                ExtractedCoachingState(
                    thoughts=("If it is bad, I am a failure.",),
                    thought_features=(
                        TextFeatureExtraction(
                            text="If it is bad, I am a failure.",
                            features=("identity_fusion", "global_label"),
                        ),
                    ),
                ),
                user_message="If it is bad, I am a failure.",
                state_id="prototype",
            )
        )
        snapshot = kernel.reasoning_snapshot("prototype")
        plan = ResponsePlan(
            validation="That sounds heavy.",
            intervention="cognitive_defusion",
            exercise=None,
        )
        sanitized = sanitize_response_plan(plan, kernel_snapshot=snapshot)

        coherent, report = cohere_response_plan(sanitized, kernel_snapshot=snapshot)

        self.assertEqual("cognitive_defusion", coherent.intervention)
        self.assertIn("noticing the thought", coherent.exercise)
        self.assertIn("missing_exercise", {item["code"] for item in report["issues"]})

    def test_plan_coherence_prioritizes_safety_risk(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            state_from_extraction(
                ExtractedCoachingState(
                    thoughts=("They will think I am incompetent.",),
                ),
                user_message="They will think I am incompetent and I want to hurt myself.",
                state_id="safety",
            )
        )
        snapshot = kernel.reasoning_snapshot("safety")
        original = ResponsePlan(
            validation="That sounds hard.",
            intervention="evidence_check",
            exercise="List the facts and guesses.",
        )
        sanitized = sanitize_response_plan(original, kernel_snapshot=snapshot)

        coherent, report = cohere_response_plan(
            sanitized,
            kernel_snapshot=snapshot,
            original_plan=original,
        )

        self.assertEqual("safety_planning", coherent.intervention)
        self.assertIsNotNone(coherent.safety_note)
        self.assertIn("safety_plan_required", {item["code"] for item in report["issues"]})

    def test_raw_user_message_preserves_dropped_identity_and_mind_reading_clauses(self):
        extraction = ExtractedCoachingState(
            situations=("Putting off sending an investor update",),
            thoughts=("I keep putting off sending the investor update.",),
            beliefs=("If the investors see the numbers, they will see them as weak.",),
            emotions=("anxiety",),
            urges=("Urge to avoid sending the investor update",),
            behaviors=("procrastination", "avoidance"),
            features=("needs_validation",),
            belief_features=(
                TextFeatureExtraction(
                    text="If the investors see the numbers, they will see them as weak.",
                    features=("future_disaster", "mind_reading_claim"),
                ),
            ),
        )
        message = (
            "I keep putting off sending the investor update. If they see the numbers "
            "are weak, they’ll think I’m incompetent, and honestly maybe that means "
            "I’m not cut out to run this company. I know I should send it, but I "
            "just keep avoiding it."
        )
        state = state_from_extraction(
            extraction,
            user_message=message,
            state_id="investor-update",
        )
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(state)

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("investor-update")
        }
        self.assertIn(("cbt", "mind_reading"), hypotheses)
        self.assertIn(("cbt", "global_labeling"), hypotheses)
        self.assertIn(("rebt", "self_downing"), hypotheses)
        self.assertIn(("rebt", "demandingness"), hypotheses)
        self.assertIn(("act", "fusion"), hypotheses)

    def test_response_plan_sanitizer_aligns_low_rank_intervention_to_kernel_candidate(self):
        kernel = TherapeuticReasoningKernel()
        state = state_from_extraction(
            ExtractedCoachingState(
                thoughts=("I keep putting off sending the investor update.",),
                behaviors=("procrastination", "avoidance"),
                features=("needs_validation",),
            ),
            user_message=(
                "I keep putting off sending the investor update. If they see the numbers "
                "are weak, they’ll think I’m incompetent, and honestly maybe that means "
                "I’m not cut out to run this company. I know I should send it, but I "
                "just keep avoiding it."
            ),
            state_id="investor-update",
        )
        kernel.add_state(state)
        snapshot = kernel.reasoning_snapshot("investor-update")
        plan = ResponsePlan(
            validation="That hesitation makes sense.",
            hypothesis="One possible frame is that mind-reading is locking in avoidance.",
            intervention="committed_action",
            exercise="Open the document and work for 10 minutes.",
            question="What is the smallest piece you can touch today?",
            tone_constraints=("concise",),
        )

        sanitized = sanitize_response_plan(plan, kernel_snapshot=snapshot)

        self.assertEqual("acceptance_committed_action", sanitized.intervention)

    def test_extraction_feature_normalization_removes_contradictory_high_distress(self):
        extraction = ExtractedCoachingState(
            thoughts=("I should send the investor update.",),
            emotions=("anxiety", "shame"),
            behaviors=("avoidance", "procrastination", "inaction"),
            distress=6,
            features=("high_distress",),
        )
        state = state_from_extraction(
            extraction,
            user_message=(
                "I keep putting off sending the investor update. If they see the numbers "
                "are weak, they’ll think I’m incompetent, and maybe I’m not cut out. "
                "I know I should send it, but I keep avoiding it."
            ),
            state_id="investor-update",
        )
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(state)
        snapshot = kernel.reasoning_snapshot("investor-update")
        plan = ResponsePlan(
            validation="That hesitation makes sense.",
            hypothesis="One possible frame is identity fusion.",
            intervention="present_moment_grounding",
            exercise="Feel your feet on the floor.",
            question="What is the smallest piece you can touch today?",
            tone_constraints=("concise",),
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("investor-update")
        }
        sanitized = sanitize_response_plan(plan, kernel_snapshot=snapshot)

        self.assertNotIn(("policy", "high_distress"), hypotheses)
        self.assertIn(("policy", "needs_validation"), hypotheses)
        self.assertEqual("acceptance_committed_action", sanitized.intervention)

    def test_turn_trace_exposes_pipeline_selection_and_prompts(self):
        extraction = ExtractedCoachingState(
            thoughts=("If it is bad, I am a failure.",),
            behaviors=("avoidance",),
            thought_features=(
                TextFeatureExtraction(
                    text="If it is bad, I am a failure.",
                    features=("global_label", "identity_fusion"),
                ),
            ),
        )

        coach = KernelGuidedCoach.__new__(KernelGuidedCoach)
        coach.kernel = TherapeuticReasoningKernel(auto_features=False)
        coach._turn_index = 0
        coach.extractor_agent = _FakeAgent(extraction)
        coach.response_agent = _FakeAgent(
            ResponsePlan(
                validation="That sounds heavy.",
                hypothesis="One possible frame is that this thought has become sticky.",
                intervention="cognitive_defusion",
                exercise="Try saying: I am noticing the thought that this will fail.",
                question="What is one small step you can take anyway?",
                tone_constraints=("brief", "tentative"),
            )
        )

        turn = coach.respond("I keep avoiding the prototype.")
        trace = build_turn_trace(turn, include_prompts=True)
        formatted = format_turn_trace(turn, include_prompts=True)

        self.assertEqual(
            [
                "structured_extraction",
                "therapeutic_kernel",
                "response_plan",
                "renderer",
            ],
            trace["pipeline"],
        )
        self.assertEqual("cognitive_defusion", trace["selection"]["intervention"])
        self.assertEqual(
            "cognitive_defusion",
            trace["selection"]["matched_candidate"]["intervention"],
        )
        self.assertIn("extraction", trace["prompts"])
        self.assertIn("response", trace["prompts"])
        self.assertIn("Trace:", formatted)
        self.assertIn("differential formulations", formatted)
        self.assertIn("formulations", trace["kernel"])
        self.assertIn("clarifying_moves", trace["kernel"])
        self.assertIn("clarifying moves", formatted)
        self.assertIn("kernel candidates", formatted)
        self.assertIn("kernel recipes", formatted)
        self.assertIn("matched_candidate_score", formatted)
        self.assertIn("matched_recipe", formatted)

    def test_response_prompt_can_include_longitudinal_context(self):
        extraction = ExtractedCoachingState(emotions=("anxiety",))
        snapshot = {"hypotheses": [], "candidates": []}
        prompt = build_response_prompt(
            "I am anxious again.",
            extraction,
            snapshot,
            longitudinal_context=(
                "Tentative multi-turn patterns from prior turns:\n"
                "- recurring anxiety avoidance loop: Anxiety and avoidance have shown up together."
            ),
        )

        self.assertIn("Longitudinal session context", prompt)
        self.assertIn("recurring anxiety avoidance loop", prompt)

    def test_response_prompt_can_include_bounded_local_context(self):
        extraction = ExtractedCoachingState(emotions=("anxiety",))
        snapshot = {"hypotheses": [], "candidates": []}
        prompt = build_response_prompt(
            "I am anxious again.",
            extraction,
            snapshot,
            local_context=(
                "user: I avoided the update yesterday.\n"
                "coach: We named one small next action.\n"
                "user: I am anxious again."
            ),
        )

        self.assertIn("Local conversation context", prompt)
        self.assertIn("bounded to the last five user turns", prompt)
        self.assertIn("only on the latest user message", prompt)
        self.assertIn("I avoided the update yesterday", prompt)

    def test_response_prompt_can_include_retrieved_memory_context(self):
        extraction = ExtractedCoachingState(emotions=("anxiety",))
        snapshot = {"hypotheses": [], "candidates": []}
        prompt = build_response_prompt(
            "I'm still avoiding the investor update.",
            extraction,
            snapshot,
            memory_context=(
                "Retrieved workspace memory packet:\n"
                "Active focus:\n"
                "- investor update [task]\n"
                "Suppressed assumptions to avoid:\n"
                "- investors will definitely reject me [belief]"
            ),
        )

        self.assertIn("Retrieved workspace memory", prompt)
        self.assertIn("continuity and user-specific learning", prompt)
        self.assertIn("investor update", prompt)
        self.assertIn("Suppressed assumptions", prompt)

    def test_response_prompt_can_include_policy_context(self):
        extraction = ExtractedCoachingState(emotions=("anxiety",))
        snapshot = {"hypotheses": [], "candidates": []}
        prompt = build_response_prompt(
            "I am anxious again.",
            extraction,
            snapshot,
            policy_context="Prior feedback said cognitive defusion helped.",
        )

        self.assertIn("Adaptive policy memory", prompt)
        self.assertIn("cognitive defusion helped", prompt)
        self.assertIn("adaptive policy memory", RESPONSE_PLAN_INSTRUCTIONS)

    def test_mirror_generation_retries_transient_agent_failure(self):
        coach = KernelGuidedCoach.__new__(KernelGuidedCoach)
        coach.mirror_agent = _FlakyAgent(
            MirrorResponse(text="Here is the LLM reflective listening pass."),
            fail_times=1,
        )

        with patch("coach.chat.time.sleep", return_value=None):
            mirror = coach.mirror_formulation(FormulationGraph(turn_count=2))

        self.assertEqual("Here is the LLM reflective listening pass.", mirror.text)
        self.assertEqual(2, len(coach.mirror_agent.calls))

    def test_mirror_generation_does_not_use_deterministic_fallback_after_retries(self):
        coach = KernelGuidedCoach.__new__(KernelGuidedCoach)
        coach.mirror_agent = _FlakyAgent(
            MirrorResponse(text="This should not be returned."),
            fail_times=3,
        )

        with patch("coach.chat.time.sleep", return_value=None):
            with self.assertRaises(RuntimeError):
                coach.mirror_formulation(FormulationGraph(turn_count=2))

        self.assertEqual(3, len(coach.mirror_agent.calls))


if __name__ == "__main__":
    unittest.main()
