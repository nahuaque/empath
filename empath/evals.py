"""Offline evaluation fixtures for the therapeutic reasoning kernel.

These evals exercise the symbolic layer only. They intentionally avoid LLM
calls so regressions in relational hypotheses, safety filtering, and ranking
policy are fast to catch in local tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .therapeutic_kernel import CoachingState, TherapeuticReasoningKernel


HypothesisKey = tuple[str, str]


@dataclass(frozen=True)
class KernelEvalCase:
    """One expected behavior fixture for the kernel."""

    name: str
    state: CoachingState
    expected_hypotheses: tuple[HypothesisKey, ...] = ()
    forbidden_hypotheses: tuple[HypothesisKey, ...] = ()
    unsafe_interventions: tuple[str, ...] = ()
    acceptable_top: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class KernelEvalResult:
    """Evaluation result plus enough context to debug a failure."""

    case_name: str
    passed: bool
    failures: tuple[str, ...]
    hypotheses: tuple[HypothesisKey, ...]
    candidates: tuple[str, ...]
    safe_interventions: tuple[str, ...]
    top_candidate: str | None


def kernel_eval_cases() -> tuple[KernelEvalCase, ...]:
    """Return the curated offline eval set.

    The first pass targets breadth rather than exhaustive therapy coverage:
    emotional minimal disclosures, CBT distortions, REBT belief types, ACT stuck
    processes, and contraindication policies.
    """

    return (
        KernelEvalCase(
            name="minimal_sadness",
            state=_state(
                "minimal_sadness", "hi, I'm feeling sad today", emotions=("sad",)
            ),
            expected_hypotheses=(
                ("emotion", "sadness"),
                ("loop", "minimal_disclosure_sad_anxious"),
                ("policy", "minimal_disclosure"),
            ),
            forbidden_hypotheses=(("act", "values_unclear"),),
            acceptable_top=("validation", "gentle_check_in"),
        ),
        KernelEvalCase(
            name="minimal_anxiety",
            state=_state("minimal_anxiety", "I'm anxious today", emotions=("anxious",)),
            expected_hypotheses=(
                ("emotion", "anxiety"),
                ("loop", "minimal_disclosure_sad_anxious"),
                ("policy", "minimal_disclosure"),
            ),
            acceptable_top=("validation", "gentle_check_in"),
        ),
        KernelEvalCase(
            name="minimal_anger",
            state=_state("minimal_anger", "I'm angry today", emotions=("angry",)),
            expected_hypotheses=(
                ("emotion", "anger"),
                ("policy", "minimal_disclosure"),
            ),
            acceptable_top=("validation", "gentle_check_in"),
        ),
        KernelEvalCase(
            name="minimal_overwhelm",
            state=_state(
                "minimal_overwhelm",
                "I'm overwhelmed today",
                emotions=("overwhelmed",),
            ),
            expected_hypotheses=(
                ("emotion", "overwhelm"),
                ("policy", "minimal_disclosure"),
            ),
            acceptable_top=("validation", "gentle_check_in"),
        ),
        KernelEvalCase(
            name="shame_self_worth",
            state=_state(
                "shame_self_worth",
                "I made a mistake and I am a failure.",
                emotions=("shame",),
            ),
            expected_hypotheses=(
                ("emotion", "shame"),
                ("cbt", "global_labeling"),
                ("rebt", "self_downing"),
                ("act", "fusion"),
                ("loop", "shame_self_worth_fusion"),
            ),
            acceptable_top=(
                "validation",
                "cognitive_defusion",
                "unconditional_self_acceptance",
                "self_compassion",
            ),
        ),
        KernelEvalCase(
            name="investor_update_moderate",
            state=_state(
                "investor_update_moderate",
                (
                    "I keep putting off sending the investor update. If they see "
                    "the numbers are weak, they'll think I'm incompetent, and maybe "
                    "that means I'm not cut out to run this company. I know I should "
                    "send it, but I just keep avoiding it."
                ),
                emotions=("anxiety", "shame"),
                distress=5,
                values=("integrity", "leadership"),
            ),
            expected_hypotheses=(
                ("act", "experiential_avoidance"),
                ("act", "fusion"),
                ("cbt", "mind_reading"),
                ("cbt", "global_labeling"),
                ("cbt", "overgeneralization"),
                ("rebt", "demandingness"),
                ("rebt", "self_downing"),
                ("loop", "avoidance_identity_threat"),
                ("loop", "shame_self_worth_fusion"),
                ("loop", "valued_action_procrastination"),
            ),
            acceptable_top=(
                "validation",
                "acceptance_committed_action",
                "cognitive_defusion",
            ),
        ),
        KernelEvalCase(
            name="investor_update_high_distress",
            state=_state(
                "investor_update_high_distress",
                (
                    "I'm panicking and I can't breathe. I keep putting off the "
                    "investor update because they'll think I'm incompetent, and "
                    "that means I'm not cut out to run this company."
                ),
                emotions=("panic", "shame"),
                distress=9,
            ),
            expected_hypotheses=(
                ("policy", "high_distress"),
                ("policy", "needs_validation"),
                ("act", "experiential_avoidance"),
                ("act", "fusion"),
                ("cbt", "mind_reading"),
                ("rebt", "self_downing"),
                ("loop", "avoidance_identity_threat"),
                ("loop", "high_distress_gating"),
                ("loop", "shame_self_worth_fusion"),
            ),
            unsafe_interventions=("rebt_disputation",),
            acceptable_top=("validation", "present_moment_grounding"),
        ),
        KernelEvalCase(
            name="prototype_identity_avoidance",
            state=_state(
                "prototype_identity_avoidance",
                (
                    "I keep avoiding the prototype because if it is bad, it proves "
                    "I am not cut out for this."
                ),
                emotions=("anxiety", "shame"),
                behaviors=("avoidance",),
                preferred_modalities=("act",),
            ),
            expected_hypotheses=(
                ("act", "experiential_avoidance"),
                ("act", "fusion"),
                ("cbt", "global_labeling"),
                ("rebt", "self_downing"),
                ("loop", "avoidance_identity_threat"),
                ("loop", "shame_self_worth_fusion"),
            ),
            acceptable_top=("acceptance_committed_action", "cognitive_defusion"),
        ),
        KernelEvalCase(
            name="mind_reading_only",
            state=_state(
                "mind_reading_only",
                "They will think I am incompetent.",
            ),
            expected_hypotheses=(("cbt", "mind_reading"),),
            acceptable_top=("evidence_check",),
        ),
        KernelEvalCase(
            name="all_or_nothing_perfection",
            state=_state(
                "all_or_nothing_perfection",
                "If it is not perfect, it is a total failure.",
            ),
            expected_hypotheses=(("cbt", "all_or_nothing"),),
            acceptable_top=("continuum_technique",),
        ),
        KernelEvalCase(
            name="catastrophizing_everything_over",
            state=_state(
                "catastrophizing_everything_over",
                "If this goes badly, everything is over.",
            ),
            expected_hypotheses=(("cbt", "catastrophizing"),),
            acceptable_top=("decatastrophizing", "socratic_question"),
        ),
        KernelEvalCase(
            name="awfulizing_failure",
            state=_state(
                "awfulizing_failure",
                "It would be awful if I failed.",
            ),
            expected_hypotheses=(
                ("cbt", "catastrophizing"),
                ("rebt", "awfulizing"),
            ),
            acceptable_top=("decatastrophizing",),
        ),
        KernelEvalCase(
            name="low_frustration_tolerance",
            state=_state(
                "low_frustration_tolerance",
                "I cannot stand this discomfort.",
            ),
            expected_hypotheses=(("rebt", "low_frustration_tolerance"),),
            acceptable_top=("acceptance_practice", "frustration_tolerance_practice"),
        ),
        KernelEvalCase(
            name="demandingness",
            state=_state("demandingness", "I must get this right."),
            expected_hypotheses=(("rebt", "demandingness"),),
            acceptable_top=("preference_rewrite", "rebt_disputation"),
        ),
        KernelEvalCase(
            name="explicit_values_unclear",
            state=_state(
                "explicit_values_unclear",
                "I don't know what matters to me anymore.",
            ),
            expected_hypotheses=(
                ("act", "values_unclear"),
                ("focus", "values_direction"),
            ),
            acceptable_top=("values_clarification",),
        ),
        KernelEvalCase(
            name="plain_long_day_no_values_inference",
            state=_state("plain_long_day_no_values_inference", "I had a long day."),
            forbidden_hypotheses=(("act", "values_unclear"),),
        ),
        KernelEvalCase(
            name="rumination_loop",
            state=_state(
                "rumination_loop",
                "I can't stop thinking about the meeting.",
            ),
            expected_hypotheses=(("act", "rumination"), ("dbt", "mindfulness_need")),
            acceptable_top=(
                "present_moment_grounding",
                "cognitive_defusion",
                "mindfulness_observe_describe",
            ),
        ),
        KernelEvalCase(
            name="present_moment_disconnection",
            state=_state(
                "present_moment_disconnection",
                "I feel numb and checked out.",
            ),
            expected_hypotheses=(
                ("act", "present_moment_disconnection"),
                ("dbt", "mindfulness_need"),
            ),
            acceptable_top=("present_moment_grounding", "mindfulness_observe_describe"),
        ),
        KernelEvalCase(
            name="inaction",
            state=_state(
                "inaction",
                "I know what to do but I am stuck.",
                behaviors=("inaction",),
            ),
            expected_hypotheses=(("act", "inaction"), ("focus", "goal_activation")),
            acceptable_top=("committed_action",),
        ),
        KernelEvalCase(
            name="procrastination_without_identity",
            state=_state(
                "procrastination_without_identity",
                "I keep putting off sending the email.",
            ),
            expected_hypotheses=(
                ("act", "experiential_avoidance"),
                ("focus", "avoidance_escape"),
                ("focus", "motivation_persistence"),
            ),
            acceptable_top=("acceptance_committed_action",),
        ),
        KernelEvalCase(
            name="withdrawal_without_identity",
            state=_state(
                "withdrawal_without_identity",
                "I shut down and withdraw when conflict shows up.",
            ),
            expected_hypotheses=(
                ("act", "experiential_avoidance"),
                ("focus", "avoidance_escape"),
            ),
            acceptable_top=("acceptance_committed_action",),
        ),
        KernelEvalCase(
            name="self_downing_no_avoidance",
            state=_state(
                "self_downing_no_avoidance",
                "I am worthless when I mess up.",
            ),
            expected_hypotheses=(
                ("act", "fusion"),
                ("cbt", "global_labeling"),
                ("focus", "cognitive_belief_work"),
                ("focus", "self_efficacy"),
                ("rebt", "self_downing"),
            ),
            acceptable_top=("cognitive_defusion", "unconditional_self_acceptance"),
        ),
        KernelEvalCase(
            name="single_event_global_conclusion",
            state=_state(
                "single_event_global_conclusion",
                "One bad meeting means I am not good enough.",
            ),
            expected_hypotheses=(
                ("act", "fusion"),
                ("cbt", "overgeneralization"),
                ("cbt", "global_labeling"),
                ("focus", "cognitive_belief_work"),
                ("focus", "self_efficacy"),
                ("rebt", "self_downing"),
            ),
            acceptable_top=("cognitive_defusion", "unconditional_self_acceptance"),
        ),
        KernelEvalCase(
            name="sadness_without_values_jump",
            state=_state(
                "sadness_without_values_jump",
                "I'm sad and I do not know why.",
                emotions=("sadness",),
            ),
            expected_hypotheses=(("emotion", "sadness"),),
            forbidden_hypotheses=(("act", "values_unclear"),),
            acceptable_top=("validation",),
        ),
        KernelEvalCase(
            name="panic_no_identity",
            state=_state(
                "panic_no_identity",
                "I'm panicking and I can't breathe.",
                emotions=("panic",),
                distress=9,
            ),
            expected_hypotheses=(
                ("dbt", "crisis_survival"),
                ("emotion", "anxiety"),
                ("loop", "high_distress_gating"),
                ("policy", "high_distress"),
                ("policy", "needs_validation"),
            ),
            acceptable_top=("validation", "present_moment_grounding"),
        ),
        KernelEvalCase(
            name="identity_high_distress",
            state=_state(
                "identity_high_distress",
                "I'm panicking. One weak result means I am a failure.",
                emotions=("panic", "shame"),
                distress=9,
            ),
            expected_hypotheses=(
                ("policy", "high_distress"),
                ("act", "fusion"),
                ("rebt", "self_downing"),
                ("loop", "high_distress_gating"),
                ("loop", "shame_self_worth_fusion"),
            ),
            unsafe_interventions=("rebt_disputation",),
            acceptable_top=("validation", "present_moment_grounding"),
        ),
        KernelEvalCase(
            name="demandingness_high_distress",
            state=_state(
                "demandingness_high_distress",
                "I can't breathe. I must fix this now.",
                emotions=("anxiety",),
                distress=9,
            ),
            expected_hypotheses=(
                ("dbt", "crisis_survival"),
                ("policy", "high_distress"),
                ("rebt", "demandingness"),
            ),
            unsafe_interventions=("rebt_disputation",),
            acceptable_top=("validation", "present_moment_grounding"),
        ),
        KernelEvalCase(
            name="safety_risk_defers_cognitive_work",
            state=_state(
                "safety_risk_defers_cognitive_work",
                ("One mistake proves I am a failure and I want to hurt myself."),
                emotions=("shame",),
            ),
            expected_hypotheses=(
                ("policy", "safety_risk"),
                ("act", "fusion"),
                ("cbt", "overgeneralization"),
                ("cbt", "global_labeling"),
                ("rebt", "self_downing"),
            ),
            unsafe_interventions=(
                "behavioral_experiment",
                "cognitive_reframe",
                "rebt_disputation",
            ),
            acceptable_top=("safety_planning",),
        ),
        KernelEvalCase(
            name="trauma_grounding",
            state=_state(
                "trauma_grounding",
                "The flashback from the assault is here and I feel numb.",
            ),
            expected_hypotheses=(
                ("act", "present_moment_disconnection"),
                ("dbt", "mindfulness_need"),
            ),
            acceptable_top=("present_moment_grounding", "mindfulness_observe_describe"),
        ),
        KernelEvalCase(
            name="preferred_cbt_mind_reading",
            state=_state(
                "preferred_cbt_mind_reading",
                "Everyone will think I am incompetent.",
                preferred_modalities=("cbt",),
            ),
            expected_hypotheses=(("cbt", "mind_reading"),),
            acceptable_top=("evidence_check",),
        ),
        KernelEvalCase(
            name="valued_hard_conversation",
            state=_state(
                "valued_hard_conversation",
                "I care about honesty but I keep avoiding the hard conversation.",
                behaviors=("avoidance",),
                values=("honesty",),
            ),
            expected_hypotheses=(
                ("act", "experiential_avoidance"),
                ("focus", "avoidance_escape"),
                ("focus", "interpersonal_boundaries"),
                ("focus", "values_direction"),
            ),
            acceptable_top=("acceptance_committed_action",),
        ),
        KernelEvalCase(
            name="dbt_emotion_dysregulation",
            state=_state(
                "dbt_emotion_dysregulation",
                "I feel emotionally flooded and I am losing it.",
            ),
            expected_hypotheses=(("dbt", "emotion_dysregulation"),),
            acceptable_top=("emotion_regulation_check_facts",),
        ),
        KernelEvalCase(
            name="dbt_boundary_difficulty",
            state=_state(
                "dbt_boundary_difficulty",
                "I can't say no and I am afraid to ask for what I need.",
            ),
            expected_hypotheses=(
                ("dbt", "interpersonal_effectiveness_need"),
                ("focus", "interpersonal_boundaries"),
            ),
            acceptable_top=("interpersonal_effectiveness_script",),
        ),
        KernelEvalCase(
            name="dbt_self_invalidation",
            state=_state(
                "dbt_self_invalidation",
                "I shouldn't feel this angry. I feel stupid for feeling this way.",
                emotions=("anger",),
            ),
            expected_hypotheses=(("dbt", "self_invalidation"),),
            acceptable_top=("self_validation", "validation"),
        ),
        KernelEvalCase(
            name="focus_decision_problem_solving",
            state=_state(
                "focus_decision_problem_solving",
                "I can't decide which option to choose without guaranteed certainty.",
            ),
            expected_hypotheses=(("focus", "decision_problem_solving"),),
            acceptable_top=("decision_clarity_map", "wise_mind_check"),
        ),
        KernelEvalCase(
            name="focus_attention_environment",
            state=_state(
                "focus_attention_environment",
                "I need a better routine, fewer notifications, and a workspace that helps me focus.",
            ),
            expected_hypotheses=(
                ("focus", "attention_environment_design"),
                ("focus", "goal_activation"),
            ),
            acceptable_top=("environment_design_plan", "goal_action_planning"),
        ),
        KernelEvalCase(
            name="focus_resilience_review",
            state=_state(
                "focus_resilience_review",
                "After this setback I want to review what I learned and recommit.",
            ),
            expected_hypotheses=(
                ("focus", "integration_review"),
                ("focus", "resilience_recovery"),
            ),
            acceptable_top=("learning_review", "setback_recovery_plan"),
        ),
        KernelEvalCase(
            name="cbt_discounting_positive",
            state=_state(
                "cbt_discounting_positive",
                "The good feedback doesn't count. It was just luck and anyone could have done it.",
            ),
            expected_hypotheses=(("cbt", "discounting_positive"),),
            acceptable_top=("strength_evidence_log",),
        ),
        KernelEvalCase(
            name="cbt_emotional_reasoning_personalization",
            state=_state(
                "cbt_emotional_reasoning_personalization",
                "Because I feel guilty, it means this is all my fault.",
            ),
            expected_hypotheses=(
                ("cbt", "emotional_reasoning"),
                ("cbt", "personalization"),
            ),
            acceptable_top=("emotion_fact_separation", "responsibility_pie"),
        ),
        KernelEvalCase(
            name="rebt_approval_certainty_failure",
            state=_state(
                "rebt_approval_certainty_failure",
                "I need everyone to approve of me, I need to know for sure before I act, and I can't fail.",
            ),
            expected_hypotheses=(
                ("rebt", "approval_demandingness"),
                ("rebt", "certainty_demandingness"),
                ("rebt", "failure_intolerance"),
            ),
            acceptable_top=(
                "approval_preference_rewrite",
                "uncertainty_tolerance_practice",
                "failure_tolerance_reframe",
                "preference_rewrite",
            ),
        ),
        KernelEvalCase(
            name="act_control_struggle_values_gap",
            state=_state(
                "act_control_struggle_values_gap",
                (
                    "I need this anxiety to go away before I work on the launch plan. "
                    "I value courage but I keep putting it off."
                ),
                behaviors=("procrastination",),
                values=("courage",),
                goals=("work on the launch plan",),
            ),
            expected_hypotheses=(
                ("act", "unworkable_control"),
                ("act", "values_action_gap"),
                ("loop", "control_struggle_loop"),
            ),
            acceptable_top=(
                "acceptance_practice",
                "values_aligned_next_step",
                "acceptance_committed_action",
            ),
        ),
        KernelEvalCase(
            name="dbt_wise_mind_certainty_avoidance",
            state=_state(
                "dbt_wise_mind_certainty_avoidance",
                "I can't decide until I have perfect information, so I keep avoiding the choice.",
            ),
            expected_hypotheses=(
                ("dbt", "wise_mind_need"),
                ("loop", "certainty_avoidance_loop"),
            ),
            acceptable_top=("wise_mind_check", "decision_clarity_map"),
        ),
        KernelEvalCase(
            name="dbt_vulnerability_factors",
            state=_state(
                "dbt_vulnerability_factors",
                "I haven't slept or eaten, and now I'm overwhelmed and losing it.",
                emotions=("overwhelm",),
            ),
            expected_hypotheses=(
                ("dbt", "vulnerability_factors"),
                ("loop", "vulnerability_distress_loop"),
            ),
            acceptable_top=("vulnerability_reduction_plan", "validation"),
        ),
        KernelEvalCase(
            name="mbsr_stress_body_tension",
            state=CoachingState(
                state_id="mbsr_stress_body_tension",
                utterance=(
                    "I'm stressed and under pressure. My shoulders are tense "
                    "and I keep overthinking everything."
                ),
                thoughts=(
                    "I'm stressed and under pressure. My shoulders are tense and I keep overthinking everything.",
                ),
                emotions=("anxiety",),
                bodily_sensations=("tense shoulders",),
                behaviors=("rumination",),
            ),
            expected_hypotheses=(
                ("mbsr", "stress_load"),
                ("mbsr", "somatic_stress"),
                ("mbsr", "rumination_stress"),
            ),
            acceptable_top=(
                "mindful_breathing_space",
                "body_scan_check_in",
                "mindful_observe_return",
                "validation",
                "present_moment_grounding",
            ),
        ),
        KernelEvalCase(
            name="mbsr_autopilot_reactivity",
            state=_state(
                "mbsr_autopilot_reactivity",
                "I'm on autopilot and about to snap at my team.",
            ),
            expected_hypotheses=(("mbsr", "autopilot_reactivity"),),
            acceptable_top=("mindful_pause", "distress_tolerance_pause", "validation"),
        ),
        KernelEvalCase(
            name="focusing_unclear_felt_sense",
            state=CoachingState(
                state_id="focusing_unclear_felt_sense",
                utterance=(
                    "There is a tight knot in my chest and I can't put words to it. "
                    "Something feels off."
                ),
                thoughts=(
                    "There is a tight knot in my chest and I can't put words to it. Something feels off.",
                ),
                bodily_sensations=("tight knot in chest",),
            ),
            expected_hypotheses=(
                ("focusing", "felt_sense_contact"),
                ("focusing", "unclear_felt_meaning"),
                ("focusing", "symbolization_needed"),
            ),
            acceptable_top=(
                "felt_sense_pause",
                "felt_sense_description",
                "resonant_word_check",
            ),
        ),
        KernelEvalCase(
            name="focusing_inner_critic",
            state=_state(
                "focusing_inner_critic",
                (
                    "Part of me says I am useless, like a harsh inner critic, "
                    "and I can feel pressure in my throat."
                ),
            ),
            expected_hypotheses=(
                ("focusing", "inner_critic_presence"),
                ("focusing", "felt_sense_contact"),
            ),
            acceptable_top=(
                "inner_critic_distance",
                "felt_sense_pause",
                "felt_sense_description",
            ),
        ),
    )


def evaluate_case(case: KernelEvalCase) -> KernelEvalResult:
    """Evaluate one fixture against a fresh kernel."""

    kernel = TherapeuticReasoningKernel()
    kernel.add_state(case.state)

    hypotheses = tuple(
        (hypothesis.source, hypothesis.pattern)
        for hypothesis in kernel.hypotheses_for(case.state.state_id)
    )
    hypothesis_set = set(hypotheses)
    candidates = kernel.candidate_intervention_names(case.state.state_id)
    candidate_set = set(candidates)
    safe_interventions = kernel.safe_intervention_names(case.state.state_id)
    safe_set = set(safe_interventions)
    ranked = kernel.ranked_interventions(case.state.state_id, limit=1)
    top_candidate = ranked[0].intervention if ranked else None

    failures: list[str] = []

    for expected in case.expected_hypotheses:
        if expected not in hypothesis_set:
            failures.append(f"missing hypothesis {expected!r}")

    for forbidden in case.forbidden_hypotheses:
        if forbidden in hypothesis_set:
            failures.append(f"forbidden hypothesis present {forbidden!r}")

    for intervention in case.unsafe_interventions:
        if intervention not in candidate_set:
            failures.append(f"unsafe intervention is not a candidate {intervention!r}")
            continue
        if intervention in safe_set:
            failures.append(f"unsafe intervention was considered safe {intervention!r}")
        reasons = kernel.contraindication_reasons(case.state.state_id, intervention)
        if not reasons:
            failures.append(
                f"unsafe intervention lacks contraindication {intervention!r}"
            )

    if case.acceptable_top and top_candidate not in set(case.acceptable_top):
        failures.append(
            "top candidate "
            f"{top_candidate!r} not in acceptable set {case.acceptable_top!r}"
        )

    return KernelEvalResult(
        case_name=case.name,
        passed=not failures,
        failures=tuple(failures),
        hypotheses=hypotheses,
        candidates=candidates,
        safe_interventions=safe_interventions,
        top_candidate=top_candidate,
    )


def run_kernel_evals(
    cases: Iterable[KernelEvalCase] | None = None,
) -> tuple[KernelEvalResult, ...]:
    """Run the offline kernel eval suite."""

    return tuple(evaluate_case(case) for case in (cases or kernel_eval_cases()))


def format_eval_failures(results: Iterable[KernelEvalResult]) -> str:
    """Format failing results for unittest and CLI output."""

    lines: list[str] = []
    for result in results:
        if result.passed:
            continue
        lines.append(f"{result.case_name}:")
        for failure in result.failures:
            lines.append(f"  - {failure}")
        lines.append(f"  hypotheses: {result.hypotheses}")
        lines.append(f"  candidates: {result.candidates}")
        lines.append(f"  safe: {result.safe_interventions}")
        lines.append(f"  top: {result.top_candidate!r}")
    return "\n".join(lines)


def main() -> None:
    results = run_kernel_evals()
    failures = tuple(result for result in results if not result.passed)
    print(f"Kernel evals: {len(results) - len(failures)}/{len(results)} passed")
    if failures:
        print(format_eval_failures(failures))
        raise SystemExit(1)


def _state(
    state_id: str,
    text: str,
    *,
    emotions: tuple[str, ...] = (),
    behaviors: tuple[str, ...] = (),
    distress: int | None = None,
    features: tuple[str, ...] = (),
    values: tuple[str, ...] = (),
    goals: tuple[str, ...] = (),
    preferred_modalities: tuple[str, ...] = (),
) -> CoachingState:
    return CoachingState(
        state_id=state_id,
        utterance=text,
        thoughts=(text,),
        emotions=emotions,
        behaviors=behaviors,
        distress=distress,
        features=features,
        values=values,
        goals=goals,
        preferred_modalities=preferred_modalities,
    )


if __name__ == "__main__":
    main()
