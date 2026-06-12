"""A small miniKanren therapeutic reasoning kernel.

The kernel treats ACT/CBT/REBT classifications as coaching hypotheses, not
diagnoses. In a full app, an LLM or parser should supply structured
observations; the light text heuristics here are only an MVP convenience.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import re
from typing import Any

from kanren import Relation, conde, eq, fact, facts, isvar, lall, reify, run, var

from .therapeutic_systems import TherapeuticSystem, default_systems


@dataclass(frozen=True)
class CoachingState:
    """Structured observations extracted from a user's utterance."""

    state_id: str
    utterance: str = ""
    situations: tuple[str, ...] = ()
    concerns: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    challenges: tuple[str, ...] = ()
    objectives: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()
    key_results: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    obstacles: tuple[str, ...] = ()
    implementation_intentions: tuple[str, ...] = ()
    waiting_for: tuple[str, ...] = ()
    time_horizons: tuple[str, ...] = ()
    success_measures: tuple[str, ...] = ()
    stakes: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    thoughts: tuple[str, ...] = ()
    beliefs: tuple[str, ...] = ()
    emotions: tuple[str, ...] = ()
    bodily_sensations: tuple[str, ...] = ()
    urges: tuple[str, ...] = ()
    behaviors: tuple[str, ...] = ()
    consequences: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    distress: int | None = None
    features: tuple[str, ...] = ()
    thought_features: Mapping[str, Iterable[str]] = field(default_factory=dict)
    belief_features: Mapping[str, Iterable[str]] = field(default_factory=dict)
    preferred_modalities: tuple[str, ...] = ()
    recent_interventions: tuple[str, ...] = ()


@dataclass(frozen=True, order=True)
class Hypothesis:
    """A pattern inferred by the relational kernel."""

    source: str
    pattern: str


@dataclass(frozen=True)
class InterventionCandidate:
    """A ranked intervention candidate plus the hypotheses that justify it."""

    intervention: str
    score: float
    modality: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...]
    exercise: str | None = None
    contraindications: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterventionRecipe:
    """A relational multi-step intervention recipe."""

    recipe: str
    score: float
    steps: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...]
    rationale: str | None = None
    contraindications: tuple[str, ...] = ()


@dataclass(frozen=True)
class DifferentialFormulation:
    """A tentative case formulation option with evidence and a discriminator."""

    formulation: str
    score: float
    label: str
    summary: str
    evidence: tuple[Hypothesis, ...]
    missing_evidence: tuple[str, ...] = ()
    discriminating_question: str | None = None
    focus: str | None = None
    interventions: tuple[str, ...] = ()
    recipes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClarifyingMove:
    """A next observation that would reduce formulation uncertainty."""

    move: str
    priority: float
    kind: str
    question: str
    rationale: str
    target_formulations: tuple[str, ...]
    supported_by: tuple[Hypothesis, ...]
    missing_evidence: tuple[str, ...] = ()
    expected_information: tuple[str, ...] = ()
    intervention: str | None = None


class TherapeuticReasoningKernel:
    """Relational ACT/CBT/REBT reasoning over structured coaching states."""

    def __init__(
        self,
        *,
        auto_features: bool = True,
        systems: Iterable[TherapeuticSystem] | None = None,
    ) -> None:
        self.auto_features = auto_features
        self._states: dict[str, CoachingState] = {}
        self.systems = tuple(systems) if systems is not None else default_systems()
        self.systems_by_source = {system.source: system for system in self.systems}

        self.has_situation = Relation("has_situation")
        self.has_concern = Relation("has_concern")
        self.has_task = Relation("has_task")
        self.has_challenge = Relation("has_challenge")
        self.has_objective = Relation("has_objective")
        self.has_project = Relation("has_project")
        self.has_key_result = Relation("has_key_result")
        self.has_next_action = Relation("has_next_action")
        self.has_obstacle = Relation("has_obstacle")
        self.has_implementation_intention = Relation("has_implementation_intention")
        self.has_waiting_for = Relation("has_waiting_for")
        self.has_time_horizon = Relation("has_time_horizon")
        self.has_success_measure = Relation("has_success_measure")
        self.has_stake = Relation("has_stake")
        self.has_domain = Relation("has_domain")
        self.has_thought = Relation("has_thought")
        self.has_belief = Relation("has_belief")
        self.has_emotion = Relation("has_emotion")
        self.has_bodily_sensation = Relation("has_bodily_sensation")
        self.has_urge = Relation("has_urge")
        self.has_behavior = Relation("has_behavior")
        self.has_consequence = Relation("has_consequence")
        self.has_value = Relation("has_value")
        self.has_goal = Relation("has_goal")
        self.activation_level = Relation("activation_level")
        self.observation_text = Relation("observation_text")
        self.state_feature = Relation("state_feature")
        self.thought_feature = Relation("thought_feature")
        self.belief_feature = Relation("belief_feature")
        self.preferred_modality = Relation("preferred_modality")
        self.recent_intervention = Relation("recent_intervention")

        self.feature_distortion = getattr(
            self.systems_by_source.get("cbt"),
            "feature_distortion",
            Relation("feature_distortion"),
        )
        self.feature_rebt_belief = getattr(
            self.systems_by_source.get("rebt"),
            "feature_rebt_belief",
            Relation("feature_rebt_belief"),
        )
        self.emotion_pattern = Relation("emotion_pattern")
        self.policy_pattern = Relation("policy_pattern")
        self.pattern_intervention = Relation("pattern_intervention")
        self.intervention_modality = Relation("intervention_modality")
        self.intervention_exercise = Relation("intervention_exercise")
        self.high_intensity_intervention = Relation("high_intensity_intervention")
        self.premature_without_validation = Relation("premature_without_validation")
        self.safety_deferred_intervention = Relation("safety_deferred_intervention")
        self.pattern_recipe = Relation("pattern_recipe")
        self.recipe_step = Relation("recipe_step")
        self.recipe_rationale = Relation("recipe_rationale")
        self.formulation_pattern = Relation("formulation_pattern")
        self.formulation_label = Relation("formulation_label")
        self.formulation_summary = Relation("formulation_summary")
        self.formulation_discriminator = Relation("formulation_discriminator")
        self.formulation_focus = Relation("formulation_focus")

        self._load_default_ontology()

    def _load_default_ontology(self) -> None:
        facts(
            self.emotion_pattern,
            ("sadness", "sadness"),
            ("anxiety", "anxiety"),
            ("shame", "shame"),
            ("anger", "anger"),
            ("overwhelm", "overwhelm"),
        )
        facts(
            self.policy_pattern,
            ("high_distress",),
            ("needs_validation",),
            ("minimal_disclosure",),
            ("safety_risk",),
        )
        facts(
            self.pattern_intervention,
            ("sadness", "validation"),
            ("sadness", "gentle_check_in"),
            ("anxiety", "validation"),
            ("anxiety", "present_moment_grounding"),
            ("shame", "validation"),
            ("shame", "self_compassion"),
            ("anger", "validation"),
            ("anger", "needs_exploration"),
            ("overwhelm", "validation"),
            ("overwhelm", "present_moment_grounding"),
            ("minimal_disclosure", "validation"),
            ("minimal_disclosure", "gentle_check_in"),
            ("high_distress", "validation"),
            ("high_distress", "present_moment_grounding"),
            ("needs_validation", "validation"),
            ("safety_risk", "safety_planning"),
        )
        facts(
            self.intervention_modality,
            ("validation", "supportive"),
            ("gentle_check_in", "supportive"),
            ("self_compassion", "supportive"),
            ("needs_exploration", "supportive"),
            ("present_moment_grounding", "act"),
            ("safety_planning", "safety"),
        )
        facts(
            self.intervention_exercise,
            (
                "validation",
                "Reflect the feeling and the stakes before offering a technique.",
            ),
            (
                "gentle_check_in",
                "Ask one gentle question about what the feeling may be connected to.",
            ),
            (
                "self_compassion",
                "Offer a kind, non-evaluative response to the painful feeling.",
            ),
            (
                "needs_exploration",
                "Look for the need, boundary, or value underneath the emotion.",
            ),
            (
                "present_moment_grounding",
                "Orient to breath, feet, and five things currently visible.",
            ),
            (
                "safety_planning",
                "Pause coaching content and address immediate safety and support.",
            ),
        )
        facts(
            self.high_intensity_intervention,
            ("exposure_exercise",),
        )
        facts(
            self.pattern_recipe,
            ("minimal_disclosure", "validate_then_check_in"),
            ("minimal_disclosure_sad_anxious", "validate_then_check_in"),
            ("sadness", "validate_then_check_in"),
            ("anxiety", "validate_then_ground"),
            ("overwhelm", "stabilize_then_choose"),
            ("high_distress", "stabilize_then_choose"),
            ("crisis_survival", "stabilize_then_choose"),
            ("safety_risk", "safety_first"),
            ("avoidance_identity_threat", "validate_defuse_act"),
            ("experiential_avoidance", "validate_accept_act"),
            ("values_action_gap", "values_to_micro_action"),
            ("valued_action_procrastination", "values_to_micro_action"),
            ("fusion", "validate_defuse_act"),
            ("self_as_content", "validate_self_context"),
            ("shame_self_worth_fusion", "validate_self_worth"),
            ("self_downing", "validate_self_worth"),
            ("global_labeling", "validate_self_worth"),
            ("mind_reading", "validate_check_facts"),
            ("approval_threat_loop", "validate_check_facts"),
            ("demandingness", "validate_preference_rewrite"),
            ("low_frustration_tolerance", "validate_tolerance_action"),
            ("certainty_avoidance_loop", "wise_mind_decide_act"),
            ("wise_mind_need", "wise_mind_decide_act"),
            ("emotion_dysregulation", "regulate_check_act"),
            ("vulnerability_distress_loop", "reduce_vulnerability_regulate"),
            ("vulnerability_factors", "reduce_vulnerability_regulate"),
            ("interpersonal_effectiveness_need", "validate_interpersonal_script"),
            ("interpersonal_boundaries", "validate_interpersonal_script"),
            ("control_struggle_loop", "acceptance_willing_action"),
            ("unworkable_control", "acceptance_willing_action"),
            ("felt_sense_contact", "pause_describe_resonate"),
            ("unclear_felt_meaning", "pause_describe_resonate"),
            ("symbolization_needed", "describe_resonate_track"),
            ("inner_critic_presence", "validate_distance_resonate"),
            ("felt_shift_possible", "pause_track_shift"),
        )
        facts(
            self.recipe_step,
            ("validate_then_check_in", 0, "validation"),
            ("validate_then_check_in", 1, "gentle_check_in"),
            ("validate_then_ground", 0, "validation"),
            ("validate_then_ground", 1, "present_moment_grounding"),
            ("stabilize_then_choose", 0, "validation"),
            ("stabilize_then_choose", 1, "present_moment_grounding"),
            ("stabilize_then_choose", 2, "gentle_check_in"),
            ("safety_first", 0, "safety_planning"),
            ("validate_defuse_act", 0, "validation"),
            ("validate_defuse_act", 1, "cognitive_defusion"),
            ("validate_defuse_act", 2, "acceptance_committed_action"),
            ("validate_accept_act", 0, "validation"),
            ("validate_accept_act", 1, "acceptance_practice"),
            ("validate_accept_act", 2, "acceptance_committed_action"),
            ("values_to_micro_action", 0, "validation"),
            ("values_to_micro_action", 1, "values_aligned_next_step"),
            ("values_to_micro_action", 2, "committed_action"),
            ("validate_self_context", 0, "validation"),
            ("validate_self_context", 1, "self_as_context"),
            ("validate_self_context", 2, "cognitive_defusion"),
            ("validate_self_worth", 0, "validation"),
            ("validate_self_worth", 1, "self_compassion"),
            ("validate_self_worth", 2, "unconditional_self_acceptance"),
            ("validate_check_facts", 0, "validation"),
            ("validate_check_facts", 1, "evidence_check"),
            ("validate_preference_rewrite", 0, "validation"),
            ("validate_preference_rewrite", 1, "preference_rewrite"),
            ("validate_tolerance_action", 0, "validation"),
            ("validate_tolerance_action", 1, "frustration_tolerance_practice"),
            ("validate_tolerance_action", 2, "committed_action"),
            ("wise_mind_decide_act", 0, "validation"),
            ("wise_mind_decide_act", 1, "wise_mind_check"),
            ("wise_mind_decide_act", 2, "committed_action"),
            ("regulate_check_act", 0, "validation"),
            ("regulate_check_act", 1, "emotion_regulation_check_facts"),
            ("regulate_check_act", 2, "opposite_action_planning"),
            ("reduce_vulnerability_regulate", 0, "validation"),
            ("reduce_vulnerability_regulate", 1, "vulnerability_reduction_plan"),
            ("reduce_vulnerability_regulate", 2, "self_soothing"),
            ("validate_interpersonal_script", 0, "validation"),
            ("validate_interpersonal_script", 1, "interpersonal_effectiveness_script"),
            ("acceptance_willing_action", 0, "validation"),
            ("acceptance_willing_action", 1, "acceptance_practice"),
            ("acceptance_willing_action", 2, "willingness_practice"),
            ("pause_describe_resonate", 0, "validation"),
            ("pause_describe_resonate", 1, "felt_sense_pause"),
            ("pause_describe_resonate", 2, "felt_sense_description"),
            ("pause_describe_resonate", 3, "resonant_word_check"),
            ("describe_resonate_track", 0, "validation"),
            ("describe_resonate_track", 1, "felt_sense_description"),
            ("describe_resonate_track", 2, "resonant_word_check"),
            ("describe_resonate_track", 3, "felt_shift_tracking"),
            ("validate_distance_resonate", 0, "validation"),
            ("validate_distance_resonate", 1, "inner_critic_distance"),
            ("validate_distance_resonate", 2, "resonant_word_check"),
            ("pause_track_shift", 0, "validation"),
            ("pause_track_shift", 1, "felt_sense_pause"),
            ("pause_track_shift", 2, "felt_shift_tracking"),
        )
        facts(
            self.recipe_rationale,
            (
                "validate_then_check_in",
                "Start with validation, then ask for one gentle bit of context.",
            ),
            (
                "validate_then_ground",
                "Validate the feeling, then reduce activation before exploring.",
            ),
            (
                "stabilize_then_choose",
                "Use stabilization before problem solving or belief work.",
            ),
            ("safety_first", "Address immediate safety before any coaching technique."),
            (
                "validate_defuse_act",
                "Soften identity-fused thoughts, then move toward a valued action.",
            ),
            (
                "validate_accept_act",
                "Make room for discomfort and take one small approach step.",
            ),
            ("values_to_micro_action", "Reconnect the value to a visible next action."),
            (
                "validate_self_context",
                "Separate the observer from the self-story before choosing a move.",
            ),
            (
                "validate_self_worth",
                "Protect worth from becoming fused with performance or approval.",
            ),
            (
                "validate_check_facts",
                "Validate the concern, then separate facts from predictions.",
            ),
            (
                "validate_preference_rewrite",
                "Move from rigid demand to flexible preference after validation.",
            ),
            (
                "validate_tolerance_action",
                "Build tolerable contact with discomfort, then choose a small action.",
            ),
            (
                "wise_mind_decide_act",
                "Balance emotion and facts, then take a reversible step.",
            ),
            (
                "regulate_check_act",
                "Regulate first, check the emotion-action loop, then choose skillfully.",
            ),
            (
                "reduce_vulnerability_regulate",
                "Lower biological/contextual vulnerability before solving.",
            ),
            (
                "validate_interpersonal_script",
                "Validate the stakes, then script an effective ask or boundary.",
            ),
            (
                "acceptance_willing_action",
                "Notice control struggle, practice willingness, and take a small step.",
            ),
            (
                "pause_describe_resonate",
                "Slow down, contact the felt sense, describe it, and check which words resonate.",
            ),
            (
                "describe_resonate_track",
                "Find language that fits the body sense and watch for a felt shift.",
            ),
            (
                "validate_distance_resonate",
                "Validate the pain, create space from the critic, then check what words fit.",
            ),
            (
                "pause_track_shift",
                "Pause with the body sense and track whether anything changes as it is named.",
            ),
        )
        facts(
            self.formulation_pattern,
            ("safety_first", "safety_risk"),
            ("safety_first", "crisis_survival"),
            ("safety_first", "high_distress"),
            ("high_distress_first", "high_distress"),
            ("high_distress_first", "high_distress_gating"),
            ("high_distress_first", "crisis_survival"),
            ("high_distress_first", "overwhelm"),
            ("minimal_disclosure_affect", "minimal_disclosure"),
            ("minimal_disclosure_affect", "minimal_disclosure_sad_anxious"),
            ("minimal_disclosure_affect", "sadness"),
            ("minimal_disclosure_affect", "anxiety"),
            ("avoidance_identity_threat", "avoidance_identity_threat"),
            ("avoidance_identity_threat", "experiential_avoidance"),
            ("avoidance_identity_threat", "fusion"),
            ("avoidance_identity_threat", "self_as_content"),
            ("avoidance_identity_threat", "self_downing"),
            ("avoidance_identity_threat", "global_labeling"),
            ("avoidance_identity_threat", "anxiety"),
            ("avoidance_identity_threat", "shame"),
            ("shame_self_worth_fusion", "shame_self_worth_fusion"),
            ("shame_self_worth_fusion", "shame"),
            ("shame_self_worth_fusion", "self_downing"),
            ("shame_self_worth_fusion", "fusion"),
            ("shame_self_worth_fusion", "self_as_content"),
            ("shame_self_worth_fusion", "self_efficacy"),
            ("valued_action_procrastination", "valued_action_procrastination"),
            ("valued_action_procrastination", "values_action_gap"),
            ("valued_action_procrastination", "experiential_avoidance"),
            ("valued_action_procrastination", "goal_activation"),
            ("valued_action_procrastination", "motivation_persistence"),
            ("certainty_decision_loop", "certainty_avoidance_loop"),
            ("certainty_decision_loop", "wise_mind_need"),
            ("certainty_decision_loop", "decision_problem_solving"),
            ("certainty_decision_loop", "certainty_demandingness"),
            ("approval_interpersonal_threat", "approval_threat_loop"),
            ("approval_interpersonal_threat", "approval_demandingness"),
            ("approval_interpersonal_threat", "interpersonal_effectiveness_need"),
            ("approval_interpersonal_threat", "interpersonal_boundaries"),
            ("approval_interpersonal_threat", "mind_reading"),
            ("control_struggle", "control_struggle_loop"),
            ("control_struggle", "unworkable_control"),
            ("control_struggle", "unwillingness"),
            ("control_struggle", "motivation_persistence"),
            ("vulnerability_distress", "vulnerability_distress_loop"),
            ("vulnerability_distress", "vulnerability_factors"),
            ("vulnerability_distress", "emotion_dysregulation"),
            ("vulnerability_distress", "emotion_distress_regulation"),
            ("felt_sense_unclear_meaning", "felt_sense_contact"),
            ("felt_sense_unclear_meaning", "unclear_felt_meaning"),
            ("felt_sense_unclear_meaning", "symbolization_needed"),
            ("felt_sense_unclear_meaning", "inner_critic_presence"),
            ("felt_sense_unclear_meaning", "felt_shift_possible"),
        )
        facts(
            self.formulation_label,
            ("safety_first", "Safety first"),
            ("high_distress_first", "High distress gating"),
            ("minimal_disclosure_affect", "Minimal disclosure affect check-in"),
            ("avoidance_identity_threat", "Avoidance plus identity threat"),
            ("shame_self_worth_fusion", "Shame and self-worth fusion"),
            ("valued_action_procrastination", "Procrastination around valued action"),
            ("certainty_decision_loop", "Certainty and decision loop"),
            ("approval_interpersonal_threat", "Approval or interpersonal threat"),
            ("control_struggle", "Control struggle"),
            ("vulnerability_distress", "Vulnerability-driven distress"),
            ("felt_sense_unclear_meaning", "Felt sense needs symbolizing"),
        )
        facts(
            self.formulation_summary,
            (
                "safety_first",
                "Safety cues may need immediate support before ordinary coaching.",
            ),
            (
                "high_distress_first",
                "The main constraint may be activation level, so stabilization comes before analysis.",
            ),
            (
                "minimal_disclosure_affect",
                "There is a real feeling but little context yet, so the best map is gentle and provisional.",
            ),
            (
                "avoidance_identity_threat",
                "Avoidance may be protecting against what the task would seem to say about the self.",
            ),
            (
                "shame_self_worth_fusion",
                "A painful self-story may be treating worth or capability as if it depends on this situation.",
            ),
            (
                "valued_action_procrastination",
                "The stuck point may be converting a meaningful direction into a small doable action.",
            ),
            (
                "certainty_decision_loop",
                "The system may be waiting for certainty before taking a reversible step.",
            ),
            (
                "approval_interpersonal_threat",
                "The pressure may center on judgment, approval, conflict, or asking for what is needed.",
            ),
            (
                "control_struggle",
                "Effort may be going into eliminating the feeling or thought before acting.",
            ),
            (
                "vulnerability_distress",
                "Distress may be amplified by depletion, overload, or other vulnerability factors.",
            ),
            (
                "felt_sense_unclear_meaning",
                "The next useful move may be to contact the bodily felt sense and find words or images that fit.",
            ),
        )
        facts(
            self.formulation_discriminator,
            (
                "safety_first",
                "Is there any immediate risk of harm or need for real-time support?",
            ),
            (
                "high_distress_first",
                "Is your system too activated for problem-solving to be useful yet?",
            ),
            (
                "minimal_disclosure_affect",
                "Is the feeling tied to something specific, or is it more of a general mood right now?",
            ),
            (
                "avoidance_identity_threat",
                "Is the hardest part the task itself, or what the outcome might seem to prove about you?",
            ),
            (
                "shame_self_worth_fusion",
                "Does this feel like a painful event, or like a verdict on your worth or capability?",
            ),
            (
                "valued_action_procrastination",
                "What specific action matters here, and what small part of it are you avoiding?",
            ),
            (
                "certainty_decision_loop",
                "What decision would become possible if you did not need complete certainty first?",
            ),
            (
                "approval_interpersonal_threat",
                "Is the main fear about the practical outcome, or about another person's judgment or response?",
            ),
            (
                "control_struggle",
                "Are you trying to make the feeling go away before you allow yourself to act?",
            ),
            (
                "vulnerability_distress",
                "Would the same problem feel different if sleep, food, workload, or overload were less strained?",
            ),
            (
                "felt_sense_unclear_meaning",
                "Is there a vague bodily sense here that seems to know more than you can put into words yet?",
            ),
        )
        facts(
            self.formulation_focus,
            ("safety_first", "safety"),
            ("high_distress_first", "stabilization"),
            ("minimal_disclosure_affect", "gentle exploration"),
            ("avoidance_identity_threat", "defusion and approach"),
            ("shame_self_worth_fusion", "self-worth protection"),
            ("valued_action_procrastination", "values to action"),
            ("certainty_decision_loop", "decision under uncertainty"),
            ("approval_interpersonal_threat", "interpersonal effectiveness"),
            ("control_struggle", "acceptance and willingness"),
            ("vulnerability_distress", "reduce vulnerability"),
            ("felt_sense_unclear_meaning", "felt sense and symbolization"),
        )
        for system in self.systems:
            system.load_ontology(self)

    def add_state(self, state: CoachingState) -> None:
        """Add one structured state to the relational fact base."""

        state_id = str(state.state_id)
        self._states[state_id] = state

        for situation in _strings(state.situations):
            fact(self.has_situation, state_id, situation)
        for concern in _strings(state.concerns):
            fact(self.has_concern, state_id, concern)
        for task in _strings(state.tasks):
            fact(self.has_task, state_id, task)
        for challenge in _strings(state.challenges):
            fact(self.has_challenge, state_id, challenge)
        for objective in _strings(state.objectives):
            fact(self.has_objective, state_id, objective)
        for project in _strings(state.projects):
            fact(self.has_project, state_id, project)
        for key_result in _strings(state.key_results):
            fact(self.has_key_result, state_id, key_result)
        for next_action in _strings(state.next_actions):
            fact(self.has_next_action, state_id, next_action)
        for obstacle in _strings(state.obstacles):
            fact(self.has_obstacle, state_id, obstacle)
        for intention in _strings(state.implementation_intentions):
            fact(self.has_implementation_intention, state_id, intention)
        for waiting_for in _strings(state.waiting_for):
            fact(self.has_waiting_for, state_id, waiting_for)
        for horizon in _strings(state.time_horizons):
            fact(self.has_time_horizon, state_id, horizon)
        for measure in _strings(state.success_measures):
            fact(self.has_success_measure, state_id, measure)
        for stake in _strings(state.stakes):
            fact(self.has_stake, state_id, stake)
        for domain in _labels(state.domains):
            fact(self.has_domain, state_id, domain)
        for emotion in _labels(state.emotions):
            fact(self.has_emotion, state_id, emotion)
        for sensation in _labels(state.bodily_sensations):
            fact(self.has_bodily_sensation, state_id, sensation)
        for urge in _labels(state.urges):
            fact(self.has_urge, state_id, urge)
        for consequence in _strings(state.consequences):
            fact(self.has_consequence, state_id, consequence)
        for value in _labels(state.values):
            fact(self.has_value, state_id, value)
        for goal in _strings(state.goals):
            fact(self.has_goal, state_id, goal)
        for modality in _labels(state.preferred_modalities):
            fact(self.preferred_modality, state_id, modality)
        for intervention in _labels(state.recent_interventions):
            fact(self.recent_intervention, state_id, intervention)

        behaviors = set(_labels(state.behaviors))
        if self.auto_features:
            behaviors.update(_infer_behaviors(state.utterance))
        for behavior in sorted(behaviors):
            fact(self.has_behavior, state_id, behavior)

        for index, thought in enumerate(_strings(state.thoughts)):
            thought_id = f"{state_id}:thought:{index}"
            fact(self.has_thought, state_id, thought_id)
            fact(self.observation_text, thought_id, thought)
            features = set(_labels(state.thought_features.get(thought, ())))
            if self.auto_features:
                features.update(_infer_text_features(thought))
            for feature in sorted(features):
                fact(self.thought_feature, thought_id, feature)

        for index, belief in enumerate(_strings(state.beliefs)):
            belief_id = f"{state_id}:belief:{index}"
            fact(self.has_belief, state_id, belief_id)
            fact(self.observation_text, belief_id, belief)
            features = set(_labels(state.belief_features.get(belief, ())))
            if self.auto_features:
                features.update(_infer_text_features(belief))
            for feature in sorted(features):
                fact(self.belief_feature, belief_id, feature)

        state_features = set(_labels(state.features))
        if self.auto_features:
            state_features.update(_infer_state_features(state.utterance))
        if state.distress is not None:
            level = _distress_level(state.distress)
            fact(self.activation_level, state_id, level)
            if level == "high":
                state_features.update(("high_distress", "needs_validation"))
            elif state.distress >= 6:
                state_features.add("needs_validation")
        for feature in sorted(state_features):
            fact(self.state_feature, state_id, feature)

    def cognitive_distortiono(self, thought: Any, distortion: Any):
        return self.systems_by_source["cbt"].cognitive_distortiono(
            self,
            thought,
            distortion,
        )

    def irrational_beliefo(self, belief: Any, belief_type: Any):
        return self.systems_by_source["rebt"].irrational_beliefo(
            self,
            belief,
            belief_type,
        )

    def thought_irrational_beliefo(self, thought: Any, belief_type: Any):
        return self.systems_by_source["rebt"].thought_irrational_beliefo(
            self,
            thought,
            belief_type,
        )

    def state_distortiono(self, state: Any, distortion: Any):
        thought = var()
        return lall(
            self.has_thought(state, thought),
            self.cognitive_distortiono(thought, distortion),
        )

    def state_irrational_beliefo(self, state: Any, belief_type: Any):
        belief = var()
        thought = var()
        return conde(
            [
                self.has_belief(state, belief),
                self.irrational_beliefo(belief, belief_type),
            ],
            [
                self.has_thought(state, thought),
                self.thought_irrational_beliefo(thought, belief_type),
            ],
        )

    def act_stuck_processo(self, state: Any, process: Any):
        return self.systems_by_source["act"].pattern_goal(
            self,
            state,
            process,
        )

    def problem_patterno(self, state: Any, source: Any, pattern: Any):
        emotion = var()
        system_clauses = [
            [
                system.pattern_goal(self, state, pattern),
                eq(source, system.source),
            ]
            for system in self.systems
        ]
        return conde(
            [
                self.has_emotion(state, emotion),
                self.emotion_pattern(emotion, pattern),
                eq(source, "emotion"),
            ],
            *system_clauses,
            [
                self.state_feature(state, pattern),
                self.policy_pattern(pattern),
                eq(source, "policy"),
            ],
        )

    def interventiono(self, state: Any, intervention: Any):
        source = var()
        pattern = var()
        return lall(
            self.problem_patterno(state, source, pattern),
            self.pattern_intervention(pattern, intervention),
        )

    def recipeo(self, state: Any, recipe: Any):
        source = var()
        pattern = var()
        return lall(
            self.problem_patterno(state, source, pattern),
            self.pattern_recipe(pattern, recipe),
        )

    def formulationo(self, state: Any, formulation: Any):
        source = var()
        pattern = var()
        return lall(
            self.problem_patterno(state, source, pattern),
            self.formulation_pattern(formulation, pattern),
        )

    def contraindication_reasono(self, state: Any, intervention: Any, reason: Any):
        return conde(
            [
                self.state_feature(state, "high_distress"),
                self.high_intensity_intervention(intervention),
                eq(reason, "too_intense_for_high_distress"),
            ],
            [
                self.state_feature(state, "needs_validation"),
                self.premature_without_validation(intervention),
                eq(reason, "validate_before_challenging"),
            ],
            [
                self.state_feature(state, "trauma_content"),
                eq(intervention, "exposure_exercise"),
                eq(reason, "exposure_requires_consent_and_stabilization"),
            ],
            [
                self.state_feature(state, "safety_risk"),
                self.safety_deferred_intervention(intervention),
                eq(reason, "defer_until_safety_addressed"),
            ],
        )

    def contraindicatedo(self, state: Any, intervention: Any):
        reason = var()
        return self.contraindication_reasono(state, intervention, reason)

    def safe_interventiono(self, state: Any, intervention: Any):
        """Candidate intervention with bounded negation-as-failure filtering."""

        return lall(
            self.interventiono(state, intervention),
            self._not_contraindicatedo(state, intervention),
        )

    def safe_recipeo(self, state: Any, recipe: Any):
        """Candidate recipe with bounded safety/timing filtering."""

        return lall(
            self.recipeo(state, recipe),
            self._not_recipe_contraindicatedo(state, recipe),
        )

    def hypotheses_for(self, state_id: str) -> tuple[Hypothesis, ...]:
        source = var()
        pattern = var()
        rows = run(
            0, (source, pattern), self.problem_patterno(state_id, source, pattern)
        )
        return tuple(Hypothesis(src, pat) for src, pat in _unique_sorted(rows))

    def candidate_intervention_names(self, state_id: str) -> tuple[str, ...]:
        intervention = var()
        return _unique_sorted(
            run(0, intervention, self.interventiono(state_id, intervention))
        )

    def safe_intervention_names(self, state_id: str) -> tuple[str, ...]:
        intervention = var()
        return _unique_sorted(
            run(0, intervention, self.safe_interventiono(state_id, intervention))
        )

    def candidate_recipe_names(self, state_id: str) -> tuple[str, ...]:
        recipe = var()
        return _unique_sorted(run(0, recipe, self.recipeo(state_id, recipe)))

    def safe_recipe_names(self, state_id: str) -> tuple[str, ...]:
        recipe = var()
        return _unique_sorted(run(0, recipe, self.safe_recipeo(state_id, recipe)))

    def candidate_formulation_names(self, state_id: str) -> tuple[str, ...]:
        formulation = var()
        return _unique_sorted(
            run(0, formulation, self.formulationo(state_id, formulation))
        )

    def states_for_intervention(self, intervention: str) -> tuple[str, ...]:
        state = var()
        return _unique_sorted(run(0, state, self.interventiono(state, intervention)))

    def safe_states_for_intervention(self, intervention: str) -> tuple[str, ...]:
        state = var()
        return _unique_sorted(
            run(0, state, self.safe_interventiono(state, intervention))
        )

    def states_for_recipe(self, recipe: str) -> tuple[str, ...]:
        state = var()
        return _unique_sorted(run(0, state, self.recipeo(state, recipe)))

    def safe_states_for_recipe(self, recipe: str) -> tuple[str, ...]:
        state = var()
        return _unique_sorted(run(0, state, self.safe_recipeo(state, recipe)))

    def states_for_formulation(self, formulation: str) -> tuple[str, ...]:
        state = var()
        return _unique_sorted(run(0, state, self.formulationo(state, formulation)))

    def contraindicated_states_for_intervention(
        self, intervention: str
    ) -> tuple[str, ...]:
        state = var()
        return _unique_sorted(
            run(
                0,
                state,
                self.interventiono(state, intervention),
                self.contraindicatedo(state, intervention),
            )
        )

    def states_for_pattern(
        self, pattern: str, *, source: str | None = None
    ) -> tuple[str, ...]:
        state = var()
        found_source = var()
        goals = [self.problem_patterno(state, found_source, pattern)]
        if source is not None:
            goals.append(eq(found_source, source))
        return _unique_sorted(run(0, state, *goals))

    def patterns_for_intervention(self, intervention: str) -> tuple[str, ...]:
        pattern = var()
        return _unique_sorted(
            run(0, pattern, self.pattern_intervention(pattern, intervention))
        )

    def patterns_for_recipe(self, recipe: str) -> tuple[str, ...]:
        pattern = var()
        return _unique_sorted(run(0, pattern, self.pattern_recipe(pattern, recipe)))

    def patterns_for_formulation(self, formulation: str) -> tuple[str, ...]:
        pattern = var()
        return _unique_sorted(
            run(0, pattern, self.formulation_pattern(formulation, pattern))
        )

    def hypotheses_for_intervention(
        self, state_id: str, intervention: str
    ) -> tuple[Hypothesis, ...]:
        return self._hypotheses_for_intervention(state_id, intervention)

    def hypotheses_for_recipe(
        self, state_id: str, recipe: str
    ) -> tuple[Hypothesis, ...]:
        source = var()
        pattern = var()
        rows = run(
            0,
            (source, pattern),
            self.problem_patterno(state_id, source, pattern),
            self.pattern_recipe(pattern, recipe),
        )
        return tuple(Hypothesis(src, pat) for src, pat in _unique_sorted(rows))

    def hypotheses_for_formulation(
        self,
        state_id: str,
        formulation: str,
    ) -> tuple[Hypothesis, ...]:
        source = var()
        pattern = var()
        rows = run(
            0,
            (source, pattern),
            self.problem_patterno(state_id, source, pattern),
            self.formulation_pattern(formulation, pattern),
        )
        return tuple(Hypothesis(src, pat) for src, pat in _unique_sorted(rows))

    def intervention_report(self, state_id: str, intervention: str) -> dict[str, Any]:
        coherent = bool(run(1, state_id, self.interventiono(state_id, intervention)))
        contraindications = self.contraindication_reasons(state_id, intervention)
        hypotheses = self.hypotheses_for_intervention(state_id, intervention)
        return {
            "state_id": state_id,
            "intervention": intervention,
            "coherent": coherent,
            "safe": coherent and not contraindications,
            "contraindications": contraindications,
            "hypotheses": [asdict(item) for item in hypotheses],
            "modality": self._modalities_for(intervention),
            "exercise": _first(self.exercises_for(intervention)),
        }

    def intervention_requirement_report(
        self,
        state_id: str,
        intervention: str,
    ) -> dict[str, Any]:
        """Explain the backward paths that would make an intervention coherent."""

        hypotheses = self.hypotheses_for_intervention(state_id, intervention)
        satisfied_patterns = _unique_sorted(
            hypothesis.pattern for hypothesis in hypotheses
        )
        possible_patterns = self.patterns_for_intervention(intervention)
        alternative_patterns = tuple(
            pattern
            for pattern in possible_patterns
            if pattern not in set(satisfied_patterns)
        )
        contraindications = self.contraindication_reasons(state_id, intervention)
        return {
            "state_id": state_id,
            "intervention": intervention,
            "coherent": bool(hypotheses),
            "safe": bool(hypotheses) and not contraindications,
            "possible_patterns": possible_patterns,
            "satisfied_patterns": satisfied_patterns,
            "alternative_patterns": alternative_patterns,
            "satisfied_hypotheses": [asdict(item) for item in hypotheses],
            "contraindications": contraindications,
        }

    def compare_intervention_across_states(
        self,
        intervention: str,
        state_ids: Iterable[str] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        ids = tuple(state_ids) if state_ids is not None else tuple(sorted(self._states))
        return tuple(
            self.intervention_report(state_id, intervention) for state_id in ids
        )

    def exercises_for(self, intervention: str) -> tuple[str, ...]:
        exercise = var()
        return _unique_sorted(
            run(0, exercise, self.intervention_exercise(intervention, exercise))
        )

    def recipe_steps_for(self, recipe: str) -> tuple[str, ...]:
        index = var()
        intervention = var()
        rows = run(
            0, (index, intervention), self.recipe_step(recipe, index, intervention)
        )
        return tuple(intervention for _index, intervention in sorted(set(rows)))

    def recipe_rationale_for(self, recipe: str) -> str | None:
        rationale = var()
        return _first(
            _unique_sorted(run(0, rationale, self.recipe_rationale(recipe, rationale)))
        )

    def ranked_interventions(
        self,
        state_id: str,
        *,
        limit: int | None = None,
        include_unsafe: bool = False,
    ) -> tuple[InterventionCandidate, ...]:
        names = (
            self.candidate_intervention_names(state_id)
            if include_unsafe
            else self.safe_intervention_names(state_id)
        )
        candidates = []
        for intervention in names:
            contraindications = self.contraindication_reasons(state_id, intervention)
            if contraindications and not include_unsafe:
                continue
            hypotheses = self._hypotheses_for_intervention(state_id, intervention)
            modality = self._modalities_for(intervention)
            exercise = _first(self.exercises_for(intervention))
            score = self._score(
                state_id,
                intervention,
                modality,
                hypotheses,
                contraindications,
            )
            candidates.append(
                InterventionCandidate(
                    intervention=intervention,
                    score=score,
                    modality=modality,
                    hypotheses=hypotheses,
                    exercise=exercise,
                    contraindications=contraindications,
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.intervention))
        if limit is not None:
            candidates = candidates[:limit]
        return tuple(candidates)

    def ranked_recipes(
        self,
        state_id: str,
        *,
        limit: int | None = None,
        include_unsafe: bool = False,
    ) -> tuple[InterventionRecipe, ...]:
        names = (
            self.candidate_recipe_names(state_id)
            if include_unsafe
            else self.safe_recipe_names(state_id)
        )
        recipes = []
        for recipe in names:
            contraindications = self.recipe_contraindication_reasons(state_id, recipe)
            if contraindications and not include_unsafe:
                continue
            hypotheses = self.hypotheses_for_recipe(state_id, recipe)
            steps = self.recipe_steps_for(recipe)
            score = self._recipe_score(
                state_id, recipe, steps, hypotheses, contraindications
            )
            recipes.append(
                InterventionRecipe(
                    recipe=recipe,
                    score=score,
                    steps=steps,
                    hypotheses=hypotheses,
                    rationale=self.recipe_rationale_for(recipe),
                    contraindications=contraindications,
                )
            )
        recipes.sort(key=lambda item: (-item.score, item.recipe))
        if limit is not None:
            recipes = recipes[:limit]
        return tuple(recipes)

    def ranked_formulations(
        self,
        state_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[DifferentialFormulation, ...]:
        names = self.candidate_formulation_names(state_id)
        formulations = []
        for name in names:
            evidence = self.hypotheses_for_formulation(state_id, name)
            if not evidence:
                continue
            possible_patterns = self.patterns_for_formulation(name)
            matched_patterns = {hypothesis.pattern for hypothesis in evidence}
            missing_evidence = tuple(
                pattern
                for pattern in possible_patterns
                if pattern not in matched_patterns
            )
            score = self._formulation_score(
                state_id,
                name,
                evidence,
                possible_patterns,
            )
            formulations.append(
                DifferentialFormulation(
                    formulation=name,
                    score=score,
                    label=self._formulation_label(name),
                    summary=self._formulation_summary(name),
                    evidence=evidence,
                    missing_evidence=missing_evidence[:5],
                    discriminating_question=self._formulation_discriminator(name),
                    focus=self._formulation_focus(name),
                    interventions=self._interventions_for_formulation(name, evidence),
                    recipes=self._recipes_for_formulation(name, evidence),
                )
            )
        formulations.sort(key=lambda item: (-item.score, item.formulation))
        if limit is not None:
            formulations = formulations[:limit]
        return tuple(formulations)

    def clarifying_moves(
        self,
        state_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[ClarifyingMove, ...]:
        """Rank small questions/observations that reduce formulation uncertainty."""

        formulations = self.ranked_formulations(state_id, limit=5)
        if not formulations:
            return ()

        moves: list[ClarifyingMove] = []
        top = formulations[0]
        runner_up = formulations[1] if len(formulations) > 1 else None
        top_patterns = {item.pattern for item in top.evidence}

        if "safety_risk" in top_patterns:
            moves.append(
                ClarifyingMove(
                    move="safety_status_check",
                    priority=12.0,
                    kind="safety_check",
                    question="Are you safe right now, and is there someone you can contact immediately?",
                    rationale="Safety cues override differential coaching until immediate risk is clearer.",
                    target_formulations=(top.formulation,),
                    supported_by=top.evidence,
                    missing_evidence=top.missing_evidence,
                    expected_information=("immediate_risk", "available_support"),
                    intervention="safety_planning",
                )
            )
        elif top.formulation == "high_distress_first":
            moves.append(
                ClarifyingMove(
                    move="activation_workability_check",
                    priority=self._clarifying_priority(top, runner_up) + 2.0,
                    kind="stabilization_check",
                    question=top.discriminating_question
                    or "Is your system too activated for problem-solving to be useful yet?",
                    rationale="High activation can make otherwise coherent coaching moves mistimed.",
                    target_formulations=_target_formulations(top, runner_up),
                    supported_by=_combined_evidence(top, runner_up),
                    missing_evidence=_combined_missing_evidence(top, runner_up),
                    expected_information=(
                        "activation_level",
                        "readiness_for_problem_solving",
                    ),
                    intervention="present_moment_grounding",
                )
            )

        if runner_up and self._needs_differential_inquiry(top, runner_up):
            moves.append(
                ClarifyingMove(
                    move=f"distinguish_{top.formulation}_vs_{runner_up.formulation}",
                    priority=self._clarifying_priority(top, runner_up),
                    kind="differential_question",
                    question=self._pairwise_discriminating_question(top, runner_up),
                    rationale=(
                        "The top formulations are close enough that one targeted answer "
                        "could change the next coaching move."
                    ),
                    target_formulations=(top.formulation, runner_up.formulation),
                    supported_by=_combined_evidence(top, runner_up),
                    missing_evidence=_combined_missing_evidence(top, runner_up),
                    expected_information=_expected_information(top, runner_up),
                    intervention="gentle_check_in",
                )
            )

        if top.discriminating_question and top.missing_evidence:
            moves.append(
                ClarifyingMove(
                    move=f"probe_{top.formulation}",
                    priority=self._clarifying_priority(top, None) - 0.25,
                    kind="evidence_probe",
                    question=top.discriminating_question,
                    rationale=(
                        "The strongest formulation is plausible but still missing "
                        "some supporting or disconfirming evidence."
                    ),
                    target_formulations=(top.formulation,),
                    supported_by=top.evidence,
                    missing_evidence=top.missing_evidence,
                    expected_information=top.missing_evidence[:4],
                    intervention="gentle_check_in",
                )
            )

        moves = _unique_clarifying_moves(moves)
        moves.sort(key=lambda item: (-item.priority, item.move))
        if limit is not None:
            moves = moves[:limit]
        return tuple(moves)

    def contraindication_reasons(
        self, state_id: str, intervention: str
    ) -> tuple[str, ...]:
        reason = var()
        return _unique_sorted(
            run(
                0,
                reason,
                self.contraindication_reasono(state_id, intervention, reason),
            )
        )

    def recipe_contraindication_reasons(
        self,
        state_id: str,
        recipe: str,
    ) -> tuple[str, ...]:
        """Return blockers for a whole recipe, honoring recipe sequencing."""

        steps = self.recipe_steps_for(recipe)
        opens_with_validation = bool(steps and steps[0] == "validation")
        reasons = []
        for step in steps:
            for reason in self.contraindication_reasons(state_id, step):
                if reason == "validate_before_challenging" and opens_with_validation:
                    continue
                reasons.append(f"{step}:{reason}")
        return _unique_sorted(reasons)

    def reasoning_snapshot(self, state_id: str, *, limit: int = 5) -> dict[str, Any]:
        return {
            "state_id": state_id,
            "note": "Patterns are coaching hypotheses, not diagnoses.",
            "operating_mode": self.operating_mode_for(state_id),
            "hypotheses": [asdict(item) for item in self.hypotheses_for(state_id)],
            "formulations": [
                asdict(item) for item in self.ranked_formulations(state_id, limit=limit)
            ],
            "clarifying_moves": [
                asdict(item) for item in self.clarifying_moves(state_id, limit=limit)
            ],
            "candidates": [
                asdict(item)
                for item in self.ranked_interventions(state_id, limit=limit)
            ],
            "recipes": [
                asdict(item) for item in self.ranked_recipes(state_id, limit=limit)
            ],
        }

    def operating_mode_for(self, state_id: str) -> dict[str, Any]:
        """Summarize whether this turn is coaching-oriented or consultative."""

        hypotheses = self.hypotheses_for(state_id)
        consultative = tuple(
            hypothesis.pattern
            for hypothesis in hypotheses
            if hypothesis.source == "consultative"
        )
        if consultative:
            return {
                "mode": "consultative_facilitation",
                "stance": "friendly_open_encouraging",
                "patterns": consultative,
            }
        return {
            "mode": "coaching_support",
            "stance": "warm_tentative_non_diagnostic",
            "patterns": (),
        }

    def _not_contraindicatedo(self, state: Any, intervention: Any):
        def goal(substitution):
            state_value, intervention_value = reify((state, intervention), substitution)
            if isvar(state_value) or isvar(intervention_value):
                return
            reason = var()
            blocked = run(
                1,
                reason,
                self.contraindication_reasono(state_value, intervention_value, reason),
            )
            if blocked:
                return
            yield substitution

        return goal

    def _not_recipe_contraindicatedo(self, state: Any, recipe: Any):
        def goal(substitution):
            state_value, recipe_value = reify((state, recipe), substitution)
            if isvar(state_value) or isvar(recipe_value):
                return
            if self.recipe_contraindication_reasons(
                str(state_value), str(recipe_value)
            ):
                return
            yield substitution

        return goal

    def _hypotheses_for_intervention(
        self, state_id: str, intervention: str
    ) -> tuple[Hypothesis, ...]:
        source = var()
        pattern = var()
        rows = run(
            0,
            (source, pattern),
            self.problem_patterno(state_id, source, pattern),
            self.pattern_intervention(pattern, intervention),
        )
        return tuple(Hypothesis(src, pat) for src, pat in _unique_sorted(rows))

    def _modalities_for(self, intervention: str) -> tuple[str, ...]:
        modality = var()
        return _unique_sorted(
            run(0, modality, self.intervention_modality(intervention, modality))
        )

    def _has_recent_intervention(self, state_id: str, intervention: str) -> bool:
        marker = var()
        return bool(
            run(
                1,
                marker,
                self.recent_intervention(state_id, intervention),
            )
        )

    def _preferred_modalities(self, state_id: str) -> tuple[str, ...]:
        modality = var()
        return _unique_sorted(
            run(0, modality, self.preferred_modality(state_id, modality))
        )

    def _score(
        self,
        state_id: str,
        intervention: str,
        modality: tuple[str, ...],
        hypotheses: tuple[Hypothesis, ...],
        contraindications: tuple[str, ...],
    ) -> float:
        patterns = {hypothesis.pattern for hypothesis in hypotheses}
        score = 1.0 + 0.75 * len(patterns)

        if "needs_validation" in patterns and intervention == "validation":
            score += 2.0
        if "minimal_disclosure" in patterns:
            if intervention == "validation":
                score += 2.0
            if intervention == "gentle_check_in":
                score += 1.75
        if patterns.intersection({"sadness", "shame", "anger", "anxiety", "overwhelm"}):
            if intervention == "validation":
                score += 1.75
            if intervention in {
                "gentle_check_in",
                "self_compassion",
                "needs_exploration",
            }:
                score += 1.25
        if "high_distress" in patterns:
            if intervention == "validation":
                score += 3.0
            if intervention == "present_moment_grounding":
                score += 2.0
        if "safety_risk" in patterns and intervention == "safety_planning":
            score += 5.0
        for system in self.systems:
            score += system.score_bonus(intervention, patterns)

        preferred = set(self._preferred_modalities(state_id))
        if preferred.intersection(modality):
            score += 1.25
        if self._has_recent_intervention(state_id, intervention):
            score -= 1.5
        if contraindications:
            score -= 20.0
        return round(score, 2)

    def _formulation_score(
        self,
        state_id: str,
        formulation: str,
        evidence: tuple[Hypothesis, ...],
        possible_patterns: tuple[str, ...],
    ) -> float:
        patterns = {hypothesis.pattern for hypothesis in evidence}
        possible = set(possible_patterns)
        coverage = len(patterns) / max(len(possible), 1)
        score = 1.0 + 0.85 * len(patterns) + 1.5 * coverage

        direct_patterns = {
            "safety_first": {"safety_risk"},
            "high_distress_first": {"high_distress_gating", "high_distress"},
            "minimal_disclosure_affect": {
                "minimal_disclosure",
                "minimal_disclosure_sad_anxious",
            },
            "avoidance_identity_threat": {"avoidance_identity_threat"},
            "shame_self_worth_fusion": {"shame_self_worth_fusion"},
            "valued_action_procrastination": {"valued_action_procrastination"},
            "certainty_decision_loop": {"certainty_avoidance_loop"},
            "approval_interpersonal_threat": {"approval_threat_loop"},
            "control_struggle": {"control_struggle_loop"},
            "vulnerability_distress": {"vulnerability_distress_loop"},
            "felt_sense_unclear_meaning": {
                "felt_sense_contact",
                "unclear_felt_meaning",
                "symbolization_needed",
            },
        }
        if patterns.intersection(direct_patterns.get(formulation, set())):
            score += 2.0

        if "safety_risk" in patterns and formulation == "safety_first":
            score += 5.0
        if patterns.intersection({"high_distress", "crisis_survival", "overwhelm"}):
            if formulation == "high_distress_first":
                score += 2.25
            elif formulation not in {"safety_first", "high_distress_first"}:
                score -= 0.75
        if (
            "minimal_disclosure" in patterns
            and formulation == "minimal_disclosure_affect"
        ):
            score += 1.25
        if (
            "avoidance_identity_threat" in patterns
            and formulation == "avoidance_identity_threat"
        ):
            score += 1.5
        if (
            "valued_action_procrastination" in patterns
            and formulation == "valued_action_procrastination"
        ):
            score += 1.25
        if (
            "shame_self_worth_fusion" in patterns
            and formulation == "shame_self_worth_fusion"
        ):
            score += 1.25
        if (
            patterns.intersection(
                {
                    "felt_sense_contact",
                    "unclear_felt_meaning",
                    "symbolization_needed",
                }
            )
            and formulation == "felt_sense_unclear_meaning"
        ):
            score += 1.25
        if (
            self._has_recent_intervention(state_id, "safety_planning")
            and formulation != "safety_first"
        ):
            score += 0.25
        return round(score, 2)

    def _needs_differential_inquiry(
        self,
        top: DifferentialFormulation,
        runner_up: DifferentialFormulation,
    ) -> bool:
        if top.formulation == runner_up.formulation:
            return False
        score_gap = top.score - runner_up.score
        if score_gap > 4.0:
            return False
        if score_gap <= 2.0:
            return True
        if top.missing_evidence and runner_up.evidence:
            return True
        return False

    def _clarifying_priority(
        self,
        top: DifferentialFormulation,
        runner_up: DifferentialFormulation | None,
    ) -> float:
        score = 4.0 + min(top.score / 4.0, 3.0)
        if top.missing_evidence:
            score += min(len(top.missing_evidence) * 0.25, 1.0)
        if runner_up is not None:
            gap = max(top.score - runner_up.score, 0.0)
            score += max(0.0, 2.0 - min(gap, 2.0))
        if top.formulation in {"minimal_disclosure_affect", "high_distress_first"}:
            score += 1.0
        return round(score, 2)

    def _pairwise_discriminating_question(
        self,
        top: DifferentialFormulation,
        runner_up: DifferentialFormulation,
    ) -> str:
        pair = frozenset({top.formulation, runner_up.formulation})
        pair_questions = {
            frozenset({"avoidance_identity_threat", "approval_interpersonal_threat"}): (
                "Is the hardest part the task itself, what the outcome might seem to prove about you, "
                "or how someone else might judge or respond?"
            ),
            frozenset({"avoidance_identity_threat", "shame_self_worth_fusion"}): (
                "Is the pain mostly about the task you're avoiding, or about the larger self-worth "
                "verdict it seems to trigger?"
            ),
            frozenset({"avoidance_identity_threat", "valued_action_procrastination"}): (
                "Is the main block not knowing the next concrete step, or fearing what the result "
                "would mean about you?"
            ),
            frozenset({"high_distress_first", "avoidance_identity_threat"}): (
                "Would settling your system first make this more workable, or is the main block "
                "still what the task might seem to prove about you?"
            ),
            frozenset({"minimal_disclosure_affect", "high_distress_first"}): (
                "Is this feeling gentle enough to explore, or does it need grounding before we look closer?"
            ),
            frozenset({"control_struggle", "valued_action_procrastination"}): (
                "Are you waiting for the feeling to change before acting, or is the next action itself unclear?"
            ),
            frozenset({"minimal_disclosure_affect", "felt_sense_unclear_meaning"}): (
                "Is this mostly a feeling to name in broad strokes, or is there a vague body sense "
                "that wants slower attention before words?"
            ),
            frozenset({"control_struggle", "felt_sense_unclear_meaning"}): (
                "Are you mainly trying to make the feeling go away, or trying to sense what it is carrying?"
            ),
            frozenset({"certainty_decision_loop", "approval_interpersonal_threat"}): (
                "Are you mainly waiting for more certainty, or trying to avoid another person's judgment?"
            ),
        }
        if pair in pair_questions:
            return pair_questions[pair]
        return (
            f"Which feels closer right now: {top.label.casefold()}, "
            f"{runner_up.label.casefold()}, or something else?"
        )

    def _interventions_for_formulation(
        self,
        formulation: str,
        evidence: tuple[Hypothesis, ...],
    ) -> tuple[str, ...]:
        interventions = []
        intervention = var()
        for hypothesis in evidence:
            rows = run(
                0,
                intervention,
                self.pattern_intervention(hypothesis.pattern, intervention),
            )
            interventions.extend(rows)
        if not interventions:
            for pattern in self.patterns_for_formulation(formulation):
                interventions.extend(
                    run(
                        0,
                        intervention,
                        self.pattern_intervention(pattern, intervention),
                    )
                )
        return _unique_sorted(interventions)[:5]

    def _recipes_for_formulation(
        self,
        formulation: str,
        evidence: tuple[Hypothesis, ...],
    ) -> tuple[str, ...]:
        recipes = []
        recipe = var()
        for hypothesis in evidence:
            rows = run(0, recipe, self.pattern_recipe(hypothesis.pattern, recipe))
            recipes.extend(rows)
        if not recipes:
            for pattern in self.patterns_for_formulation(formulation):
                recipes.extend(run(0, recipe, self.pattern_recipe(pattern, recipe)))
        return _unique_sorted(recipes)[:5]

    def _formulation_label(self, formulation: str) -> str:
        label = var()
        return (
            _first(
                _unique_sorted(
                    run(0, label, self.formulation_label(formulation, label))
                )
            )
            or formulation
        )

    def _formulation_summary(self, formulation: str) -> str:
        summary = var()
        return (
            _first(
                _unique_sorted(
                    run(0, summary, self.formulation_summary(formulation, summary))
                )
            )
            or ""
        )

    def _formulation_discriminator(self, formulation: str) -> str | None:
        question = var()
        return _first(
            _unique_sorted(
                run(0, question, self.formulation_discriminator(formulation, question))
            )
        )

    def _formulation_focus(self, formulation: str) -> str | None:
        focus = var()
        return _first(
            _unique_sorted(run(0, focus, self.formulation_focus(formulation, focus)))
        )

    def _recipe_score(
        self,
        state_id: str,
        recipe: str,
        steps: tuple[str, ...],
        hypotheses: tuple[Hypothesis, ...],
        contraindications: tuple[str, ...],
    ) -> float:
        patterns = {hypothesis.pattern for hypothesis in hypotheses}
        score = 1.25 + 0.9 * len(patterns) + 0.15 * len(steps)

        if "safety_risk" in patterns and recipe == "safety_first":
            score += 5.0
        if patterns.intersection({"high_distress", "crisis_survival", "overwhelm"}):
            if recipe == "stabilize_then_choose":
                score += 3.0
            if recipe == "validate_then_ground":
                score += 1.25
        if "minimal_disclosure" in patterns and recipe == "validate_then_check_in":
            score += 2.5
        if "avoidance_identity_threat" in patterns and recipe == "validate_defuse_act":
            score += 2.25
        if "shame_self_worth_fusion" in patterns and recipe == "validate_self_worth":
            score += 2.0
        if (
            "valued_action_procrastination" in patterns
            and recipe == "values_to_micro_action"
        ):
            score += 2.0
        if "certainty_avoidance_loop" in patterns and recipe == "wise_mind_decide_act":
            score += 1.75
        if (
            "control_struggle_loop" in patterns
            and recipe == "acceptance_willing_action"
        ):
            score += 1.75
        if (
            "vulnerability_distress_loop" in patterns
            and recipe == "reduce_vulnerability_regulate"
        ):
            score += 1.75
        if "approval_threat_loop" in patterns and recipe == "validate_check_facts":
            score += 1.5
        if (
            patterns.intersection({"felt_sense_contact", "unclear_felt_meaning"})
            and recipe == "pause_describe_resonate"
        ):
            score += 1.75
        if "symbolization_needed" in patterns and recipe == "describe_resonate_track":
            score += 1.5
        if (
            "inner_critic_presence" in patterns
            and recipe == "validate_distance_resonate"
        ):
            score += 1.5
        if "felt_shift_possible" in patterns and recipe == "pause_track_shift":
            score += 1.25

        preferred = set(self._preferred_modalities(state_id))
        if preferred:
            recipe_modalities = {
                modality for step in steps for modality in self._modalities_for(step)
            }
            if preferred.intersection(recipe_modalities):
                score += 1.0
        for step in steps:
            if self._has_recent_intervention(state_id, step):
                score -= 0.75
        if contraindications:
            score -= 20.0
        return round(score, 2)


def _target_formulations(
    top: DifferentialFormulation,
    runner_up: DifferentialFormulation | None,
) -> tuple[str, ...]:
    if runner_up is None:
        return (top.formulation,)
    return (top.formulation, runner_up.formulation)


def _combined_evidence(
    top: DifferentialFormulation,
    runner_up: DifferentialFormulation | None,
) -> tuple[Hypothesis, ...]:
    evidence = list(top.evidence)
    if runner_up is not None:
        evidence.extend(runner_up.evidence)
    return tuple(dict.fromkeys(evidence))[:8]


def _combined_missing_evidence(
    top: DifferentialFormulation,
    runner_up: DifferentialFormulation | None,
) -> tuple[str, ...]:
    missing = list(top.missing_evidence)
    if runner_up is not None:
        missing.extend(runner_up.missing_evidence)
    return _unique_preserving(missing)[:8]


def _expected_information(
    top: DifferentialFormulation,
    runner_up: DifferentialFormulation,
) -> tuple[str, ...]:
    expected = [
        top.focus or top.formulation,
        runner_up.focus or runner_up.formulation,
        *top.missing_evidence[:2],
        *runner_up.missing_evidence[:2],
    ]
    return _unique_preserving(expected)[:6]


def _unique_clarifying_moves(moves: list[ClarifyingMove]) -> list[ClarifyingMove]:
    by_question: dict[str, ClarifyingMove] = {}
    for move in moves:
        key = move.question.casefold()
        existing = by_question.get(key)
        if existing is None or move.priority > existing.priority:
            by_question[key] = move
    return list(by_question.values())


def demo_kernel() -> TherapeuticReasoningKernel:
    kernel = TherapeuticReasoningKernel()
    kernel.add_state(
        CoachingState(
            state_id="prototype",
            utterance=(
                "I keep avoiding working on the prototype because if it is bad, "
                "it proves I am not cut out for this."
            ),
            situations=("working on prototype",),
            thoughts=("If it is bad, it proves I am not cut out for this.",),
            emotions=("anxiety", "shame"),
            behaviors=("avoidance",),
            values=("mastery", "autonomy"),
            goals=("ship prototype",),
            distress=5,
            preferred_modalities=("act",),
        )
    )
    return kernel


def main() -> None:
    kernel = demo_kernel()
    snapshot = kernel.reasoning_snapshot("prototype", limit=4)
    for hypothesis in snapshot["hypotheses"]:
        print(f"{hypothesis['source']}: {hypothesis['pattern']}")
    print()
    for candidate in snapshot["candidates"]:
        print(f"{candidate['score']:>4}  {candidate['intervention']}")
        print(f"      {candidate['exercise']}")


def _label(value: str) -> str:
    label = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().casefold()).strip("_")
    return _LABEL_ALIASES.get(label, label)


_LABEL_ALIASES = {
    "sad": "sadness",
    "anxious": "anxiety",
    "worried": "anxiety",
    "panic": "anxiety",
    "panicked": "anxiety",
    "ashamed": "shame",
    "embarrassed": "shame",
    "angry": "anger",
    "mad": "anger",
    "overwhelmed": "overwhelm",
}


def _labels(values: Iterable[str] | str | None) -> tuple[str, ...]:
    return _unique_preserving(
        _label(value) for value in _iter_values(values) if str(value).strip()
    )


def _strings(values: Iterable[str] | str | None) -> tuple[str, ...]:
    return tuple(
        str(value).strip() for value in _iter_values(values) if str(value).strip()
    )


def _iter_values(values: Iterable[str] | str | None):
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def _unique_preserving(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _unique_sorted(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(set(values), key=repr))


def _first(values: tuple[str, ...]) -> str | None:
    return values[0] if values else None


def _distress_level(distress: int) -> str:
    if distress >= 7:
        return "high"
    if distress >= 4:
        return "medium"
    return "low"


def _infer_behaviors(text: str) -> set[str]:
    lowered = text.casefold()
    behaviors: set[str] = set()
    if re.search(r"\b(avoid|avoiding|procrastinat\w*|putting off)\b", lowered):
        behaviors.add("avoidance")
    if re.search(
        r"\b(procrastinat\w*|putting off|keep putting off|put off)\b", lowered
    ):
        behaviors.add("procrastination")
    if re.search(r"\b(withdraw|shut down|isolat)\b", lowered):
        behaviors.add("withdrawal")
    if re.search(r"\b(ruminat|spiral|overthinking|can't stop thinking)\b", lowered):
        behaviors.add("rumination")
    return behaviors


def _infer_state_features(text: str) -> set[str]:
    lowered = text.casefold()
    features: set[str] = set()
    consultative_features: set[str] = set()
    if re.search(
        r"\b(you are useless|you're useless|youre useless|you suck|shut up|"
        r"fuck you|stupid bot|idiot|garbage assistant|terrible assistant|"
        r"worthless assistant|bad assistant|useless assistant)\b",
        lowered,
    ):
        features.update(("empath_directed_aggression", "needs_validation"))
    if re.search(
        r"\b(what is|what's|who is|when is|where is|define|meaning of|"
        r"how many|how much|quick answer|short answer|briefly answer)\b",
        lowered,
    ):
        consultative_features.add("factual_question")
    if re.search(
        r"\b(explain|how does|how do|teach me|walk me through|"
        r"main ideas|overview of|what are the main ideas|how it works)\b",
        lowered,
    ):
        consultative_features.add("instructional_request")
    if re.search(
        r"\b(what should i do|what would you recommend|recommend|best way|"
        r"how would you approach|what are my options|options for|advice on)\b",
        lowered,
    ):
        consultative_features.add("advisory_request")
    if re.search(
        r"\b(compare|comparison|evaluate|tradeoffs?|pros and cons|analy[sz]e|"
        r"research|evidence|decision support|synthesis|criteria)\b",
        lowered,
    ):
        consultative_features.add("analytical_request")
    if re.search(
        r"\b(brainstorm|ideas?|name|naming|tagline|draft|write|design|"
        r"worldbuild|creative options|come up with)\b",
        lowered,
    ):
        consultative_features.add("creative_ideation_request")
    if re.search(
        r"\b(help me think through|think through|explore this|hidden assumptions|"
        r"question my|what am i missing|socratic|challenge my assumption)\b",
        lowered,
    ):
        consultative_features.add("socratic_exploration_request")
    if consultative_features:
        features.update(consultative_features)
        features.add("consultative_request")
    if re.search(r"\b(trauma|flashback|assault)\b", lowered):
        features.add("trauma_content")
    if re.search(r"\b(suicid|self-harm|hurt myself|end it all)\b", lowered):
        features.add("safety_risk")
    if re.search(r"\b(numb|checked out|not present|dissociat)\b", lowered):
        features.add("present_moment_disconnection")
    if re.search(
        r"\b(stress|stressed|stressful|under pressure|pressure|overloaded|"
        r"workload|burned out|burnt out|too much to handle|deadline pressure)\b",
        lowered,
    ):
        features.add("stress_load")
    if re.search(
        r"\b(tense|tension|tight shoulders?|tight jaw|clenched|racing heart|"
        r"stress in my body|stress in the body|headache|shallow breathing)\b",
        lowered,
    ):
        features.add("body_tension")
    if re.search(
        r"\b(autopilot|on automatic|reactive|reactivity|about to snap|"
        r"snap at|snapping at|knee-jerk|lash out)\b",
        lowered,
    ):
        features.add("autopilot_reactivity")
    if re.search(
        r"\b(panic|panicking|can't breathe|cannot breathe|can't cope|cannot cope)\b",
        lowered,
    ):
        features.update(("high_distress", "needs_validation"))
    if re.search(
        r"\b(out of control|emotionally flooded|flooded|losing it|spiraling|"
        r"can't calm down|cannot calm down|overwhelmed by (my )?emotion)\b",
        lowered,
    ):
        features.add("emotion_dysregulation")
    if re.search(
        r"\b(get rid of (this|the) (feeling|emotion|thought|anxiety|sadness|shame)|"
        r"make (this|the) (feeling|emotion|thought|anxiety|sadness|shame) stop|"
        r"stop feeling|can't feel|cannot feel|can't have this thought|"
        r"cannot have this thought|need this "
        r"(feeling|emotion|thought|anxiety|sadness|shame) to go away)\b",
        lowered,
    ):
        features.add("control_struggle")
    if re.search(
        r"\b(sleep deprived|haven't slept|have not slept|didn't sleep|did not sleep|"
        r"haven't eaten|have not eaten|skipped meals?|burned out|burnt out|"
        r"exhausted|depleted|running on empty)\b",
        lowered,
    ):
        features.add("vulnerability_factors")
    if re.search(
        r"\b(urge to (yell|scream|text|call|drink|use|quit|run away|lash out)|"
        r"about to (yell|scream|text|call|drink|use|quit|lash out))\b",
        lowered,
    ):
        features.add("crisis_urge")
    if re.search(
        r"\b(argument|fight with|conflict with|relationship conflict|"
        r"they'?re mad at me|they are mad at me|hard conversation)\b",
        lowered,
    ):
        features.add("interpersonal_conflict")
    if re.search(
        r"\b(can't say no|cannot say no|afraid to ask|scared to ask|"
        r"set a boundary|setting a boundary|hold a boundary|ask for what i need)\b",
        lowered,
    ):
        features.add("boundary_difficulty")
    if re.search(
        r"\b(approval|approve of me|like me|respect me|disapprove|disapproval|"
        r"rejection|reject me|disappoint them|judged|judge me)\b",
        lowered,
    ):
        features.add("approval_threat")
    if re.search(
        r"\b(shouldn't feel|should not feel|stupid for feeling|wrong to feel|"
        r"bad for feeling|i hate that i feel)\b",
        lowered,
    ):
        features.add("self_invalidation")
    if re.search(
        r"\b(goal|milestone|next step|habit|routine|plan|schedule|deadline|"
        r"outcome|objective|target|okr|key result|success measure|project)\b",
        lowered,
    ):
        features.add("goal_setting")
    if re.search(
        r"\b(open loops?|too many things|too much on my plate|everything is in my head|"
        r"scattered|overloaded by tasks|too many tasks|can't track|cannot track|"
        r"mental clutter|inbox)\b",
        lowered,
    ):
        features.update(("open_loops", "overloaded_open_loops"))
    if re.search(
        r"\b(too many active|too much in progress|wip|work in progress|"
        r"doing too many|juggling too many|spread too thin)\b",
        lowered,
    ):
        features.add("wip_overload")
    if re.search(
        r"\b(obstacle|blocker|gets in the way|if .* then|implementation intention|"
        r"woop|when .* happens)\b",
        lowered,
    ):
        features.add("obstacle_planning")
    if re.search(
        r"\b(too big|scope is too large|boil the ocean|over-scoped|overscoped|"
        r"huge project|massive project|can't start because it's too big)\b",
        lowered,
    ):
        features.add("scope_too_large")
    if re.search(
        r"\b(weekly review|review due|need to review|retrospective|retro|"
        r"look back at the week)\b",
        lowered,
    ):
        features.add("review_due")
    if re.search(
        r"\b(don't know how to measure|do not know how to measure|"
        r"unclear success|what counts as progress|no metric|no measure)\b",
        lowered,
    ):
        features.add("success_measure_unclear")
    if re.search(
        r"\b(no motivation|not motivated|don't feel like|do not feel like|"
        r"too hard|can't get myself|cannot get myself|keep going|persist)\b",
        lowered,
    ):
        features.add("motivation_block")
    if re.search(
        r"\b(not capable|can't do this|cannot do this|not cut out|no confidence|"
        r"i'm incompetent|im incompetent|i am incompetent)\b",
        lowered,
    ):
        features.add("self_efficacy_doubt")
    if re.search(
        r"\b(can't decide|cannot decide|decision|which option|what should i choose|"
        r"pros and cons|uncertain what to do|don't know what to do|do not know what to do)\b",
        lowered,
    ):
        features.add("decision_uncertainty")
    if re.search(
        r"\b(need to be certain|must be certain|guarantee|guaranteed|"
        r"can't decide until|cannot decide until|perfect information)\b",
        lowered,
    ):
        features.add("certainty_demand")
    if re.search(
        r"\b(distracted|can't focus|cannot focus|focus|environment|workspace|"
        r"phone|notifications|cue|friction|implementation intention)\b",
        lowered,
    ):
        features.add("attention_environment")
    if re.search(
        r"\b(setback|relapse|fell off|off track|failed again|inconsistent|"
        r"rejection|recover|restart|start over|backslid)\b",
        lowered,
    ):
        features.add("setback_recovery")
    if re.search(
        r"\b(what am i learning|what have i learned|review|pattern keeps|"
        r"keeps happening|again and again|what's working|what is working|adjust)\b",
        lowered,
    ):
        features.add("integration_review")
    if re.search(
        r"\b(felt sense|in my body|body knows|gut feeling|tightness|tight|"
        r"knot|heavy|heaviness|weight|pressure|hollow|stomach|chest|"
        r"throat|belly|body feeling)\b",
        lowered,
    ):
        features.add("felt_sense")
    if re.search(
        r"\b(something feels off|can't put (it|this) into words|"
        r"cannot put (it|this) into words|can't put words to (it|this)|"
        r"cannot put words to (it|this)|hard to name|can't name it|"
        r"cannot name it|vague|fuzzy|murky|wordless|stuck feeling)\b",
        lowered,
    ):
        features.update(("unclear_felt_sense", "hard_to_name"))
    if re.search(
        r"\b(right word|right phrase|word for (it|this)|name (it|this)|"
        r"what is this feeling|image for (it|this)|handle for (it|this)|"
        r"can't put words to (it|this)|cannot put words to (it|this))\b",
        lowered,
    ):
        features.add("symbolization_needed")
    if re.search(
        r"\b(inner critic|critical voice|voice says|part of me says|"
        r"judging myself|judge myself|harsh voice|self-critical)\b",
        lowered,
    ):
        features.add("inner_critic")
    if re.search(
        r"\b(softens?|eases?|loosens?|opens up|felt shift|something shifted|"
        r"body says yes|body says no|not quite)\b",
        lowered,
    ):
        features.add("felt_shift")
    if re.search(
        r"\b(always happens|keeps happening|again and again|same pattern)\b", lowered
    ):
        features.add("recurring_pattern")
    if re.search(
        r"\b(don't know what matters|do not know what matters|don't know what i want|"
        r"do not know what i want|aimless|nothing feels meaningful|what matters)\b",
        lowered,
    ):
        features.add("values_unclear")
    if _looks_like_minimal_disclosure(lowered):
        features.add("minimal_disclosure")
    return features


def _looks_like_minimal_disclosure(lowered_text: str) -> bool:
    words = re.findall(r"[a-z']+", lowered_text)
    if len(words) > 8:
        return False
    return bool(
        re.search(
            r"\b(sad|down|anxious|worried|scared|ashamed|angry|mad|overwhelmed|"
            r"rough day|bad day|hard day|not great)\b",
            lowered_text,
        )
    )


def _infer_text_features(text: str) -> set[str]:
    lowered = text.casefold()
    features: set[str] = set()

    if re.search(
        r"\b(disaster|catastrophe|ruined|everything is over|worst)\b", lowered
    ):
        features.add("future_disaster")
    if re.search(r"\b(awful|terrible|horrible|unbearable)\b", lowered):
        features.add("awful_outcome")
    if re.search(
        r"\b(can't stand|cannot stand|can't bear|cannot bear|unbearable)\b", lowered
    ):
        features.add("unbearable_claim")
    if re.search(r"\b(must|should|have to|need to)\b", lowered):
        features.add("demanding_rule")
    if re.search(
        r"\b(they will think|they'll think|they’ll think|people will think|"
        r"everyone will think|he thinks|she thinks|he'll think|he’ll think|"
        r"she'll think|she’ll think|investors? will think)\b",
        lowered,
    ):
        features.add("mind_reading_claim")
    if re.search(r"\b(always|never|completely|perfect|total failure)\b", lowered):
        features.add("binary_evaluation")
    if re.search(
        r"\b(proves?|means) (that )?(i am|i'm|im|i cannot|i can't)\b", lowered
    ):
        features.add("single_event_global_conclusion")
    if (
        re.search(
            r"\b(i am|i'm|im) (a )?(failure|worthless|broken|not good enough)\b",
            lowered,
        )
        or "not cut out" in lowered
    ):
        features.update(
            {
                "global_label",
                "identity_global_rating",
                "identity_fusion",
            }
        )
    if re.search(
        r"\b(i can't stop thinking|this thought is true|it proves who i am)\b", lowered
    ):
        features.add("sticky_thought")
    if re.search(
        r"\b(doesn't count|does not count|just luck|only luck|anyone could|"
        r"only because|bare minimum|not a real win|no big deal)\b",
        lowered,
    ):
        features.add("discounting_positive_claim")
    if re.search(
        r"\b(i feel like .*(so|therefore|which means|that means)|"
        r"because i feel .* (it means|that means|it proves)|"
        r"feel(s)? true|must be true because i feel)\b",
        lowered,
    ):
        features.add("feeling_as_fact")
    if re.search(
        r"\b(all my fault|entirely my fault|i caused everything|i ruined (it|this|everything)|"
        r"i'm to blame for everything|i am to blame for everything)\b",
        lowered,
    ):
        features.add("personal_responsibility_claim")
    if re.search(
        r"\b(only see|all i can see|all i can think about|nothing good|"
        r"ignore the good|can't see anything good|cannot see anything good)\b",
        lowered,
    ):
        features.add("negative_filter")
    if re.search(
        r"\b(need (them|everyone|people|him|her|investors?|my boss) to "
        r"(approve|like|respect|validate)|must be liked|must be approved|"
        r"can't handle (disapproval|rejection)|cannot handle (disapproval|rejection))\b",
        lowered,
    ):
        features.add("approval_demand")
    if re.search(
        r"\b(need to know for sure|must know for sure|need to be certain|"
        r"must be certain|can't act unless|cannot act unless|"
        r"can't decide until|cannot decide until|perfect information)\b",
        lowered,
    ):
        features.add("certainty_demand_claim")
    if re.search(
        r"\b(can't fail|cannot fail|failure isn't an option|failure is not an option|"
        r"can't handle failing|cannot handle failing|can't handle failure|"
        r"cannot handle failure)\b",
        lowered,
    ):
        features.add("failure_intolerance_claim")
    if re.search(
        r"\b(get rid of (this|the) (feeling|emotion|thought|anxiety|sadness|shame)|"
        r"make (this|the) (feeling|emotion|thought|anxiety|sadness|shame) stop|"
        r"can't have this thought|cannot have this thought|"
        r"need this (feeling|emotion|thought|anxiety|sadness|shame) to go away)\b",
        lowered,
    ):
        features.add("control_struggle")
    if re.search(
        r"\b(right word|right phrase|word for (it|this)|name (it|this)|"
        r"what is this feeling|image for (it|this)|handle for (it|this)|"
        r"can't put (it|this) into words|cannot put (it|this) into words|"
        r"can't put words to (it|this)|cannot put words to (it|this))\b",
        lowered,
    ):
        features.add("symbolization_need")
    if re.search(
        r"\b(inner critic|critical voice|voice says|part of me says|"
        r"judging myself|judge myself|harsh voice|self-critical)\b",
        lowered,
    ):
        features.add("inner_critic_claim")
    return features


if __name__ == "__main__":
    main()
