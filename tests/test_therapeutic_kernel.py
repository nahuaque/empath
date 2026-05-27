import unittest

from kanren import eq, facts, lall, run, var

from empath.therapeutic_systems import (
    CoachingFocusSystem,
    ConsultativeSystem,
    DBTSystem,
    FocusingSystem,
    MBSRSystem,
    TherapeuticSystem,
)
from empath.therapeutic_kernel import CoachingState, TherapeuticReasoningKernel


class _DBTSystem(TherapeuticSystem):
    source = "dbt"

    def load_ontology(self, kernel):
        facts(
            kernel.pattern_intervention,
            ("emotion_dysregulation", "paced_breathing"),
        )
        facts(kernel.intervention_modality, ("paced_breathing", "dbt"))
        facts(
            kernel.intervention_exercise,
            ("paced_breathing", "Practice paced breathing for one minute."),
        )

    def pattern_goal(self, kernel, state, pattern):
        return lall(
            kernel.state_feature(state, "emotion_dysregulation"),
            eq(pattern, "emotion_dysregulation"),
        )

    def score_bonus(self, intervention, patterns):
        if intervention == "paced_breathing" and "emotion_dysregulation" in patterns:
            return 2.0
        return 0.0


class TherapeuticReasoningKernelTests(unittest.TestCase):
    def test_state_focus_context_is_queryable_relationally(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="investor-presentation",
                concerns=("investor presentation",),
                tasks=("prepare investor presentation",),
                challenges=("avoidance or procrastination",),
                objectives=("deliver investor presentation",),
                stakes=("investor judgment",),
                domains=("work", "identity"),
            )
        )

        item = var()

        self.assertEqual(
            ("investor presentation",),
            run(0, item, kernel.has_concern("investor-presentation", item)),
        )
        self.assertEqual(
            ("prepare investor presentation",),
            run(0, item, kernel.has_task("investor-presentation", item)),
        )
        self.assertEqual(
            ("avoidance or procrastination",),
            run(0, item, kernel.has_challenge("investor-presentation", item)),
        )
        self.assertEqual(
            ("deliver investor presentation",),
            run(0, item, kernel.has_objective("investor-presentation", item)),
        )
        self.assertEqual(
            ("investor judgment",),
            run(0, item, kernel.has_stake("investor-presentation", item)),
        )
        self.assertEqual(
            {"work", "identity"},
            set(run(0, item, kernel.has_domain("investor-presentation", item))),
        )

    def test_goal_direction_facts_and_candidates_are_queryable(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="goal-direction",
                values=("mastery",),
                objectives=("ship prototype",),
                projects=("prototype",),
                next_actions=("open the prototype work",),
                obstacles=("fear of what the result might mean",),
                time_horizons=("this week",),
                success_measures=("prototype progress is visible",),
                behaviors=("procrastination",),
                features=("goal_setting",),
            )
        )

        item = var()
        hypotheses = {
            (hypothesis.source, hypothesis.pattern)
            for hypothesis in kernel.hypotheses_for("goal-direction")
        }
        interventions = {
            candidate.intervention
            for candidate in kernel.ranked_interventions("goal-direction")
        }
        recipes = {recipe.recipe for recipe in kernel.ranked_recipes("goal-direction")}

        self.assertEqual(
            ("prototype",),
            run(0, item, kernel.has_project("goal-direction", item)),
        )
        self.assertEqual(
            ("open the prototype work",),
            run(0, item, kernel.has_next_action("goal-direction", item)),
        )
        self.assertIn(("goal_direction", "objective_without_next_action"), hypotheses)
        self.assertIn(("goal_direction", "goal_obstacle_gap"), hypotheses)
        self.assertIn("define_next_action", interventions)
        self.assertIn("woop_obstacle_plan", interventions)
        self.assertIn("woop_then_next_action", recipes)

    def test_prototype_case_infers_act_cbt_rebt_candidates(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="prototype",
                thoughts=("If it is bad, it proves I am not cut out for this.",),
                emotions=("anxiety", "shame"),
                behaviors=("avoidance",),
                values=("mastery", "autonomy"),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("prototype")}
        self.assertIn(("act", "experiential_avoidance"), hypotheses)
        self.assertIn(("act", "fusion"), hypotheses)
        self.assertIn(("cbt", "global_labeling"), hypotheses)
        self.assertIn(("rebt", "self_downing"), hypotheses)

        interventions = {
            item.intervention for item in kernel.ranked_interventions("prototype")
        }
        self.assertIn("cognitive_defusion", interventions)
        self.assertIn("acceptance_committed_action", interventions)
        self.assertIn("unconditional_self_acceptance", interventions)

    def test_backward_query_finds_states_that_justify_intervention(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="prototype",
                thoughts=("If it is bad, it proves I am not cut out for this.",),
                behaviors=("avoidance",),
                values=("mastery",),
            )
        )

        self.assertIn("prototype", kernel.states_for_intervention("cognitive_defusion"))

    def test_backward_requirement_report_explains_matched_and_alternative_paths(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="prototype",
                thoughts=("If it is bad, it proves I am not cut out for this.",),
                behaviors=("avoidance",),
                values=("mastery",),
            )
        )

        report = kernel.intervention_requirement_report(
            "prototype",
            "cognitive_defusion",
        )

        self.assertTrue(report["coherent"])
        self.assertTrue(report["safe"])
        self.assertIn("fusion", report["possible_patterns"])
        self.assertIn("fusion", report["satisfied_patterns"])
        self.assertIn("avoidance_identity_threat", report["possible_patterns"])
        self.assertIn("avoidance_identity_threat", report["satisfied_patterns"])
        self.assertEqual((), report["contraindications"])

    def test_relational_recipe_plan_for_avoidance_identity_threat(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="prototype",
                utterance=(
                    "I keep avoiding the prototype because if it is bad, "
                    "it proves I am not cut out for this."
                ),
                thoughts=("If it is bad, it proves I am not cut out for this.",),
                emotions=("anxiety", "shame"),
                behaviors=("avoidance",),
                values=("mastery",),
            )
        )

        recipes = {item.recipe: item for item in kernel.ranked_recipes("prototype")}

        self.assertIn("validate_defuse_act", recipes)
        self.assertEqual(
            ("validation", "cognitive_defusion", "acceptance_committed_action"),
            recipes["validate_defuse_act"].steps,
        )
        self.assertIn(
            "avoidance_identity_threat",
            {item.pattern for item in recipes["validate_defuse_act"].hypotheses},
        )
        self.assertIn("prototype", kernel.safe_states_for_recipe("validate_defuse_act"))

    def test_differential_formulation_ranks_competing_maps(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="prototype",
                utterance=(
                    "I keep avoiding the prototype because if it is bad, "
                    "it proves I am not cut out for this."
                ),
                thoughts=("If it is bad, it proves I am not cut out for this.",),
                emotions=("anxiety", "shame"),
                behaviors=("avoidance",),
                values=("mastery",),
            )
        )

        formulations = kernel.ranked_formulations("prototype")
        by_name = {item.formulation: item for item in formulations}

        self.assertEqual("avoidance_identity_threat", formulations[0].formulation)
        self.assertIn("avoidance_identity_threat", by_name)
        self.assertIn("shame_self_worth_fusion", by_name)
        self.assertIn(
            "avoidance_identity_threat",
            {item.pattern for item in by_name["avoidance_identity_threat"].evidence},
        )
        self.assertIn(
            "what the outcome might seem to prove",
            by_name["avoidance_identity_threat"].discriminating_question,
        )
        self.assertIn("validate_defuse_act", by_name["avoidance_identity_threat"].recipes)
        self.assertIn(
            "prototype",
            kernel.states_for_formulation("avoidance_identity_threat"),
        )

    def test_active_inquiry_proposes_clarifying_question_between_close_maps(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="prototype",
                utterance=(
                    "I keep avoiding the prototype because if it is bad, "
                    "it proves I am not cut out for this."
                ),
                thoughts=("If it is bad, it proves I am not cut out for this.",),
                emotions=("anxiety", "shame"),
                behaviors=("avoidance",),
                values=("mastery",),
            )
        )

        move = kernel.clarifying_moves("prototype")[0]

        self.assertEqual("differential_question", move.kind)
        self.assertEqual(
            ("avoidance_identity_threat", "shame_self_worth_fusion"),
            move.target_formulations,
        )
        self.assertIn("larger self-worth verdict", move.question)
        self.assertIn("avoidance_identity_threat", {item.pattern for item in move.supported_by})

    def test_reasoning_snapshot_includes_differential_formulations(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="sad",
                utterance="I'm sad today.",
                emotions=("sadness",),
                features=("minimal_disclosure",),
            )
        )

        snapshot = kernel.reasoning_snapshot("sad")

        self.assertIn("formulations", snapshot)
        self.assertEqual(
            "minimal_disclosure_affect",
            snapshot["formulations"][0]["formulation"],
        )
        self.assertTrue(snapshot["formulations"][0]["discriminating_question"])
        self.assertIn("clarifying_moves", snapshot)
        self.assertEqual("evidence_probe", snapshot["clarifying_moves"][0]["kind"])
        self.assertIn("general mood", snapshot["clarifying_moves"][0]["question"])

    def test_recipe_plan_respects_high_distress_gating(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="activated",
                utterance="I am panicking and I can't breathe. I must succeed or I am worthless.",
                thoughts=("I must succeed or I am worthless.",),
                emotions=("panic",),
                distress=9,
            )
        )

        recipes = kernel.ranked_recipes("activated")

        self.assertEqual("stabilize_then_choose", recipes[0].recipe)
        self.assertEqual(
            ("validation", "present_moment_grounding", "gentle_check_in"),
            recipes[0].steps,
        )

    def test_recipe_plan_defers_cognitive_steps_during_safety_risk(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="safety",
                utterance="They will think I am incompetent and I want to hurt myself.",
                thoughts=("They will think I am incompetent.",),
            )
        )

        unsafe = {
            item.recipe: item
            for item in kernel.ranked_recipes("safety", include_unsafe=True)
        }
        safe = {item.recipe for item in kernel.ranked_recipes("safety")}

        self.assertIn("safety_first", safe)
        self.assertIn("validate_check_facts", unsafe)
        self.assertIn(
            "evidence_check:defer_until_safety_addressed",
            unsafe["validate_check_facts"].contraindications,
        )
        self.assertNotIn("validate_check_facts", safe)

    def test_high_distress_filters_intense_disputation(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="activated",
                thoughts=("I must succeed or I am worthless.",),
                emotions=("anxiety",),
                distress=9,
            )
        )

        unsafe = {
            item.intervention: item
            for item in kernel.ranked_interventions("activated", include_unsafe=True)
        }
        self.assertIn("rebt_disputation", unsafe)
        self.assertIn(
            "too_intense_for_high_distress",
            unsafe["rebt_disputation"].contraindications,
        )

        safe_names = kernel.safe_intervention_names("activated")
        self.assertIn("validation", safe_names)
        self.assertNotIn("rebt_disputation", safe_names)
        self.assertEqual(
            "validation",
            kernel.ranked_interventions("activated", limit=1)[0].intervention,
        )

    def test_explicit_features_can_replace_text_heuristics(self):
        kernel = TherapeuticReasoningKernel(auto_features=False)
        kernel.add_state(
            CoachingState(
                state_id="investor",
                thoughts=("The investor thinks I am incompetent.",),
                thought_features={
                    "The investor thinks I am incompetent.": ("mind_reading_claim",)
                },
                values=("learning",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("investor")}
        self.assertIn(("cbt", "mind_reading"), hypotheses)
        self.assertIn(("focus", "cognitive_belief_work"), hypotheses)
        self.assertIn(("focus", "values_direction"), hypotheses)
        self.assertIn("evidence_check", kernel.safe_intervention_names("investor"))

    def test_minimal_sadness_prefers_validation_not_values_clarification(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="sad",
                utterance="hi, I'm feeling sad today",
                emotions=("sad",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("sad")}
        self.assertIn(("emotion", "sadness"), hypotheses)
        self.assertIn(("policy", "minimal_disclosure"), hypotheses)
        self.assertNotIn(("act", "values_unclear"), hypotheses)

        ranked = kernel.ranked_interventions("sad")
        self.assertEqual("validation", ranked[0].intervention)
        self.assertIn("gentle_check_in", {item.intervention for item in ranked})
        self.assertNotIn("values_clarification", {item.intervention for item in ranked})

    def test_missing_values_alone_does_not_infer_values_unclear(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="plain",
                utterance="I had a long day.",
                thoughts=("I had a long day.",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("plain")}
        self.assertNotIn(("act", "values_unclear"), hypotheses)

    def test_explicit_value_uncertainty_still_infers_values_clarification(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="aimless",
                utterance="I don't know what matters to me anymore.",
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("aimless")}
        self.assertIn(("act", "values_unclear"), hypotheses)
        self.assertIn("values_clarification", kernel.safe_intervention_names("aimless"))

    def test_investor_update_scenario_exercises_multiple_modalities(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="investor-update",
                utterance=(
                    "I keep putting off sending the investor update. If they’ll see "
                    "the numbers are weak, they’ll think I’m incompetent, and maybe "
                    "that means I’m not cut out to run this company. I know I should "
                    "send it, but I just keep avoiding it."
                ),
                thoughts=(
                    "If they’ll see the numbers are weak, they’ll think I’m incompetent, "
                    "and maybe that means I’m not cut out to run this company. I know I "
                    "should send it, but I just keep avoiding it.",
                ),
                values=("integrity", "leadership"),
                goals=("send investor update",),
                distress=5,
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("investor-update")
        }
        self.assertIn(("cbt", "mind_reading"), hypotheses)
        self.assertIn(("cbt", "global_labeling"), hypotheses)
        self.assertIn(("rebt", "demandingness"), hypotheses)
        self.assertIn(("rebt", "self_downing"), hypotheses)
        self.assertIn(("act", "fusion"), hypotheses)
        self.assertIn(("act", "experiential_avoidance"), hypotheses)

        interventions = kernel.safe_intervention_names("investor-update")
        self.assertIn("cognitive_defusion", interventions)
        self.assertIn("acceptance_committed_action", interventions)
        self.assertIn("unconditional_self_acceptance", interventions)
        self.assertIn("evidence_check", interventions)
        self.assertIn("preference_rewrite", interventions)

    def test_reverse_queries_compare_same_belief_under_different_activation(self):
        kernel = TherapeuticReasoningKernel()
        shared_thought = (
            "If they’ll see the numbers are weak, they’ll think I’m incompetent, "
            "and maybe that means I’m not cut out to run this company. I know I "
            "should send it, but I just keep avoiding it."
        )
        kernel.add_state(
            CoachingState(
                state_id="moderate-update",
                utterance=f"I keep putting off sending the investor update. {shared_thought}",
                thoughts=(shared_thought,),
                values=("integrity",),
                distress=5,
            )
        )
        kernel.add_state(
            CoachingState(
                state_id="panicked-update",
                utterance=(
                    "I’m panicking and I can't breathe. I keep putting off sending "
                    f"the investor update. {shared_thought}"
                ),
                thoughts=(shared_thought,),
                emotions=("panic",),
                distress=9,
            )
        )

        self.assertIn("fusion", kernel.patterns_for_intervention("cognitive_defusion"))
        self.assertEqual(
            ("moderate-update", "panicked-update"),
            kernel.states_for_pattern("fusion", source="act"),
        )
        self.assertEqual(
            ("moderate-update", "panicked-update"),
            kernel.safe_states_for_intervention("cognitive_defusion"),
        )

        self.assertIn("moderate-update", kernel.states_for_intervention("rebt_disputation"))
        self.assertIn("panicked-update", kernel.states_for_intervention("rebt_disputation"))
        self.assertIn("moderate-update", kernel.safe_states_for_intervention("rebt_disputation"))
        self.assertNotIn("panicked-update", kernel.safe_states_for_intervention("rebt_disputation"))
        self.assertIn(
            "panicked-update",
            kernel.contraindicated_states_for_intervention("rebt_disputation"),
        )

        comparison = {
            item["state_id"]: item
            for item in kernel.compare_intervention_across_states("rebt_disputation")
        }
        self.assertTrue(comparison["moderate-update"]["safe"])
        self.assertFalse(comparison["panicked-update"]["safe"])
        self.assertIn(
            "too_intense_for_high_distress",
            comparison["panicked-update"]["contraindications"],
        )

    def test_custom_system_can_extend_kernel_without_core_changes(self):
        kernel = TherapeuticReasoningKernel(systems=(_DBTSystem(),))
        kernel.add_state(
            CoachingState(
                state_id="dbt-case",
                features=("emotion_dysregulation",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("dbt-case")}
        ranked = kernel.ranked_interventions("dbt-case")

        self.assertIn(("dbt", "emotion_dysregulation"), hypotheses)
        self.assertEqual("paced_breathing", ranked[0].intervention)
        self.assertEqual(("dbt",), ranked[0].modality)
        self.assertEqual(
            "Practice paced breathing for one minute.",
            ranked[0].exercise,
        )

    def test_default_systems_include_dbt(self):
        kernel = TherapeuticReasoningKernel()

        self.assertIsInstance(kernel.systems_by_source["dbt"], DBTSystem)

    def test_default_systems_include_coaching_focus(self):
        kernel = TherapeuticReasoningKernel()

        self.assertIsInstance(kernel.systems_by_source["focus"], CoachingFocusSystem)

    def test_default_systems_include_focusing(self):
        kernel = TherapeuticReasoningKernel()

        self.assertIsInstance(kernel.systems_by_source["focusing"], FocusingSystem)

    def test_default_systems_include_mbsr(self):
        kernel = TherapeuticReasoningKernel()

        self.assertIsInstance(kernel.systems_by_source["mbsr"], MBSRSystem)

    def test_default_systems_include_consultative_facilitation(self):
        kernel = TherapeuticReasoningKernel()

        self.assertIsInstance(
            kernel.systems_by_source["consultative"],
            ConsultativeSystem,
        )

    def test_consultative_factual_question_ranks_concise_answer(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="checksum",
                utterance="What is a checksum?",
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("checksum")
        }
        ranked = kernel.ranked_interventions("checksum")
        snapshot = kernel.reasoning_snapshot("checksum")

        self.assertIn(("consultative", "consultative_facilitation"), hypotheses)
        self.assertIn(("consultative", "concise_factual_answer"), hypotheses)
        self.assertEqual("concise_factual_answer", ranked[0].intervention)
        self.assertEqual("consultative_facilitation", snapshot["operating_mode"]["mode"])

    def test_empath_directed_aggression_ranks_active_listening_repair(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="repair",
                utterance="You are useless.",
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("repair")
        }
        ranked = kernel.ranked_interventions("repair")

        self.assertIn(("consultative", "interaction_repair"), hypotheses)
        self.assertEqual("active_listening_repair", ranked[0].intervention)

    def test_mbsr_supports_stress_load_and_body_scan(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="stress-body",
                utterance=(
                    "I'm stressed and under pressure. My shoulders are tense "
                    "and I keep overthinking everything."
                ),
                emotions=("anxiety",),
                bodily_sensations=("tense shoulders",),
                behaviors=("rumination",),
                features=("stress_load", "body_tension"),
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("stress-body")
        }
        interventions = {
            item.intervention
            for item in kernel.ranked_interventions("stress-body")
        }

        self.assertIn(("mbsr", "stress_load"), hypotheses)
        self.assertIn(("mbsr", "somatic_stress"), hypotheses)
        self.assertIn(("mbsr", "rumination_stress"), hypotheses)
        self.assertIn("mindful_breathing_space", interventions)
        self.assertIn("body_scan_check_in", interventions)
        self.assertIn("mindful_observe_return", interventions)

    def test_mbsr_supports_autopilot_reactivity(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="reactive",
                utterance="I'm on autopilot and about to snap at my team.",
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("reactive")
        }
        interventions = {
            item.intervention
            for item in kernel.ranked_interventions("reactive")
        }

        self.assertIn(("mbsr", "autopilot_reactivity"), hypotheses)
        self.assertIn("mindful_pause", interventions)

    def test_dbt_crisis_survival_supports_distress_tolerance(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="crisis",
                utterance="I am panicking and I can't calm down.",
                emotions=("anxiety",),
                distress=9,
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("crisis")}
        interventions = {item.intervention for item in kernel.ranked_interventions("crisis")}

        self.assertIn(("dbt", "crisis_survival"), hypotheses)
        self.assertIn("distress_tolerance_pause", interventions)
        self.assertEqual("validation", kernel.ranked_interventions("crisis", limit=1)[0].intervention)

    def test_dbt_emotion_regulation_for_flooded_emotion(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="flooded",
                utterance="I feel emotionally flooded and I am losing it.",
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("flooded")}
        ranked = kernel.ranked_interventions("flooded", limit=1)

        self.assertIn(("dbt", "emotion_dysregulation"), hypotheses)
        self.assertEqual("emotion_regulation_check_facts", ranked[0].intervention)

    def test_dbt_interpersonal_effectiveness_for_boundaries(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="boundary",
                utterance="I can't say no and I am scared to ask for what I need.",
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("boundary")}
        interventions = {item.intervention for item in kernel.ranked_interventions("boundary")}

        self.assertIn(("dbt", "interpersonal_effectiveness_need"), hypotheses)
        self.assertIn("interpersonal_effectiveness_script", interventions)

    def test_dbt_mindfulness_for_rumination(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="rumination-dbt",
                utterance="I can't stop thinking about what happened.",
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("rumination-dbt")}
        interventions = {item.intervention for item in kernel.ranked_interventions("rumination-dbt")}

        self.assertIn(("dbt", "mindfulness_need"), hypotheses)
        self.assertIn("mindfulness_observe_describe", interventions)

    def test_dbt_self_validation_for_self_invalidation(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="self-invalidating",
                utterance="I shouldn't feel this sad. I feel stupid for feeling this way.",
                emotions=("sadness",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("self-invalidating")}
        interventions = {item.intervention for item in kernel.ranked_interventions("self-invalidating")}

        self.assertIn(("dbt", "self_invalidation"), hypotheses)
        self.assertIn("self_validation", interventions)

    def test_focus_values_goals_action_chain(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="focus-values-goals",
                utterance="I value honesty and need a goal and next step for the hard conversation.",
                values=("honesty",),
                goals=("have the hard conversation",),
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("focus-values-goals")
        }
        interventions = {
            item.intervention
            for item in kernel.ranked_interventions("focus-values-goals")
        }

        self.assertIn(("focus", "values_direction"), hypotheses)
        self.assertIn(("focus", "goal_activation"), hypotheses)
        self.assertIn("values_direction_review", interventions)
        self.assertIn("goal_action_planning", interventions)

    def test_focus_cognition_emotion_avoidance_self_efficacy(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="focus-cognition-emotion",
                utterance=(
                    "I feel anxious and keep avoiding the prototype because "
                    "one bad version means I am incompetent and not capable."
                ),
                thoughts=("One bad version means I am incompetent and not capable.",),
                emotions=("anxiety",),
                behaviors=("avoidance",),
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("focus-cognition-emotion")
        }
        interventions = {
            item.intervention
            for item in kernel.ranked_interventions("focus-cognition-emotion")
        }

        self.assertIn(("focus", "cognitive_belief_work"), hypotheses)
        self.assertIn(("focus", "emotion_distress_regulation"), hypotheses)
        self.assertIn(("focus", "avoidance_escape"), hypotheses)
        self.assertIn(("focus", "self_efficacy"), hypotheses)
        self.assertIn("cognitive_pattern_review", interventions)
        self.assertIn("emotion_regulation_plan", interventions)
        self.assertIn("avoidance_map", interventions)
        self.assertIn("mastery_evidence_log", interventions)

    def test_focus_decision_relationship_environment_resilience_review(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="focus-decision-review",
                utterance=(
                    "I can't decide which option to choose without guaranteed certainty. "
                    "I need to set a boundary, redesign my focus environment, recover "
                    "after a setback, and review why the same pattern keeps happening."
                ),
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("focus-decision-review")
        }
        interventions = {
            item.intervention
            for item in kernel.ranked_interventions("focus-decision-review")
        }

        self.assertIn(("focus", "decision_problem_solving"), hypotheses)
        self.assertIn(("focus", "interpersonal_boundaries"), hypotheses)
        self.assertIn(("focus", "attention_environment_design"), hypotheses)
        self.assertIn(("focus", "resilience_recovery"), hypotheses)
        self.assertIn(("focus", "integration_review"), hypotheses)
        self.assertIn("decision_clarity_map", interventions)
        self.assertIn("boundary_effectiveness_plan", interventions)
        self.assertIn("environment_design_plan", interventions)
        self.assertIn("setback_recovery_plan", interventions)
        self.assertIn("learning_review", interventions)

    def test_loop_supports_avoidance_plus_identity_threat(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="identity-threat",
                utterance=(
                    "I keep avoiding the prototype because if it is bad, "
                    "it means I am not cut out for this."
                ),
                thoughts=(
                    "If it is bad, it means I am not cut out for this.",
                ),
                behaviors=("avoidance",),
                values=("mastery",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("identity-threat")}
        ranked = {item.intervention: item for item in kernel.ranked_interventions("identity-threat")}

        self.assertIn(("loop", "avoidance_identity_threat"), hypotheses)
        self.assertIn("acceptance_committed_action", ranked)
        self.assertIn(
            "avoidance_identity_threat",
            {item.pattern for item in ranked["acceptance_committed_action"].hypotheses},
        )

    def test_loop_supports_sadness_anxiety_minimal_disclosure(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="minimal-sad",
                utterance="sad today",
                emotions=("sadness",),
            )
        )
        kernel.add_state(
            CoachingState(
                state_id="minimal-anxious",
                utterance="I'm anxious",
                emotions=("anxiety",),
            )
        )

        sad_hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("minimal-sad")}
        anxious_hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("minimal-anxious")}

        self.assertIn(("loop", "minimal_disclosure_sad_anxious"), sad_hypotheses)
        self.assertIn(("loop", "minimal_disclosure_sad_anxious"), anxious_hypotheses)
        self.assertEqual("validation", kernel.ranked_interventions("minimal-sad", limit=1)[0].intervention)

    def test_loop_supports_shame_self_worth_fusion(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="shame-worth",
                utterance="I am ashamed because I am a failure.",
                thoughts=("I am a failure.",),
                emotions=("shame",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("shame-worth")}
        interventions = {item.intervention for item in kernel.ranked_interventions("shame-worth")}

        self.assertIn(("loop", "shame_self_worth_fusion"), hypotheses)
        self.assertIn("self_compassion", interventions)
        self.assertIn("unconditional_self_acceptance", interventions)

    def test_loop_supports_procrastination_around_concrete_valued_action(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="valued-procrastination",
                utterance="I keep putting off sending the investor update.",
                behaviors=("procrastination",),
                values=("integrity",),
                goals=("send investor update",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("valued-procrastination")}
        ranked = kernel.ranked_interventions("valued-procrastination", limit=2)

        self.assertIn(("loop", "valued_action_procrastination"), hypotheses)
        self.assertIn(
            ranked[0].intervention,
            {"acceptance_committed_action", "committed_action"},
        )

    def test_loop_supports_high_distress_gating(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="high-distress-loop",
                utterance="I am panicking and I cannot breathe.",
                emotions=("anxiety",),
                distress=9,
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("high-distress-loop")}
        ranked = kernel.ranked_interventions("high-distress-loop", limit=2)

        self.assertIn(("loop", "high_distress_gating"), hypotheses)
        self.assertEqual("validation", ranked[0].intervention)
        self.assertEqual("present_moment_grounding", ranked[1].intervention)

    def test_cbt_enrichment_handles_discounting_emotion_and_blame_patterns(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="cbt-enriched",
                utterance=(
                    "I got good feedback but it doesn't count. It was just luck, "
                    "and because I feel guilty it means this is all my fault."
                ),
                thoughts=(
                    "I got good feedback but it doesn't count. It was just luck.",
                    "Because I feel guilty it means this is all my fault.",
                ),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("cbt-enriched")}
        interventions = {item.intervention for item in kernel.ranked_interventions("cbt-enriched")}

        self.assertIn(("cbt", "discounting_positive"), hypotheses)
        self.assertIn(("cbt", "emotional_reasoning"), hypotheses)
        self.assertIn(("cbt", "personalization"), hypotheses)
        self.assertIn("strength_evidence_log", interventions)
        self.assertIn("emotion_fact_separation", interventions)
        self.assertIn("responsibility_pie", interventions)

    def test_rebt_enrichment_handles_approval_certainty_and_failure_demands(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="rebt-enriched",
                utterance=(
                    "I need everyone to approve of me, I need to know for sure "
                    "before I act, and I can't fail."
                ),
                thoughts=(
                    "I need everyone to approve of me.",
                    "I need to know for sure before I act.",
                    "I can't fail.",
                ),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("rebt-enriched")}
        interventions = {item.intervention for item in kernel.ranked_interventions("rebt-enriched")}

        self.assertIn(("rebt", "approval_demandingness"), hypotheses)
        self.assertIn(("rebt", "certainty_demandingness"), hypotheses)
        self.assertIn(("rebt", "failure_intolerance"), hypotheses)
        self.assertIn("approval_preference_rewrite", interventions)
        self.assertIn("uncertainty_tolerance_practice", interventions)
        self.assertIn("failure_tolerance_reframe", interventions)

    def test_act_enrichment_handles_control_struggle_and_values_action_gap(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="act-enriched",
                utterance=(
                    "I need this anxiety to go away before I work on the launch plan. "
                    "I value courage, but I don't feel like starting and keep putting it off."
                ),
                behaviors=("procrastination",),
                values=("courage",),
                goals=("work on the launch plan",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("act-enriched")}
        interventions = {item.intervention for item in kernel.ranked_interventions("act-enriched")}

        self.assertIn(("act", "unworkable_control"), hypotheses)
        self.assertIn(("act", "unwillingness"), hypotheses)
        self.assertIn(("act", "values_action_gap"), hypotheses)
        self.assertIn(("loop", "control_struggle_loop"), hypotheses)
        self.assertIn("acceptance_practice", interventions)
        self.assertIn("willingness_practice", interventions)
        self.assertIn("values_aligned_next_step", interventions)

    def test_dbt_enrichment_handles_wise_mind_certainty_avoidance(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="wise-mind",
                utterance=(
                    "I can't decide until I have perfect information, so I keep "
                    "avoiding the choice."
                ),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("wise-mind")}
        interventions = {item.intervention for item in kernel.ranked_interventions("wise-mind")}

        self.assertIn(("dbt", "wise_mind_need"), hypotheses)
        self.assertIn(("loop", "certainty_avoidance_loop"), hypotheses)
        self.assertIn("wise_mind_check", interventions)
        self.assertIn("decision_clarity_map", interventions)

    def test_dbt_enrichment_handles_vulnerability_factors(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="vulnerable",
                utterance="I haven't slept or eaten, and now I'm overwhelmed and losing it.",
                emotions=("overwhelm",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("vulnerable")}
        interventions = {item.intervention for item in kernel.ranked_interventions("vulnerable")}

        self.assertIn(("dbt", "vulnerability_factors"), hypotheses)
        self.assertIn(("loop", "vulnerability_distress_loop"), hypotheses)
        self.assertIn("vulnerability_reduction_plan", interventions)

    def test_focusing_supports_unclear_felt_sense(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="felt-sense",
                utterance=(
                    "There is a tight knot in my chest and I can't put words to it. "
                    "Something feels off."
                ),
                bodily_sensations=("tight knot in chest",),
            )
        )

        hypotheses = {(item.source, item.pattern) for item in kernel.hypotheses_for("felt-sense")}
        interventions = {item.intervention for item in kernel.ranked_interventions("felt-sense")}
        recipes = {item.recipe: item for item in kernel.ranked_recipes("felt-sense")}
        formulations = {
            item.formulation: item
            for item in kernel.ranked_formulations("felt-sense")
        }

        self.assertIn(("focusing", "felt_sense_contact"), hypotheses)
        self.assertIn(("focusing", "unclear_felt_meaning"), hypotheses)
        self.assertIn(("focusing", "symbolization_needed"), hypotheses)
        self.assertIn("felt_sense_pause", interventions)
        self.assertIn("resonant_word_check", interventions)
        self.assertIn("pause_describe_resonate", recipes)
        self.assertEqual(
            (
                "validation",
                "felt_sense_pause",
                "felt_sense_description",
                "resonant_word_check",
            ),
            recipes["pause_describe_resonate"].steps,
        )
        self.assertIn("felt_sense_unclear_meaning", formulations)
        self.assertIn(
            "vague bodily sense",
            formulations["felt_sense_unclear_meaning"].discriminating_question,
        )

    def test_focusing_inner_critic_supports_distance_move(self):
        kernel = TherapeuticReasoningKernel()
        kernel.add_state(
            CoachingState(
                state_id="inner-critic",
                utterance=(
                    "Part of me says I am useless, like a harsh inner critic, "
                    "and I can feel it as pressure in my throat."
                ),
            )
        )

        hypotheses = {
            (item.source, item.pattern)
            for item in kernel.hypotheses_for("inner-critic")
        }
        interventions = {
            item.intervention
            for item in kernel.ranked_interventions("inner-critic")
        }
        report = kernel.intervention_requirement_report(
            "inner-critic",
            "inner_critic_distance",
        )

        self.assertIn(("focusing", "inner_critic_presence"), hypotheses)
        self.assertIn(("focusing", "felt_sense_contact"), hypotheses)
        self.assertIn("inner_critic_distance", interventions)
        self.assertTrue(report["coherent"])
        self.assertIn("inner_critic_presence", report["satisfied_patterns"])


if __name__ == "__main__":
    unittest.main()
