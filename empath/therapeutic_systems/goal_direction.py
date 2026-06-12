"""Goal-direction and execution methodology layer."""

from __future__ import annotations

from typing import Any

from kanren import conde, eq, facts, var

from .base import TherapeuticSystem


class GoalDirectionSystem(TherapeuticSystem):
    """Relational planning layer: values, outcomes, obstacles, and next actions."""

    source = "goal_direction"

    def load_ontology(self, kernel: Any) -> None:
        facts(
            kernel.pattern_intervention,
            ("values_to_objective_gap", "clarify_objective"),
            ("objective_without_next_action", "define_next_action"),
            ("goal_obstacle_gap", "woop_obstacle_plan"),
            ("open_loop_overload", "capture_open_loops"),
            ("wip_overload", "limit_work_in_progress"),
            ("implementation_intention_need", "implementation_intention"),
            ("scope_too_large", "reduce_scope"),
            ("waiting_for_tracking", "waiting_for_review"),
            ("review_recommitment_needed", "weekly_review"),
            ("success_measure_missing", "define_success_measure"),
        )
        facts(
            kernel.intervention_modality,
            ("clarify_objective", "goal_direction"),
            ("define_next_action", "goal_direction"),
            ("woop_obstacle_plan", "goal_direction"),
            ("capture_open_loops", "goal_direction"),
            ("limit_work_in_progress", "goal_direction"),
            ("implementation_intention", "goal_direction"),
            ("reduce_scope", "goal_direction"),
            ("waiting_for_review", "goal_direction"),
            ("weekly_review", "goal_direction"),
            ("define_success_measure", "goal_direction"),
        )
        facts(
            kernel.intervention_exercise,
            (
                "clarify_objective",
                "Write one plain-language objective: the meaningful outcome you are moving toward, without making it a verdict on you.",
            ),
            (
                "define_next_action",
                "Name the next visible action so concretely that someone could watch you do it in 10 minutes or less.",
            ),
            (
                "woop_obstacle_plan",
                "Use WOOP: wish, outcome, obstacle, then an if-then plan for the obstacle that is most likely to interrupt action.",
            ),
            (
                "capture_open_loops",
                "Do a two-minute capture of every open loop, then mark only one item as the next action to clarify.",
            ),
            (
                "limit_work_in_progress",
                "Pick no more than three active items for now; move the rest to later, waiting, or parked.",
            ),
            (
                "implementation_intention",
                "Write one if-then plan: if the predictable obstacle appears, then I will do the smallest next action.",
            ),
            (
                "reduce_scope",
                "Shrink the action until it is small enough to start today without needing confidence or a perfect plan.",
            ),
            (
                "waiting_for_review",
                "List what is waiting on someone or something else, the owner, and the next check-in date.",
            ),
            (
                "weekly_review",
                "Review what moved, what stalled, what mattered, and the one commitment to carry into the next week.",
            ),
            (
                "define_success_measure",
                "Define one observable sign that would count as progress, even if the whole outcome is not finished yet.",
            ),
        )
        facts(
            kernel.pattern_recipe,
            ("values_to_objective_gap", "values_to_objective"),
            ("objective_without_next_action", "objective_to_next_action"),
            ("goal_obstacle_gap", "woop_then_next_action"),
            ("open_loop_overload", "capture_clarify_engage"),
            ("wip_overload", "kanban_limit_choose"),
            ("implementation_intention_need", "if_then_start_plan"),
            ("review_recommitment_needed", "review_learn_recommit"),
            ("success_measure_missing", "objective_measure_action"),
        )
        facts(
            kernel.recipe_step,
            ("values_to_objective", 0, "validation"),
            ("values_to_objective", 1, "clarify_objective"),
            ("values_to_objective", 2, "define_success_measure"),
            ("objective_to_next_action", 0, "validation"),
            ("objective_to_next_action", 1, "define_next_action"),
            ("woop_then_next_action", 0, "validation"),
            ("woop_then_next_action", 1, "woop_obstacle_plan"),
            ("woop_then_next_action", 2, "implementation_intention"),
            ("capture_clarify_engage", 0, "validation"),
            ("capture_clarify_engage", 1, "capture_open_loops"),
            ("capture_clarify_engage", 2, "define_next_action"),
            ("kanban_limit_choose", 0, "validation"),
            ("kanban_limit_choose", 1, "limit_work_in_progress"),
            ("kanban_limit_choose", 2, "define_next_action"),
            ("if_then_start_plan", 0, "validation"),
            ("if_then_start_plan", 1, "implementation_intention"),
            ("if_then_start_plan", 2, "reduce_scope"),
            ("review_learn_recommit", 0, "validation"),
            ("review_learn_recommit", 1, "weekly_review"),
            ("review_learn_recommit", 2, "define_next_action"),
            ("objective_measure_action", 0, "validation"),
            ("objective_measure_action", 1, "define_success_measure"),
            ("objective_measure_action", 2, "define_next_action"),
        )
        facts(
            kernel.recipe_rationale,
            (
                "values_to_objective",
                "Translate values into a concrete objective and one observable measure.",
            ),
            (
                "objective_to_next_action",
                "Keep the goal from staying abstract by naming the next visible action.",
            ),
            (
                "woop_then_next_action",
                "Plan for the predictable obstacle before asking for follow-through.",
            ),
            (
                "capture_clarify_engage",
                "Reduce mental clutter, clarify one next action, then engage.",
            ),
            (
                "kanban_limit_choose",
                "Limit active work so execution is not overloaded by too many simultaneous commitments.",
            ),
            ("if_then_start_plan", "Pair a likely obstacle with a tiny start plan."),
            (
                "review_learn_recommit",
                "Treat execution as a learning loop and recommit to the next action.",
            ),
            (
                "objective_measure_action",
                "Make progress observable, then choose the next behavior.",
            ),
        )
        facts(
            kernel.formulation_pattern,
            ("execution_pathway_gap", "values_to_objective_gap"),
            ("execution_pathway_gap", "objective_without_next_action"),
            ("execution_pathway_gap", "success_measure_missing"),
            ("obstacle_planning_gap", "goal_obstacle_gap"),
            ("obstacle_planning_gap", "implementation_intention_need"),
            ("open_loop_overload", "open_loop_overload"),
            ("open_loop_overload", "wip_overload"),
            ("review_recommitment_gap", "review_recommitment_needed"),
        )
        facts(
            kernel.formulation_label,
            ("execution_pathway_gap", "Goal-direction execution gap"),
            ("obstacle_planning_gap", "Obstacle plan gap"),
            ("open_loop_overload", "Open-loop overload"),
            ("review_recommitment_gap", "Review and recommitment gap"),
        )
        facts(
            kernel.formulation_summary,
            (
                "execution_pathway_gap",
                "The practical constraint may be translating direction into an observable next action and measure.",
            ),
            (
                "obstacle_planning_gap",
                "The goal may be clear enough, but the predictable obstacle needs an if-then plan.",
            ),
            (
                "open_loop_overload",
                "Too many open loops or active items may be creating execution drag.",
            ),
            (
                "review_recommitment_gap",
                "A short review may be needed before choosing what to carry forward.",
            ),
        )
        facts(
            kernel.formulation_discriminator,
            (
                "execution_pathway_gap",
                "Is the objective unclear, the success measure unclear, or the next action unclear?",
            ),
            (
                "obstacle_planning_gap",
                "What obstacle most predictably interrupts the next action?",
            ),
            (
                "open_loop_overload",
                "Is the main problem one task, or too many open loops competing for attention?",
            ),
            (
                "review_recommitment_gap",
                "What did the last attempt teach you that should change the next commitment?",
            ),
        )
        facts(
            kernel.formulation_focus,
            ("execution_pathway_gap", "direction to execution"),
            ("obstacle_planning_gap", "obstacle planning"),
            ("open_loop_overload", "execution hygiene"),
            ("review_recommitment_gap", "weekly review"),
        )

    def pattern_goal(self, kernel: Any, state: Any, pattern: Any):
        return conde(
            [
                self._has_directiono(kernel, state),
                self._goal_settingo(kernel, state),
                eq(pattern, "values_to_objective_gap"),
            ],
            [
                self._has_goal_or_projecto(kernel, state),
                eq(pattern, "objective_without_next_action"),
            ],
            [
                self._has_goal_or_projecto(kernel, state),
                self._has_obstacleo(kernel, state),
                eq(pattern, "goal_obstacle_gap"),
            ],
            [
                self._open_loopo(kernel, state),
                eq(pattern, "open_loop_overload"),
            ],
            [
                self._wip_overloado(kernel, state),
                eq(pattern, "wip_overload"),
            ],
            [
                self._has_obstacleo(kernel, state),
                self._start_blocko(kernel, state),
                eq(pattern, "implementation_intention_need"),
            ],
            [
                self._scope_too_largeo(kernel, state),
                eq(pattern, "scope_too_large"),
            ],
            [
                self._waiting_foro(kernel, state),
                eq(pattern, "waiting_for_tracking"),
            ],
            [
                self._review_neededo(kernel, state),
                eq(pattern, "review_recommitment_needed"),
            ],
            [
                self._has_goal_or_projecto(kernel, state),
                self._measure_neededo(kernel, state),
                eq(pattern, "success_measure_missing"),
            ],
        )

    def score_bonus(self, intervention: str, patterns: set[str]) -> float:
        score = 0.0
        goal_direction_interventions = {
            "clarify_objective",
            "define_next_action",
            "woop_obstacle_plan",
            "capture_open_loops",
            "limit_work_in_progress",
            "implementation_intention",
            "reduce_scope",
            "waiting_for_review",
            "weekly_review",
            "define_success_measure",
        }
        if intervention in goal_direction_interventions:
            score -= 0.75
        if "objective_without_next_action" in patterns:
            if intervention == "define_next_action":
                score += 0.75
            if intervention == "reduce_scope":
                score += 0.5
        if "goal_obstacle_gap" in patterns:
            if intervention == "woop_obstacle_plan":
                score += 0.75
            if intervention == "implementation_intention":
                score += 0.5
        if "open_loop_overload" in patterns:
            if intervention == "capture_open_loops":
                score += 0.0
            if intervention == "limit_work_in_progress":
                score += 0.25
        if "wip_overload" in patterns and intervention == "limit_work_in_progress":
            score += 0.0
        if (
            "values_to_objective_gap" in patterns
            and intervention == "clarify_objective"
        ):
            score += 0.75
        if "implementation_intention_need" in patterns:
            if intervention == "implementation_intention":
                score += 0.75
            if intervention == "reduce_scope":
                score += 0.5
        if "review_recommitment_needed" in patterns and intervention == "weekly_review":
            score += 0.0
        if (
            "success_measure_missing" in patterns
            and intervention == "define_success_measure"
        ):
            score += 0.75
        return score

    def _has_directiono(self, kernel: Any, state: Any):
        item = var()
        return conde(
            [kernel.has_value(state, item)],
            [kernel.has_goal(state, item)],
            [kernel.has_objective(state, item)],
        )

    def _has_goal_or_projecto(self, kernel: Any, state: Any):
        item = var()
        return conde(
            [kernel.has_goal(state, item)],
            [kernel.has_objective(state, item)],
            [kernel.has_project(state, item)],
            [kernel.has_task(state, item)],
        )

    def _has_obstacleo(self, kernel: Any, state: Any):
        item = var()
        return conde(
            [kernel.has_obstacle(state, item)],
            [kernel.has_challenge(state, item)],
            [kernel.has_behavior(state, "avoidance")],
            [kernel.has_behavior(state, "procrastination")],
            [kernel.state_feature(state, "obstacle_planning")],
        )

    def _goal_settingo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "goal_setting")],
            [kernel.state_feature(state, "values_unclear")],
        )

    def _open_loopo(self, kernel: Any, state: Any):
        item = var()
        return conde(
            [kernel.state_feature(state, "open_loops")],
            [kernel.state_feature(state, "overloaded_open_loops")],
            [kernel.has_waiting_for(state, item)],
            [kernel.has_behavior(state, "rumination")],
        )

    def _wip_overloado(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "wip_overload")],
            [kernel.state_feature(state, "attention_environment")],
            [kernel.has_emotion(state, "overwhelm")],
        )

    def _start_blocko(self, kernel: Any, state: Any):
        return conde(
            [kernel.has_behavior(state, "avoidance")],
            [kernel.has_behavior(state, "procrastination")],
            [kernel.state_feature(state, "motivation_block")],
            [kernel.state_feature(state, "attention_environment")],
        )

    def _scope_too_largeo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "scope_too_large")],
            [kernel.has_emotion(state, "overwhelm")],
            [kernel.state_feature(state, "wip_overload")],
        )

    def _waiting_foro(self, kernel: Any, state: Any):
        item = var()
        return kernel.has_waiting_for(state, item)

    def _review_neededo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "integration_review")],
            [kernel.state_feature(state, "setback_recovery")],
            [kernel.state_feature(state, "review_due")],
        )

    def _measure_neededo(self, kernel: Any, state: Any):
        return conde(
            [kernel.state_feature(state, "goal_setting")],
            [kernel.state_feature(state, "success_measure_unclear")],
        )
