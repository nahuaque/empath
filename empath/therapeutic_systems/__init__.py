"""Default therapeutic reasoning systems."""

from __future__ import annotations

from .act import ACTSystem
from .base import TherapeuticSystem
from .cbt import CBTSystem
from .consultative import ConsultativeSystem
from .dbt import DBTSystem
from .focusing import FocusingSystem
from .focus import CoachingFocusSystem
from .goal_direction import GoalDirectionSystem
from .loops import LoopSystem
from .mbsr import MBSRSystem
from .rebt import REBTSystem


def default_systems() -> tuple[TherapeuticSystem, ...]:
    """Return the default therapeutic system set."""

    return (
        CBTSystem(),
        REBTSystem(),
        ACTSystem(),
        DBTSystem(),
        MBSRSystem(),
        FocusingSystem(),
        GoalDirectionSystem(),
        CoachingFocusSystem(),
        LoopSystem(),
        ConsultativeSystem(),
    )


__all__ = [
    "ACTSystem",
    "CBTSystem",
    "CoachingFocusSystem",
    "ConsultativeSystem",
    "DBTSystem",
    "FocusingSystem",
    "GoalDirectionSystem",
    "LoopSystem",
    "MBSRSystem",
    "REBTSystem",
    "TherapeuticSystem",
    "default_systems",
]
