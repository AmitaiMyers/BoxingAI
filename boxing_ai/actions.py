"""Action space: how neural-network outputs become a boxer command.

The policy has three independent categorical heads (a ``MultiDiscrete`` space):

==========  =====  ====================================================
Head        Size   Choices
==========  =====  ====================================================
locomotion  9      stand still, or step in one of 8 directions
defense     4      none, high guard, low guard, slip
offense     5      none, jab, cross, hook, uppercut
==========  =====  ====================================================

Priorities when heads disagree are handled by the boxer itself: a punch takes
the hands (cancelling a guard), and nothing new can start while a punch or a
slip is still in progress.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from boxing_ai.boxer import BoxerCommand, GuardState
from boxing_ai.strikes import StrikeType

# (forward, strafe) for each locomotion choice. Index 0 is "stand still".
LOCOMOTION_CHOICES: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
DEFENSE_CHOICES: tuple[GuardState, ...] = (
    GuardState.NONE,
    GuardState.HIGH,
    GuardState.LOW,
    GuardState.SLIP,
)
OFFENSE_CHOICES: tuple[StrikeType | None, ...] = (
    None,
    StrikeType.JAB,
    StrikeType.CROSS,
    StrikeType.HOOK,
    StrikeType.UPPERCUT,
)

ACTION_HEAD_SIZES = (len(LOCOMOTION_CHOICES), len(DEFENSE_CHOICES), len(OFFENSE_CHOICES))


def build_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(ACTION_HEAD_SIZES)


def decode_action(action: np.ndarray) -> BoxerCommand:
    """Translate one ``MultiDiscrete`` sample into a ``BoxerCommand``."""
    assert action.shape == (len(ACTION_HEAD_SIZES),), (
        f"action must have shape ({len(ACTION_HEAD_SIZES)},), got {action.shape}"
    )
    locomotion_index, defense_index, offense_index = (int(value) for value in action)
    forward, strafe = LOCOMOTION_CHOICES[locomotion_index]
    return BoxerCommand(
        forward=forward,
        strafe=strafe,
        defense=DEFENSE_CHOICES[defense_index],
        strike=OFFENSE_CHOICES[offense_index],
    )
