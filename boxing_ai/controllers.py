"""Controllers decide what a boxer does: a human, a script, or (via the
environment) a neural network. Anything with ``reset`` and ``decide`` can
stand in the opposite corner from the learning agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from boxing_ai.boxer import GUARD_FOR_ZONE, Boxer, BoxerCommand, GuardState
from boxing_ai.config import GameConfig
from boxing_ai.simulation import BoxingMatch, other_index
from boxing_ai.strikes import StrikePhase, StrikeType, TargetZone


class BoxerController(Protocol):
    def reset(self) -> None:
        """Called at the start of every round."""

    def decide(self, match: BoxingMatch, boxer_index: int) -> BoxerCommand:
        """Choose the command for ``match.boxers[boxer_index]``."""


@dataclass(frozen=True)
class BotProfile:
    """Personality of the scripted bot."""

    # How often it punches while the opponent is in range.
    strikes_per_second: float
    # Chance that it defends against any given incoming punch.
    defense_probability: float
    # Of the head punches it defends, the fraction it slips instead of blocks.
    slip_preference: float
    # Frames of the opponent's punch it needs to see before it can react.
    reaction_delay_frames: int
    # Distance it tries to keep from the opponent.
    preferred_distance: float
    # It backs off to recover when stamina falls below this...
    rest_below_stamina: float
    # ...and comes back once stamina has climbed above this.
    resume_above_stamina: float

    def __post_init__(self) -> None:
        assert self.strikes_per_second >= 0.0, "strikes_per_second must be >= 0"
        assert 0.0 <= self.defense_probability <= 1.0, "defense_probability in [0, 1]"
        assert 0.0 <= self.slip_preference <= 1.0, "slip_preference in [0, 1]"
        assert self.reaction_delay_frames >= 0, "reaction_delay_frames must be >= 0"
        assert self.preferred_distance > 0.0, "preferred_distance must be > 0"
        assert self.rest_below_stamina < self.resume_above_stamina, (
            "resume threshold must be above the rest threshold"
        )


BOT_PROFILES: Mapping[str, BotProfile] = {
    # A walking heavy bag: approaches, rarely punches, never defends.
    "passive": BotProfile(0.3, 0.0, 0.0, 0, 0.9, 0.15, 0.50),
    "easy": BotProfile(0.8, 0.2, 0.2, 10, 0.9, 0.20, 0.55),
    "medium": BotProfile(1.5, 0.5, 0.3, 6, 0.9, 0.25, 0.60),
    "hard": BotProfile(2.5, 0.8, 0.4, 3, 0.85, 0.30, 0.65),
}

# The bot only reacts to punches thrown from within reach plus this margin.
THREAT_RANGE_MARGIN = 0.3
# It is happy anywhere within this distance of its preferred range.
DISTANCE_TOLERANCE = 0.15
# How often, on average, it changes the direction it circles in.
STRAFE_CHANGES_PER_SECOND = 0.7
STRAFE_CHOICES = (-1, 0, 1)


def strike_reach(config: GameConfig, strike_type: StrikeType) -> float:
    """Greatest distance between two boxers at which this punch can land."""
    spec = config.combat.strikes[strike_type]
    hurtbox_radius = (
        config.body.head_radius
        if spec.target_zone is TargetZone.HEAD
        else config.body.torso_radius
    )
    return spec.extended_offset.forward + config.body.glove_radius + hurtbox_radius


class ScriptedBot:
    """Rule-based sparring partner used for fast headless training."""

    def __init__(self, config: GameConfig, profile: BotProfile, seed: int) -> None:
        self._config = config
        self._profile = profile
        self._rng = np.random.default_rng(seed)
        self._reach = {strike: strike_reach(config, strike) for strike in StrikeType}
        self.reset()

    def reset(self) -> None:
        self._is_resting = False
        self._strafe = 0
        self._reacted_to_current_strike = False
        self._planned_defense: GuardState | None = None

    def decide(self, match: BoxingMatch, boxer_index: int) -> BoxerCommand:
        me = match.boxers[boxer_index]
        opponent = match.boxers[other_index(boxer_index)]
        distance = match.distance_between_boxers()

        if me.is_exhausted:
            return BoxerCommand(forward=-1)

        defense = self._react_to_threat(opponent, distance)
        if defense is not None:
            return BoxerCommand(defense=defense)

        self._update_resting(me)
        if self._is_resting:
            is_crowded = distance < self._profile.preferred_distance + DISTANCE_TOLERANCE
            return BoxerCommand(forward=-1 if is_crowded else 0)

        return BoxerCommand(
            forward=self._approach_direction(distance),
            strafe=self._circle_direction(),
            strike=self._choose_strike(distance),
        )

    # ---------------------------------------------------------------- defense

    def _react_to_threat(self, opponent: Boxer, distance: float) -> GuardState | None:
        is_threatening = opponent.strike_phase in (StrikePhase.WINDUP, StrikePhase.ACTIVE)
        if not is_threatening:
            self._reacted_to_current_strike = False
            self._planned_defense = None
            return None

        has_seen_enough = opponent.strike_frame >= self._profile.reaction_delay_frames
        in_danger = distance <= self._reach[opponent.strike_type] + THREAT_RANGE_MARGIN
        if not self._reacted_to_current_strike and has_seen_enough and in_danger:
            # One roll per incoming punch, not one per decision.
            self._reacted_to_current_strike = True
            self._planned_defense = self._pick_defense(opponent.strike_type)
        return self._planned_defense

    def _pick_defense(self, incoming: StrikeType) -> GuardState | None:
        if self._rng.random() >= self._profile.defense_probability:
            return None
        zone = self._config.combat.strikes[incoming].target_zone
        can_slip = zone is TargetZone.HEAD and incoming is not StrikeType.UPPERCUT
        if can_slip and self._rng.random() < self._profile.slip_preference:
            return GuardState.SLIP
        return GUARD_FOR_ZONE[zone]

    # ----------------------------------------------------------------- pacing

    def _update_resting(self, me: Boxer) -> None:
        if me.stamina < self._profile.rest_below_stamina:
            self._is_resting = True
        elif me.stamina > self._profile.resume_above_stamina:
            self._is_resting = False

    # ---------------------------------------------------------------- offense

    def _choose_strike(self, distance: float) -> StrikeType | None:
        in_range = [strike for strike in StrikeType if distance <= self._reach[strike]]
        if not in_range:
            return None
        if self._rng.random() >= self._per_decision(self._profile.strikes_per_second):
            return None
        return in_range[self._rng.integers(len(in_range))]

    # --------------------------------------------------------------- movement

    def _approach_direction(self, distance: float) -> int:
        if distance > self._profile.preferred_distance + DISTANCE_TOLERANCE:
            return 1
        if distance < self._profile.preferred_distance - DISTANCE_TOLERANCE:
            return -1
        return 0

    def _circle_direction(self) -> int:
        if self._rng.random() < self._per_decision(STRAFE_CHANGES_PER_SECOND):
            self._strafe = STRAFE_CHOICES[self._rng.integers(len(STRAFE_CHOICES))]
        return self._strafe

    def _per_decision(self, rate_per_second: float) -> float:
        """Probability of an event with the given rate on a single decision."""
        return rate_per_second / self._config.timing.decisions_per_second
