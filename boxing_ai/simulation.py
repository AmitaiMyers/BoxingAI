"""The match: two boxers, a ring, and the rules for when a punch lands.

``BoxingMatch`` knows nothing about reinforcement learning, rewards, rendering
or who controls the boxers. It takes two commands per frame and reports what
happened as a ``FrameEvents`` record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from boxing_ai.boxer import GUARD_FOR_ZONE, Boxer, BoxerCommand
from boxing_ai.config import GameConfig
from boxing_ai.geometry import heading_vector, segment_point_distance
from boxing_ai.strikes import StrikePhase, StrikeType, TargetZone

BOXER_COUNT = 2


def other_index(boxer_index: int) -> int:
    assert boxer_index in range(BOXER_COUNT), "boxer_index must be 0 or 1"
    return 1 - boxer_index


@dataclass(frozen=True)
class HitEvent:
    """A punch made contact (it may still have been blocked)."""

    attacker_index: int
    strike_type: StrikeType
    damage: float
    # Speed of the glove relative to the defender at contact, in m/s.
    impact_speed: float
    was_blocked: bool
    # Blocked by a guard that was raised in reaction, not held indefinitely.
    was_timed_block: bool


@dataclass(frozen=True)
class EvadeEvent:
    """A punch that would have hit a standing head was slipped."""

    defender_index: int
    strike_type: StrikeType


@dataclass(frozen=True)
class FrameEvents:
    hits: tuple[HitEvent, ...]
    evades: tuple[EvadeEvent, ...]
    newly_exhausted: tuple[int, ...]
    strikes_started: tuple[int, ...]


@dataclass(frozen=True)
class StartPositions:
    """Where a round begins: boxers stand on opposite sides of ``center``."""

    center: np.ndarray
    # Direction from the centre toward boxer 1, in radians.
    axis_angle: float


class BoxingMatch:
    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self.boxers = tuple(Boxer(config) for _ in range(BOXER_COUNT))
        self.frame_count = 0
        self.reset(StartPositions(center=np.zeros(2), axis_angle=0.0))

    @property
    def config(self) -> GameConfig:
        return self._config

    def reset(self, start: StartPositions) -> None:
        offset = self._config.ring.start_offset * heading_vector(start.axis_angle)
        # Each boxer starts facing the other.
        self.boxers[0].reset(start.center - offset, facing=start.axis_angle)
        self.boxers[1].reset(start.center + offset, facing=start.axis_angle + math.pi)
        self.frame_count = 0

    # ------------------------------------------------------------- round state

    @property
    def knocked_out(self) -> tuple[int, ...]:
        return tuple(i for i, boxer in enumerate(self.boxers) if boxer.is_knocked_out)

    @property
    def is_time_up(self) -> bool:
        return self.frame_count >= self._config.timing.round_frames

    @property
    def is_over(self) -> bool:
        return bool(self.knocked_out) or self.is_time_up

    @property
    def time_remaining_fraction(self) -> float:
        return max(0.0, 1.0 - self.frame_count / self._config.timing.round_frames)

    def distance_between_boxers(self) -> float:
        return float(np.linalg.norm(self.boxers[1].position - self.boxers[0].position))

    # -------------------------------------------------------------------- step

    def step(self, commands: tuple[BoxerCommand, BoxerCommand]) -> FrameEvents:
        assert not self.is_over, "the round is over; call reset() first"
        assert len(commands) == BOXER_COUNT, "exactly one command per boxer"

        # Positions are read before anyone moves so that turn order is fair.
        positions = [boxer.position.copy() for boxer in self.boxers]
        for index, boxer in enumerate(self.boxers):
            boxer.begin_frame(commands[index], positions[other_index(index)])

        self._move_boxers()
        for boxer in self.boxers:
            boxer.update_gloves()

        hits: list[HitEvent] = []
        evades: list[EvadeEvent] = []
        for attacker_index in range(BOXER_COUNT):
            self._resolve_strike(attacker_index, hits, evades)

        strikes_started = self._boxers_where(lambda b: b.strike_started_this_frame)
        for boxer in self.boxers:
            boxer.end_frame()
        self.frame_count += 1

        return FrameEvents(
            hits=tuple(hits),
            evades=tuple(evades),
            # Read last: blocking and missing can both empty the tank.
            newly_exhausted=self._boxers_where(lambda b: b.just_exhausted),
            strikes_started=strikes_started,
        )

    def _boxers_where(self, predicate) -> tuple[int, ...]:
        return tuple(i for i, boxer in enumerate(self.boxers) if predicate(boxer))

    # ---------------------------------------------------------------- movement

    def _move_boxers(self) -> None:
        frame_seconds = self._config.timing.frame_seconds
        for boxer in self.boxers:
            boxer.position = boxer.position + boxer.velocity * frame_seconds
        self._keep_inside_ropes()
        self._separate_bodies()
        # Separation can push a cornered boxer back out; the ropes win.
        self._keep_inside_ropes()

    def _keep_inside_ropes(self) -> None:
        limit = self._config.ring.half_size - self._config.body.body_radius
        for boxer in self.boxers:
            boxer.position = np.clip(boxer.position, -limit, limit)

    def _separate_bodies(self) -> None:
        first, second = self.boxers
        minimum_distance = 2.0 * self._config.body.body_radius
        offset = second.position - first.position
        distance = float(np.linalg.norm(offset))
        if distance >= minimum_distance:
            return
        direction = offset / distance if distance > 0.0 else heading_vector(first.facing)
        correction = 0.5 * (minimum_distance - distance) * direction
        first.position = first.position - correction
        second.position = second.position + correction

    # ---------------------------------------------------------- hit resolution

    def _resolve_strike(
        self, attacker_index: int, hits: list[HitEvent], evades: list[EvadeEvent]
    ) -> None:
        attacker = self.boxers[attacker_index]
        if attacker.strike_phase is not StrikePhase.ACTIVE or attacker.strike_connected:
            return

        defender_index = other_index(attacker_index)
        defender = self.boxers[defender_index]
        zone = attacker.strike_spec.target_zone

        if self._glove_reaches(attacker, defender.hurtbox_center(zone), zone):
            hits.append(self._land_strike(attacker_index))
            return

        slipped_it = (
            zone is TargetZone.HEAD
            and defender.is_slipping
            and not attacker.strike_evaded
            and self._glove_reaches(attacker, defender.standing_head_center, zone)
        )
        if slipped_it:
            attacker.strike_evaded = True
            evades.append(EvadeEvent(defender_index, attacker.strike_type))

    def _glove_reaches(self, attacker: Boxer, target: np.ndarray, zone: TargetZone) -> bool:
        """Swept test: did the glove pass within contact distance this frame?"""
        body = self._config.body
        hurtbox_radius = body.head_radius if zone is TargetZone.HEAD else body.torso_radius
        glove = attacker.striking_glove
        distance = segment_point_distance(glove.previous_position, glove.position, target)
        return distance <= body.glove_radius + hurtbox_radius

    def _land_strike(self, attacker_index: int) -> HitEvent:
        attacker = self.boxers[attacker_index]
        defender = self.boxers[other_index(attacker_index)]
        spec = attacker.strike_spec
        strike_type = attacker.strike_type
        defense = self._config.defense

        # An exhausted boxer has no guard, so mitigation is off automatically.
        was_blocked = defender.guard is GUARD_FOR_ZONE[spec.target_zone]
        was_timed_block = (
            was_blocked and defender.guard_held_frames < defense.timed_block_window_frames
        )

        damage = self._config.combat.base_strike_damage * spec.damage_multiplier
        if was_blocked:
            damage *= 1.0 - defense.guard_damage_reduction
        if defender.is_slipping and strike_type is StrikeType.UPPERCUT:
            damage *= defense.slip_uppercut_damage_multiplier

        defender_velocity = np.array([*defender.velocity, 0.0])
        impact_speed = float(
            np.linalg.norm(attacker.striking_glove.velocity - defender_velocity)
        )

        attacker.strike_connected = True
        defender.take_damage(damage)
        if was_blocked:
            defender.pay_block_cost()

        return HitEvent(
            attacker_index=attacker_index,
            strike_type=strike_type,
            damage=damage,
            impact_speed=impact_speed,
            was_blocked=was_blocked,
            was_timed_block=was_timed_block,
        )
