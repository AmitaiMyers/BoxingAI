"""One boxer: resources, stance, punch state machine and glove kinematics.

The same class is used for the learning agent, the human player and the
scripted bot, so every rule applies to both corners.

A frame is processed in three calls, driven by ``BoxingMatch``:

1. :meth:`Boxer.begin_frame` - turn the controller's command into state
   changes (start a punch, raise a guard, set the movement velocity).
2. :meth:`Boxer.update_gloves` - after the match has moved the bodies, place
   the gloves and derive their velocity and acceleration.
3. :meth:`Boxer.end_frame` - advance timers and settle the stamina bill.

Hits are resolved by the match between steps 2 and 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from boxing_ai.config import GameConfig
from boxing_ai.geometry import (
    PLANAR_DIMENSIONS,
    SPATIAL_DIMENSIONS,
    GloveOffset,
    heading_vector,
    left_vector,
    local_to_world,
    rotate_towards,
)
from boxing_ai.strikes import Hand, StrikePhase, StrikeSpec, StrikeType, TargetZone

FULL_RESOURCE = 1.0
EMPTY_RESOURCE = 0.0


class GuardState(IntEnum):
    NONE = 0
    HIGH = 1
    LOW = 2
    SLIP = 3


# Which guard protects which target zone.
GUARD_FOR_ZONE = {TargetZone.HEAD: GuardState.HIGH, TargetZone.BODY: GuardState.LOW}

# Guards that are held (as opposed to the slip, which is a timed manoeuvre).
HELD_GUARDS = (GuardState.HIGH, GuardState.LOW)


@dataclass(frozen=True)
class BoxerCommand:
    """What a controller asks its boxer to do this frame.

    ``forward`` and ``strafe`` are -1, 0 or +1 in the boxer's local frame
    (+forward is toward where it faces, +strafe is to its left).
    """

    forward: int = 0
    strafe: int = 0
    defense: GuardState = GuardState.NONE
    strike: StrikeType | None = None

    def __post_init__(self) -> None:
        assert self.forward in (-1, 0, 1), "forward must be -1, 0 or 1"
        assert self.strafe in (-1, 0, 1), "strafe must be -1, 0 or 1"
        assert isinstance(self.defense, GuardState), "defense must be a GuardState"
        assert self.strike is None or isinstance(self.strike, StrikeType), (
            "strike must be a StrikeType or None"
        )


IDLE_COMMAND = BoxerCommand()


@dataclass
class GloveState:
    """World-space kinematics of one glove (3D vectors)."""

    position: np.ndarray
    previous_position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class Boxer:
    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self.reset(position=np.zeros(PLANAR_DIMENSIONS), facing=0.0)

    # ------------------------------------------------------------------ setup

    def reset(self, position: np.ndarray, facing: float) -> None:
        assert position.shape == (PLANAR_DIMENSIONS,), "position must be planar"
        self.position = position.astype(np.float64)
        self.velocity = np.zeros(PLANAR_DIMENSIONS)
        self.facing = facing

        self.health = FULL_RESOURCE
        self.stamina = FULL_RESOURCE

        self.guard = GuardState.NONE
        self.guard_held_frames = 0

        self.strike_type: StrikeType | None = None
        self.strike_frame = 0
        self.strike_connected = False
        self.strike_evaded = False

        self.slip_frames_remaining = 0
        self.slip_cooldown_frames_remaining = 0
        self.exhausted_frames_remaining = 0

        # One-frame flags read by the match to build its event list.
        self.just_exhausted = False
        self.strike_started_this_frame = False

        self._locomotion = (0, 0)
        self._strike_start_offset = np.zeros(SPATIAL_DIMENSIONS)
        self._glove_local = {
            hand: self._resting_offset().to_local(hand.side_sign) for hand in Hand
        }
        self.gloves = {hand: self._still_glove(hand) for hand in Hand}

    def _still_glove(self, hand: Hand) -> GloveState:
        position = self._glove_world_position(hand)
        return GloveState(
            position=position,
            previous_position=position.copy(),
            velocity=np.zeros(SPATIAL_DIMENSIONS),
            acceleration=np.zeros(SPATIAL_DIMENSIONS),
        )

    # ------------------------------------------------------------- properties

    @property
    def is_striking(self) -> bool:
        return self.strike_type is not None

    @property
    def is_slipping(self) -> bool:
        return self.slip_frames_remaining > 0

    @property
    def is_exhausted(self) -> bool:
        return self.exhausted_frames_remaining > 0

    @property
    def is_knocked_out(self) -> bool:
        return self.health <= EMPTY_RESOURCE

    @property
    def can_start_action(self) -> bool:
        """True when the hands are free to start a punch or a slip."""
        return not (self.is_exhausted or self.is_striking or self.is_slipping)

    @property
    def strike_spec(self) -> StrikeSpec:
        assert self.strike_type is not None, "no strike is in progress"
        return self._config.combat.strikes[self.strike_type]

    @property
    def strike_phase(self) -> StrikePhase:
        if self.strike_type is None:
            return StrikePhase.IDLE
        return self.strike_spec.phase_at(self.strike_frame)

    @property
    def strike_progress(self) -> float:
        """Fraction of the current punch already performed (0 when idle)."""
        if self.strike_type is None:
            return 0.0
        return self.strike_frame / self.strike_spec.total_frames

    @property
    def striking_glove(self) -> GloveState:
        return self.gloves[self.strike_spec.hand]

    @property
    def head_height(self) -> float:
        body = self._config.body
        return body.ducked_head_height if self.is_slipping else body.standing_head_height

    @property
    def head_center(self) -> np.ndarray:
        return np.array([*self.position, self.head_height])

    @property
    def standing_head_center(self) -> np.ndarray:
        """Where the head would be without the slip (used to credit evasions)."""
        return np.array([*self.position, self._config.body.standing_head_height])

    @property
    def torso_center(self) -> np.ndarray:
        return np.array([*self.position, self._config.body.torso_height])

    def hurtbox_center(self, zone: TargetZone) -> np.ndarray:
        return self.head_center if zone is TargetZone.HEAD else self.torso_center

    # ---------------------------------------------------------- frame: step 1

    def begin_frame(self, command: BoxerCommand, opponent_position: np.ndarray) -> None:
        self.just_exhausted = False
        self.strike_started_this_frame = False
        self._locomotion = (command.forward, command.strafe)

        if self.can_start_action:
            if command.strike is not None:
                self._start_strike(command.strike)
            elif (
                command.defense is GuardState.SLIP
                and self.slip_cooldown_frames_remaining == 0
            ):
                self._start_slip()

        self._update_guard(command.defense)
        self._update_facing(opponent_position)
        self.velocity = self._commanded_velocity(command)

    def _start_strike(self, strike_type: StrikeType) -> None:
        self.strike_type = strike_type
        self.strike_frame = 0
        self.strike_connected = False
        self.strike_evaded = False
        self._strike_start_offset = self._glove_local[self.strike_spec.hand].copy()
        # Paying may empty the tank, which cancels the punch (see _exhaust).
        self._spend_stamina(self.strike_spec.stamina_cost)
        self.strike_started_this_frame = self.is_striking

    def _start_slip(self) -> None:
        self.slip_frames_remaining = self._config.defense.slip_duration_frames
        self._spend_stamina(self._config.stamina.slip_cost)

    def _update_guard(self, requested: GuardState) -> None:
        if self.is_slipping:
            new_guard = GuardState.SLIP
        elif self.is_exhausted or self.is_striking:
            new_guard = GuardState.NONE
        elif requested in HELD_GUARDS:
            new_guard = requested
        else:
            new_guard = GuardState.NONE
        self.guard_held_frames = (
            self.guard_held_frames + 1 if new_guard is self.guard else 0
        )
        self.guard = new_guard

    def _update_facing(self, opponent_position: np.ndarray) -> None:
        # A boxer commits to the direction of a punch; otherwise it keeps
        # turning to face its opponent at a limited rate.
        if self.is_striking:
            return
        to_opponent = opponent_position - self.position
        if not to_opponent.any():
            return
        target_facing = math.atan2(to_opponent[1], to_opponent[0])
        max_step = self._config.body.turn_rate * self._config.timing.frame_seconds
        self.facing = rotate_towards(self.facing, target_facing, max_step)

    def _commanded_velocity(self, command: BoxerCommand) -> np.ndarray:
        body = self._config.body
        forward_speed = body.forward_speed if command.forward > 0 else body.backward_speed
        local_velocity = np.array(
            [command.forward * forward_speed, command.strafe * body.strafe_speed]
        )
        if command.forward != 0 and command.strafe != 0:
            local_velocity /= math.sqrt(2.0)
        return local_to_world(local_velocity, self.facing) * self._speed_multiplier()

    def _speed_multiplier(self) -> float:
        body = self._config.body
        if self.is_exhausted:
            return body.exhausted_speed_multiplier
        if self.is_slipping:
            return body.slipping_speed_multiplier
        if self.is_striking:
            return body.striking_speed_multiplier
        if self.guard in HELD_GUARDS:
            return body.guarding_speed_multiplier
        return 1.0

    # ---------------------------------------------------------- frame: step 2

    def update_gloves(self) -> None:
        frame_seconds = self._config.timing.frame_seconds
        for hand in Hand:
            self._glove_local[hand] = self._next_glove_local(hand)
            glove = self.gloves[hand]
            new_position = self._glove_world_position(hand)
            new_velocity = (new_position - glove.position) / frame_seconds
            glove.acceleration = (new_velocity - glove.velocity) / frame_seconds
            glove.previous_position = glove.position
            glove.position = new_position
            glove.velocity = new_velocity

    def _next_glove_local(self, hand: Hand) -> np.ndarray:
        if self.is_striking and self.strike_spec.hand is hand:
            return self._strike_glove_local()
        target = self._resting_offset().to_local(hand.side_sign)
        current = self._glove_local[hand]
        return current + (target - current) * self._config.body.glove_blend_per_frame

    def _strike_glove_local(self) -> np.ndarray:
        """Glove offset at the end of the current frame of the punch."""
        spec = self.strike_spec
        side_sign = spec.hand.side_sign
        windup = spec.windup_offset.to_local(side_sign)
        extended = spec.extended_offset.to_local(side_sign)
        rest = self._config.body.neutral_glove_offset.to_local(side_sign)

        frames_done = self.strike_frame + 1
        if frames_done <= spec.windup_frames:
            start, end, fraction = (
                self._strike_start_offset,
                windup,
                frames_done / spec.windup_frames,
            )
        elif frames_done <= spec.windup_frames + spec.active_frames:
            start, end, fraction = (
                windup,
                extended,
                (frames_done - spec.windup_frames) / spec.active_frames,
            )
        else:
            recovery_frames_done = frames_done - spec.windup_frames - spec.active_frames
            start, end, fraction = (
                extended,
                rest,
                recovery_frames_done / spec.recovery_frames,
            )
        return start + (end - start) * fraction

    def _resting_offset(self) -> GloveOffset:
        body = self._config.body
        if self.is_exhausted:
            return body.exhausted_glove_offset
        if self.is_slipping:
            return body.slip_glove_offset
        if self.guard is GuardState.HIGH:
            return body.high_guard_glove_offset
        if self.guard is GuardState.LOW:
            return body.low_guard_glove_offset
        return body.neutral_glove_offset

    def _glove_world_position(self, hand: Hand) -> np.ndarray:
        forward, left, height = self._glove_local[hand]
        planar = (
            self.position
            + forward * heading_vector(self.facing)
            + left * left_vector(self.facing)
        )
        return np.array([*planar, height])

    # ------------------------------------------------- called by the match

    def take_damage(self, damage: float) -> None:
        assert damage >= 0.0, "damage must be >= 0"
        self.health = max(EMPTY_RESOURCE, self.health - damage)

    def pay_block_cost(self) -> None:
        self._spend_stamina(self._config.stamina.block_cost)

    # ---------------------------------------------------------- frame: step 3

    def end_frame(self) -> None:
        # Settled first, while the state still describes the frame just played.
        self._settle_stamina()
        self._advance_strike()
        self._advance_slip()
        self._advance_exhaustion()

    def _advance_strike(self) -> None:
        if not self.is_striking:
            return
        self.strike_frame += 1
        if self.strike_frame < self.strike_spec.total_frames:
            return
        stamina_cost = self.strike_spec.stamina_cost
        was_missed = not self.strike_connected
        self.strike_type = None
        self.strike_frame = 0
        if was_missed:
            self._spend_stamina(
                stamina_cost * self._config.stamina.missed_strike_extra_cost_fraction
            )

    def _advance_slip(self) -> None:
        if self.slip_cooldown_frames_remaining > 0:
            self.slip_cooldown_frames_remaining -= 1
        if not self.is_slipping:
            return
        self.slip_frames_remaining -= 1
        if not self.is_slipping:
            self.slip_cooldown_frames_remaining = self._config.defense.slip_cooldown_frames

    def _advance_exhaustion(self) -> None:
        if self.is_exhausted:
            self.exhausted_frames_remaining -= 1

    def _settle_stamina(self) -> None:
        """Per-frame stamina flow. Exactly one rule applies each frame."""
        stamina = self._config.stamina
        frame_seconds = self._config.timing.frame_seconds
        forward, strafe = self._locomotion
        if self.is_exhausted:
            # Always recovers while stunned so a boxer can never get stuck at 0.
            self._regain_stamina(stamina.idle_regen_per_second * frame_seconds)
        elif self.is_striking or self.is_slipping:
            return  # Already paid up front; no recovery while committed.
        elif self.guard in HELD_GUARDS:
            self._spend_stamina(stamina.guard_cost_per_second * frame_seconds)
        elif forward == 0 and strafe == 0:
            self._regain_stamina(stamina.idle_regen_per_second * frame_seconds)
        elif forward < 0:
            self._regain_stamina(stamina.retreat_regen_per_second * frame_seconds)
        else:
            self._spend_stamina(stamina.move_cost_per_second * frame_seconds)

    # ---------------------------------------------------------------- stamina

    def _regain_stamina(self, amount: float) -> None:
        self.stamina = min(FULL_RESOURCE, self.stamina + amount)

    def _spend_stamina(self, amount: float) -> None:
        assert amount >= 0.0, "stamina cost must be >= 0"
        if self.is_exhausted:
            return
        self.stamina = max(EMPTY_RESOURCE, self.stamina - amount)
        if self.stamina == EMPTY_RESOURCE:
            self._exhaust()

    def _exhaust(self) -> None:
        """Stamina hit zero: hands drop and whatever was in progress stops."""
        self.exhausted_frames_remaining = self._config.stamina.exhaustion_duration_frames
        self.just_exhausted = True
        self.strike_type = None
        self.strike_frame = 0
        self.slip_frames_remaining = 0
        self.guard = GuardState.NONE
        self.guard_held_frames = 0
