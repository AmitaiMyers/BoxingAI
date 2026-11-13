"""Strike definitions: the frame data and trajectory of every punch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Mapping

from boxing_ai.geometry import GloveOffset


class StrikeType(IntEnum):
    JAB = 0
    CROSS = 1
    HOOK = 2
    UPPERCUT = 3


class StrikePhase(IntEnum):
    """Where a boxer is inside a punch. IDLE means no punch is in progress."""

    IDLE = 0
    WINDUP = 1
    ACTIVE = 2
    RECOVERY = 3


class TargetZone(Enum):
    HEAD = "head"
    BODY = "body"


class Hand(Enum):
    """LEAD is the left hand, REAR the right hand (orthodox stance)."""

    LEAD = "lead"
    REAR = "rear"

    @property
    def side_sign(self) -> float:
        """+1.0 for the left side of the body, -1.0 for the right side."""
        return 1.0 if self is Hand.LEAD else -1.0


@dataclass(frozen=True)
class StrikeSpec:
    """Everything that makes one punch different from another.

    The glove travels: rest -> ``windup_offset`` during the wind-up frames,
    ``windup_offset`` -> ``extended_offset`` during the active frames (the only
    frames in which it can hit), then back to rest during the recovery frames.
    """

    windup_frames: int
    active_frames: int
    recovery_frames: int
    damage_multiplier: float
    stamina_cost: float
    target_zone: TargetZone
    hand: Hand
    windup_offset: GloveOffset
    extended_offset: GloveOffset

    def __post_init__(self) -> None:
        assert self.windup_frames > 0, "windup_frames must be > 0"
        assert self.active_frames > 0, "active_frames must be > 0"
        assert self.recovery_frames > 0, "recovery_frames must be > 0"
        assert self.damage_multiplier > 0.0, "damage_multiplier must be > 0"
        assert 0.0 < self.stamina_cost <= 1.0, "stamina_cost must be in (0, 1]"

    @property
    def total_frames(self) -> int:
        return self.windup_frames + self.active_frames + self.recovery_frames

    def phase_at(self, frame: int) -> StrikePhase:
        """Phase of the punch on its ``frame``-th frame (0-based)."""
        assert 0 <= frame < self.total_frames, "frame is outside the strike"
        if frame < self.windup_frames:
            return StrikePhase.WINDUP
        if frame < self.windup_frames + self.active_frames:
            return StrikePhase.ACTIVE
        return StrikePhase.RECOVERY


def default_strike_specs() -> Mapping[StrikeType, StrikeSpec]:
    """Frame data at 60 frames per second.

    * Jab: fast, cheap, weak, long. The range finder.
    * Cross: slower straight power punch to the head.
    * Hook: swings in from the side into the body; beaten by Low Guard.
    * Uppercut: slow and short, but rises through the space a ducking head
      occupies, so it is the punch that punishes a slip.
    """
    return {
        StrikeType.JAB: StrikeSpec(
            windup_frames=6,
            active_frames=3,
            recovery_frames=8,
            damage_multiplier=1.0,
            stamina_cost=0.04,
            target_zone=TargetZone.HEAD,
            hand=Hand.LEAD,
            windup_offset=GloveOffset(forward=0.18, outward=0.14, height=1.50),
            extended_offset=GloveOffset(forward=1.00, outward=0.0, height=1.60),
        ),
        StrikeType.CROSS: StrikeSpec(
            windup_frames=10,
            active_frames=3,
            recovery_frames=14,
            damage_multiplier=2.0,
            stamina_cost=0.07,
            target_zone=TargetZone.HEAD,
            hand=Hand.REAR,
            windup_offset=GloveOffset(forward=0.10, outward=0.16, height=1.50),
            extended_offset=GloveOffset(forward=1.05, outward=0.0, height=1.60),
        ),
        StrikeType.HOOK: StrikeSpec(
            windup_frames=12,
            active_frames=4,
            recovery_frames=16,
            damage_multiplier=2.5,
            stamina_cost=0.09,
            target_zone=TargetZone.BODY,
            hand=Hand.LEAD,
            windup_offset=GloveOffset(forward=0.15, outward=0.40, height=1.20),
            extended_offset=GloveOffset(forward=0.80, outward=0.0, height=1.15),
        ),
        StrikeType.UPPERCUT: StrikeSpec(
            windup_frames=14,
            active_frames=4,
            recovery_frames=20,
            damage_multiplier=3.25,
            stamina_cost=0.11,
            target_zone=TargetZone.HEAD,
            hand=Hand.REAR,
            windup_offset=GloveOffset(forward=0.30, outward=0.10, height=0.95),
            extended_offset=GloveOffset(forward=0.75, outward=0.0, height=1.75),
        ),
    }
