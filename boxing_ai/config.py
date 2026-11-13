"""Every tunable number in the game lives here, as named, validated fields.

Units: metres, seconds, radians. Health and stamina are fractions in [0, 1].
Durations that the spec expresses in frames are stored in frames; everything
else is stored per second and converted with :class:`TimingConfig`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from boxing_ai.geometry import GloveOffset
from boxing_ai.strikes import StrikeSpec, StrikeType, default_strike_specs


def _assert_fraction(value: float, name: str) -> None:
    assert 0.0 <= value <= 1.0, f"{name} must be in [0, 1], got {value}"


def _assert_positive(value: float, name: str) -> None:
    assert value > 0.0, f"{name} must be > 0, got {value}"


@dataclass(frozen=True)
class TimingConfig:
    frames_per_second: int = 60
    # The policy (and the opponent) choose a new action every N physics frames.
    decision_interval_frames: int = 2
    round_duration_seconds: float = 60.0

    def __post_init__(self) -> None:
        _assert_positive(self.frames_per_second, "frames_per_second")
        _assert_positive(self.decision_interval_frames, "decision_interval_frames")
        _assert_positive(self.round_duration_seconds, "round_duration_seconds")

    @property
    def frame_seconds(self) -> float:
        return 1.0 / self.frames_per_second

    @property
    def decisions_per_second(self) -> float:
        return self.frames_per_second / self.decision_interval_frames

    @property
    def round_frames(self) -> int:
        return self.seconds_to_frames(self.round_duration_seconds)

    def seconds_to_frames(self, seconds: float) -> int:
        return round(seconds * self.frames_per_second)


@dataclass(frozen=True)
class RingConfig:
    # The ring is a square; this is the distance from its centre to each rope.
    half_size: float = 3.0
    # Distance from the centre at which the two boxers start a round.
    start_offset: float = 1.0
    # A round may start anywhere within this distance of the ring centre.
    start_center_jitter: float = 1.0

    def __post_init__(self) -> None:
        _assert_positive(self.half_size, "half_size")
        _assert_positive(self.start_offset, "start_offset")
        assert self.start_center_jitter >= 0.0, "start_center_jitter must be >= 0"

    @property
    def diagonal(self) -> float:
        return 2.0 * self.half_size * math.sqrt(2.0)


@dataclass(frozen=True)
class BodyConfig:
    # Collision and hurtbox sizes.
    body_radius: float = 0.25
    head_radius: float = 0.14
    torso_radius: float = 0.20
    glove_radius: float = 0.10
    standing_head_height: float = 1.60
    ducked_head_height: float = 1.20
    torso_height: float = 1.15

    # Locomotion.
    forward_speed: float = 2.5
    backward_speed: float = 2.0
    strafe_speed: float = 2.0
    turn_rate: float = 6.0
    striking_speed_multiplier: float = 0.3
    guarding_speed_multiplier: float = 0.6
    slipping_speed_multiplier: float = 0.0
    exhausted_speed_multiplier: float = 0.5

    # Where the gloves rest in each stance.
    neutral_glove_offset: GloveOffset = GloveOffset(0.25, 0.14, 1.40)
    high_guard_glove_offset: GloveOffset = GloveOffset(0.20, 0.08, 1.58)
    low_guard_glove_offset: GloveOffset = GloveOffset(0.22, 0.12, 1.15)
    slip_glove_offset: GloveOffset = GloveOffset(0.20, 0.12, 1.20)
    exhausted_glove_offset: GloveOffset = GloveOffset(0.15, 0.18, 1.00)
    # Fraction of the remaining distance a resting glove covers each frame
    # when the stance changes (keeps glove motion continuous).
    glove_blend_per_frame: float = 0.35

    def __post_init__(self) -> None:
        for name in ("body_radius", "head_radius", "torso_radius", "glove_radius"):
            _assert_positive(getattr(self, name), name)
        assert self.ducked_head_height < self.standing_head_height, (
            "a ducked head must be lower than a standing head"
        )
        for name in ("forward_speed", "backward_speed", "strafe_speed", "turn_rate"):
            _assert_positive(getattr(self, name), name)
        for name in (
            "striking_speed_multiplier",
            "guarding_speed_multiplier",
            "slipping_speed_multiplier",
            "exhausted_speed_multiplier",
            "glove_blend_per_frame",
        ):
            _assert_fraction(getattr(self, name), name)

    @property
    def max_speed(self) -> float:
        return max(self.forward_speed, self.backward_speed, self.strafe_speed)


@dataclass(frozen=True)
class DefenseConfig:
    # A matching guard removes this fraction of a punch's damage.
    guard_damage_reduction: float = 0.8
    # A slip is a committed duck: it lasts this long, then cannot be repeated
    # until the cooldown has passed. It must overlap the punch to evade it.
    slip_duration_frames: int = 18
    slip_cooldown_frames: int = 24
    # Extra damage an uppercut does to a boxer who is slipping.
    slip_uppercut_damage_multiplier: float = 1.5
    # A block only counts as "timed" if the guard went up this recently.
    timed_block_window_frames: int = 30

    def __post_init__(self) -> None:
        _assert_fraction(self.guard_damage_reduction, "guard_damage_reduction")
        _assert_positive(self.slip_duration_frames, "slip_duration_frames")
        assert self.slip_cooldown_frames >= 0, "slip_cooldown_frames must be >= 0"
        assert self.slip_uppercut_damage_multiplier >= 1.0, (
            "slip_uppercut_damage_multiplier must be >= 1"
        )
        _assert_positive(self.timed_block_window_frames, "timed_block_window_frames")


@dataclass(frozen=True)
class StaminaConfig:
    idle_regen_per_second: float = 0.12
    retreat_regen_per_second: float = 0.06
    move_cost_per_second: float = 0.03
    guard_cost_per_second: float = 0.05
    slip_cost: float = 0.05
    block_cost: float = 0.01
    # A punch that hits nothing costs this extra fraction of its stamina cost.
    missed_strike_extra_cost_fraction: float = 0.5
    # How long a boxer stays exhausted after stamina reaches zero.
    exhaustion_duration_frames: int = 120

    def __post_init__(self) -> None:
        for name in (
            "idle_regen_per_second",
            "retreat_regen_per_second",
            "move_cost_per_second",
            "guard_cost_per_second",
            "slip_cost",
            "block_cost",
            "missed_strike_extra_cost_fraction",
        ):
            assert getattr(self, name) >= 0.0, f"{name} must be >= 0"
        _assert_positive(self.exhaustion_duration_frames, "exhaustion_duration_frames")


@dataclass(frozen=True)
class CombatConfig:
    # Health removed by a clean punch with a damage multiplier of 1.0.
    base_strike_damage: float = 0.04
    strikes: Mapping[StrikeType, StrikeSpec] = field(
        default_factory=default_strike_specs
    )

    def __post_init__(self) -> None:
        _assert_positive(self.base_strike_damage, "base_strike_damage")
        assert set(self.strikes) == set(StrikeType), (
            "strikes must define exactly one spec per StrikeType"
        )


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the shaping reward. See ``rewards.py`` for the formulas."""

    # +weight * (impact speed / reference speed, capped at 1) per clean punch.
    clean_hit_reward: float = 5.0
    impact_speed_reference: float = 20.0
    # Per punch evaded by a slip or stopped by a freshly raised guard.
    timed_defense_reward: float = 3.0
    # Per second spent closer to the ring centre than the opponent.
    center_control_reward_per_second: float = 0.25
    center_control_radius: float = 2.0
    # Per health point lost (1 health point = 1% of full health).
    damage_penalty_per_health_point: float = 0.5
    # Once, every time stamina hits zero.
    exhaustion_penalty: float = 10.0
    # Per second, after going this long without throwing a punch in range.
    passivity_grace_seconds: float = 3.0
    passivity_penalty_per_second: float = 0.5
    # Per second of guarding beyond the grace period without punching back.
    guard_spam_grace_seconds: float = 2.0
    guard_spam_penalty_per_second: float = 0.5
    # A punch only counts as "engaging" if the opponent is this close.
    engagement_range: float = 1.5
    # End of round.
    knockout_reward: float = 20.0
    timeout_health_advantage_reward: float = 10.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            assert value >= 0.0, f"{name} must be >= 0, got {value}"
        _assert_positive(self.impact_speed_reference, "impact_speed_reference")


@dataclass(frozen=True)
class ObservationConfig:
    # Number of evenly spaced "lidar" rays measuring the distance to the ropes.
    ring_ray_count: int = 8
    # Divisors that bring each quantity into roughly [-1, 1].
    glove_distance_scale: float = 2.0
    glove_speed_scale: float = 20.0
    glove_acceleration_scale: float = 1000.0

    def __post_init__(self) -> None:
        _assert_positive(self.ring_ray_count, "ring_ray_count")
        _assert_positive(self.glove_distance_scale, "glove_distance_scale")
        _assert_positive(self.glove_speed_scale, "glove_speed_scale")
        _assert_positive(self.glove_acceleration_scale, "glove_acceleration_scale")


@dataclass(frozen=True)
class TrainingConfig:
    """PPO hyper-parameters (Stable-Baselines3 names)."""

    learning_rate: float = 3e-4
    rollout_steps_per_env: int = 2048
    batch_size: int = 256
    epochs_per_update: int = 10
    discount: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.01
    hidden_layer_sizes: tuple[int, ...] = (128, 128)
    parallel_envs: int = 8
    # Headless training saves a checkpoint every time this many steps pass.
    checkpoint_interval_steps: int = 100_000

    def __post_init__(self) -> None:
        _assert_positive(self.learning_rate, "learning_rate")
        _assert_positive(self.rollout_steps_per_env, "rollout_steps_per_env")
        _assert_positive(self.batch_size, "batch_size")
        _assert_positive(self.epochs_per_update, "epochs_per_update")
        _assert_fraction(self.discount, "discount")
        _assert_fraction(self.gae_lambda, "gae_lambda")
        _assert_positive(self.clip_range, "clip_range")
        assert self.entropy_coefficient >= 0.0, "entropy_coefficient must be >= 0"
        assert self.hidden_layer_sizes, "hidden_layer_sizes must not be empty"
        _assert_positive(self.parallel_envs, "parallel_envs")
        _assert_positive(self.checkpoint_interval_steps, "checkpoint_interval_steps")


@dataclass(frozen=True)
class GameConfig:
    timing: TimingConfig = TimingConfig()
    ring: RingConfig = RingConfig()
    body: BodyConfig = BodyConfig()
    defense: DefenseConfig = DefenseConfig()
    stamina: StaminaConfig = StaminaConfig()
    combat: CombatConfig = field(default_factory=CombatConfig)
    rewards: RewardConfig = RewardConfig()
    observation: ObservationConfig = ObservationConfig()
    training: TrainingConfig = TrainingConfig()

    def __post_init__(self) -> None:
        usable_half_size = self.ring.half_size - self.body.body_radius
        assert self.ring.start_offset + self.ring.start_center_jitter <= usable_half_size, (
            "boxers would start outside the ropes"
        )
        assert self.ring.start_offset >= self.body.body_radius, (
            "boxers would start overlapping"
        )
