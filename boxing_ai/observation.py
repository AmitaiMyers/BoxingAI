"""Observation space: the agent's "sensor array".

Everything is expressed in the observing boxer's own local frame (forward /
left / up), so the policy does not have to relearn boxing for every direction
it might face. Values are scaled to roughly [-1, 1] and clipped to that range.

Layout of the observation vector (see ``ObservationBuilder.build``):

* Self (proprioception): health, stamina, guard one-hot, exhausted flag,
  punch-phase one-hot, punch progress, slip availability.
* Opponent (spatial awareness): relative 3D position and velocity, heading
  vector, health, stamina, guard one-hot, exhausted flag.
* Threat: for each opponent glove its relative 3D position, velocity and
  acceleration; plus a wind-up flag per punch type, an "active hitbox" flag
  and the punch progress.
* Ring: distances to the ropes along evenly spaced rays (like a lidar),
  distance to the nearest corner, and the fraction of the round remaining.

The guard state is fed as a one-hot vector rather than the raw 0-3 code so the
network does not read an ordering into unrelated stances.
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from boxing_ai.boxer import Boxer, GuardState
from boxing_ai.config import GameConfig
from boxing_ai.geometry import heading_vector, ring_ray_distance, world_to_local
from boxing_ai.simulation import BoxingMatch, other_index
from boxing_ai.strikes import Hand, StrikePhase, StrikeType

OBSERVATION_LIMIT = 1.0


def _one_hot(index: int, size: int) -> np.ndarray:
    encoded = np.zeros(size)
    encoded[index] = 1.0
    return encoded


class ObservationBuilder:
    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self.size = self._measure_size()
        self.space = spaces.Box(
            low=-OBSERVATION_LIMIT,
            high=OBSERVATION_LIMIT,
            shape=(self.size,),
            dtype=np.float32,
        )

    def _measure_size(self) -> int:
        return self._assemble(BoxingMatch(self._config), observer_index=0).size

    def build(self, match: BoxingMatch, observer_index: int) -> np.ndarray:
        observation = self._assemble(match, observer_index)
        assert observation.shape == (self.size,), "observation layout changed size"
        assert np.all(np.isfinite(observation)), "observation contains NaN or inf"
        return np.clip(observation, -OBSERVATION_LIMIT, OBSERVATION_LIMIT).astype(
            np.float32
        )

    def _assemble(self, match: BoxingMatch, observer_index: int) -> np.ndarray:
        observer = match.boxers[observer_index]
        opponent = match.boxers[other_index(observer_index)]
        return np.concatenate(
            [
                self._proprioception(observer),
                self._opponent_state(observer, opponent),
                self._threat(observer, opponent),
                self._ring(observer),
                [match.time_remaining_fraction],
            ]
        )

    # ------------------------------------------------------------------ parts

    def _proprioception(self, observer: Boxer) -> np.ndarray:
        slip_cooldown = self._config.defense.slip_cooldown_frames
        slip_readiness = (
            1.0 - observer.slip_cooldown_frames_remaining / slip_cooldown
            if slip_cooldown > 0
            else 1.0
        )
        return np.concatenate(
            [
                [observer.health, observer.stamina],
                _one_hot(observer.guard, len(GuardState)),
                [float(observer.is_exhausted)],
                _one_hot(observer.strike_phase, len(StrikePhase)),
                [observer.strike_progress, slip_readiness],
            ]
        )

    def _opponent_state(self, observer: Boxer, opponent: Boxer) -> np.ndarray:
        ring_scale = self._config.ring.diagonal
        height_scale = self._config.body.standing_head_height
        # Both boxers can move at full speed toward each other.
        speed_scale = 2.0 * self._config.body.max_speed

        relative_position = world_to_local(
            opponent.position - observer.position, observer.facing
        )
        relative_velocity = world_to_local(
            opponent.velocity - observer.velocity, observer.facing
        )
        relative_height = opponent.head_height - observer.head_height
        opponent_heading = world_to_local(heading_vector(opponent.facing), observer.facing)
        return np.concatenate(
            [
                relative_position / ring_scale,
                [relative_height / height_scale],
                relative_velocity / speed_scale,
                # Heads change height instantly, so vertical velocity is 0.
                [0.0],
                opponent_heading,
                [opponent.health, opponent.stamina],
                _one_hot(opponent.guard, len(GuardState)),
                [float(opponent.is_exhausted)],
            ]
        )

    def _threat(self, observer: Boxer, opponent: Boxer) -> np.ndarray:
        scales = self._config.observation
        parts = []
        for hand in Hand:
            glove = opponent.gloves[hand]
            planar_offset = world_to_local(
                glove.position[:2] - observer.position, observer.facing
            )
            height_offset = glove.position[2] - observer.head_height
            parts.append(
                np.array([*planar_offset, height_offset]) / scales.glove_distance_scale
            )
            parts.append(
                self._to_local_3d(glove.velocity, observer.facing)
                / scales.glove_speed_scale
            )
            parts.append(
                self._to_local_3d(glove.acceleration, observer.facing)
                / scales.glove_acceleration_scale
            )

        windup_flags = np.zeros(len(StrikeType))
        if opponent.strike_phase is StrikePhase.WINDUP:
            windup_flags[opponent.strike_type] = 1.0
        is_active = float(opponent.strike_phase is StrikePhase.ACTIVE)
        parts.append(windup_flags)
        parts.append(np.array([is_active, opponent.strike_progress]))
        return np.concatenate(parts)

    @staticmethod
    def _to_local_3d(vector: np.ndarray, facing: float) -> np.ndarray:
        return np.array([*world_to_local(vector[:2], facing), vector[2]])

    def _ring(self, observer: Boxer) -> np.ndarray:
        ring = self._config.ring
        ray_count = self._config.observation.ring_ray_count
        ray_distances = [
            ring_ray_distance(
                observer.position,
                heading_vector(observer.facing + ray * 2.0 * math.pi / ray_count),
                ring.half_size,
            )
            for ray in range(ray_count)
        ]
        nearest_corner_distance = float(
            np.linalg.norm(ring.half_size - np.abs(observer.position))
        )
        return np.array([*ray_distances, nearest_corner_distance]) / ring.diagonal
