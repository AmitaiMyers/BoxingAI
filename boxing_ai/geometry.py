"""Pure geometry helpers shared by the simulation, observations and renderer.

Conventions used everywhere in the project:

* The ring floor is the world XY plane, centred on the origin. Z is height.
* A boxer's *facing* is an angle in radians, measured counter-clockwise from
  the world +X axis.
* A boxer's *local frame* has +forward along its facing and +left 90 degrees
  counter-clockwise from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PLANAR_DIMENSIONS = 2
SPATIAL_DIMENSIONS = 3


@dataclass(frozen=True)
class GloveOffset:
    """Where a glove sits relative to its owner, in the owner's local frame.

    ``outward`` is the distance away from the body's centre line toward the
    side of the hand that owns the glove, so one offset describes both hands.
    """

    forward: float
    outward: float
    height: float

    def __post_init__(self) -> None:
        assert self.outward >= 0.0, "outward is a magnitude and must be >= 0"
        assert self.height > 0.0, "a glove must be above the ring floor"

    def to_local(self, side_sign: float) -> np.ndarray:
        """Return ``[forward, left, height]`` for the hand on ``side_sign``."""
        assert side_sign in (-1.0, 1.0), "side_sign must be -1.0 or 1.0"
        return np.array([self.forward, self.outward * side_sign, self.height])


def heading_vector(facing: float) -> np.ndarray:
    """Unit vector pointing along ``facing`` on the ring floor."""
    return np.array([math.cos(facing), math.sin(facing)])


def left_vector(facing: float) -> np.ndarray:
    """Unit vector pointing to the left of ``facing`` on the ring floor."""
    return np.array([-math.sin(facing), math.cos(facing)])


def world_to_local(vector: np.ndarray, facing: float) -> np.ndarray:
    """Express a planar world vector as ``[forward, left]`` for ``facing``."""
    assert vector.shape == (PLANAR_DIMENSIONS,), "expected a planar vector"
    return np.array(
        [vector @ heading_vector(facing), vector @ left_vector(facing)]
    )


def local_to_world(local: np.ndarray, facing: float) -> np.ndarray:
    """Convert a planar ``[forward, left]`` vector back to world axes."""
    assert local.shape == (PLANAR_DIMENSIONS,), "expected a planar vector"
    return local[0] * heading_vector(facing) + local[1] * left_vector(facing)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotate_towards(current: float, target: float, max_step: float) -> float:
    """Turn ``current`` toward ``target`` by at most ``max_step`` radians."""
    assert max_step >= 0.0, "max_step must be >= 0"
    difference = wrap_angle(target - current)
    step = max(-max_step, min(max_step, difference))
    return wrap_angle(current + step)


def segment_point_distance(
    segment_start: np.ndarray, segment_end: np.ndarray, point: np.ndarray
) -> float:
    """Shortest distance between ``point`` and a line segment (any dimension).

    Used for swept hit detection: a glove that moves from ``segment_start`` to
    ``segment_end`` during one frame hits everything close to that path, so a
    fast punch can never tunnel through its target between two frames.
    """
    segment = segment_end - segment_start
    squared_length = float(segment @ segment)
    if squared_length == 0.0:
        return float(np.linalg.norm(point - segment_start))
    fraction = float((point - segment_start) @ segment) / squared_length
    closest = segment_start + min(1.0, max(0.0, fraction)) * segment
    return float(np.linalg.norm(point - closest))


def ring_ray_distance(
    position: np.ndarray, direction: np.ndarray, ring_half_size: float
) -> float:
    """Distance from ``position`` to the ropes along the unit ``direction``.

    The ring is the axis-aligned square ``[-half, half]^2``; ``position`` must
    be inside it.
    """
    assert np.all(np.abs(position) <= ring_half_size), "position outside ring"
    distances = []
    for axis in range(PLANAR_DIMENSIONS):
        component = float(direction[axis])
        if component > 0.0:
            distances.append((ring_half_size - position[axis]) / component)
        elif component < 0.0:
            distances.append((-ring_half_size - position[axis]) / component)
    assert distances, "direction must be a non-zero vector"
    return min(distances)
