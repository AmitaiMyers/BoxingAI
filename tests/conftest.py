"""Shared fixtures and helpers for the test-suite."""

from __future__ import annotations

import os

# Tests never open a real window or sound device.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import pytest

from boxing_ai.boxer import IDLE_COMMAND, BoxerCommand
from boxing_ai.config import GameConfig
from boxing_ai.simulation import BoxingMatch, FrameEvents, StartPositions

ATTACKER = 0
DEFENDER = 1


@pytest.fixture
def config() -> GameConfig:
    return GameConfig()


@pytest.fixture
def match(config: GameConfig) -> BoxingMatch:
    """Two boxers one metre apart in the middle of the ring, facing each other."""
    return place_boxers(BoxingMatch(config), distance=1.0)


def place_boxers(match: BoxingMatch, distance: float) -> BoxingMatch:
    match.reset(StartPositions(center=np.zeros(2), axis_angle=0.0))
    match.boxers[ATTACKER].position = np.array([-distance / 2.0, 0.0])
    match.boxers[DEFENDER].position = np.array([distance / 2.0, 0.0])
    return match


def run_frames(
    match: BoxingMatch,
    frames: int,
    attacker_command: BoxerCommand = IDLE_COMMAND,
    defender_command: BoxerCommand = IDLE_COMMAND,
) -> list[FrameEvents]:
    """Hold the same two commands for ``frames`` frames; return every event."""
    return [match.step((attacker_command, defender_command)) for _ in range(frames)]


def all_hits(events: list[FrameEvents]) -> list:
    return [hit for frame in events for hit in frame.hits]


def all_evades(events: list[FrameEvents]) -> list:
    return [evade for frame in events for evade in frame.evades]
