"""Gymnasium environment: the learning boxer versus any ``BoxerController``.

Boxer 0 is the learning agent; boxer 1 is the opponent (human, scripted bot,
...). One environment step is one *decision*: the chosen action is held for
``decision_interval_frames`` physics frames and the rewards of those frames
are summed.

Episode end:
* ``terminated`` - a boxer was knocked out.
* ``truncated``  - the round timer ran out (the fight itself was not decided,
  so value bootstrapping is still correct).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import gymnasium as gym
import numpy as np

from boxing_ai.actions import build_action_space, decode_action
from boxing_ai.config import GameConfig
from boxing_ai.controllers import BoxerController
from boxing_ai.observation import ObservationBuilder
from boxing_ai.rendering import HudInfo, Renderer
from boxing_ai.rewards import RewardBreakdown, RewardCalculator
from boxing_ai.simulation import BoxingMatch, StartPositions

AGENT_INDEX = 0
OPPONENT_INDEX = 1

RENDER_MODE_HUMAN = "human"
RENDER_MODE_RGB_ARRAY = "rgb_array"


class RoundOutcome(str, Enum):
    ONGOING = "ongoing"
    AGENT_KNOCKOUT_WIN = "agent_knockout_win"
    AGENT_KNOCKOUT_LOSS = "agent_knockout_loss"
    DOUBLE_KNOCKOUT = "double_knockout"
    TIME_UP = "time_up"


class BoxingEnv(gym.Env):
    metadata = {"render_modes": [RENDER_MODE_HUMAN, RENDER_MODE_RGB_ARRAY]}

    def __init__(
        self,
        opponent: BoxerController,
        config: GameConfig,
        render_mode: str | None = None,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"unsupported render_mode: {render_mode}"
        )
        self.config = config
        self.render_mode = render_mode
        self.metadata = {
            **self.metadata,
            "render_fps": round(config.timing.decisions_per_second),
        }
        # Free text shown in the HUD, e.g. "Round 3" (set by the game loop).
        self.status_text = ""

        self._opponent = opponent
        self._match = BoxingMatch(config)
        self._observations = ObservationBuilder(config)
        self._rewards = RewardCalculator(config, AGENT_INDEX)
        self._renderer: Renderer | None = None
        self._episode_reward = 0.0
        self._last_reward = 0.0
        self._needs_reset = True

        self.observation_space = self._observations.space
        self.action_space = build_action_space()

    @property
    def match(self) -> BoxingMatch:
        return self._match

    @property
    def renderer(self) -> Renderer:
        """The Pygame renderer (created on first use)."""
        assert self.render_mode is not None, "this env was created without rendering"
        if self._renderer is None:
            self._renderer = Renderer(
                self.config, show_window=self.render_mode == RENDER_MODE_HUMAN
            )
        return self._renderer

    # ------------------------------------------------------------------ reset

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._match.reset(self._sample_start_positions())
        self._opponent.reset()
        self._rewards.reset()
        self._episode_reward = 0.0
        self._last_reward = 0.0
        self._needs_reset = False
        return self._observe(), self._info(RewardBreakdown())

    def _sample_start_positions(self) -> StartPositions:
        jitter = self.config.ring.start_center_jitter
        return StartPositions(
            center=self.np_random.uniform(-jitter, jitter, size=2),
            axis_angle=float(self.np_random.uniform(-math.pi, math.pi)),
        )

    # ------------------------------------------------------------------- step

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert not self._needs_reset, "call reset() before step()"
        action = np.asarray(action)
        assert self.action_space.contains(action), f"invalid action: {action}"

        agent_command = decode_action(action)
        opponent_command = self._opponent.decide(self._match, OPPONENT_INDEX)

        reward = RewardBreakdown()
        for _ in range(self.config.timing.decision_interval_frames):
            events = self._match.step((agent_command, opponent_command))
            reward = reward + self._rewards.frame_reward(self._match, events)
            if self._match.is_over:
                break

        terminated = bool(self._match.knocked_out)
        truncated = self._match.is_time_up and not terminated
        self._needs_reset = terminated or truncated
        self._last_reward = reward.total
        self._episode_reward += reward.total
        return self._observe(), reward.total, terminated, truncated, self._info(reward)

    def _observe(self) -> np.ndarray:
        return self._observations.build(self._match, AGENT_INDEX)

    def _info(self, reward: RewardBreakdown) -> dict[str, Any]:
        return {
            "reward_terms": reward.as_dict(),
            "agent_health": self._match.boxers[AGENT_INDEX].health,
            "opponent_health": self._match.boxers[OPPONENT_INDEX].health,
            "outcome": self._outcome().value,
        }

    def _outcome(self) -> RoundOutcome:
        knocked_out = self._match.knocked_out
        if len(knocked_out) == len(self._match.boxers):
            return RoundOutcome.DOUBLE_KNOCKOUT
        if OPPONENT_INDEX in knocked_out:
            return RoundOutcome.AGENT_KNOCKOUT_WIN
        if AGENT_INDEX in knocked_out:
            return RoundOutcome.AGENT_KNOCKOUT_LOSS
        if self._match.is_time_up:
            return RoundOutcome.TIME_UP
        return RoundOutcome.ONGOING

    # ----------------------------------------------------------------- render

    def render(self) -> np.ndarray | None:
        self.renderer.draw(
            self._match,
            HudInfo(
                status_text=self.status_text,
                episode_reward=self._episode_reward,
                last_reward=self._last_reward,
            ),
        )
        if self.render_mode == RENDER_MODE_RGB_ARRAY:
            return self.renderer.frame_array()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
