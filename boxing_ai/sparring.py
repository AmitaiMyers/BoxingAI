"""Real-time game loop: live sparring (with learning) and watch mode (without).

Sparring flow, one round at a time:

1. The round is played at real-time speed. Every decision the policy makes is
   recorded together with its reward - no learning happens yet, so the game
   never stutters.
2. On the bell (or a knockout) the recorded experience is turned into one PPO
   update while the "corner break" message is on screen.
3. The updated weights are saved as a new checkpoint and the next round starts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import Logger

from boxing_ai.checkpoints import (
    list_checkpoints,
    load_latest_or_create,
    load_model,
    save_checkpoint,
)
from boxing_ai.config import GameConfig
from boxing_ai.controllers import BOT_PROFILES, BoxerController, ScriptedBot
from boxing_ai.env import RENDER_MODE_HUMAN, BoxingEnv, RoundOutcome
from boxing_ai.rendering import KeyboardController

HUMAN_PLAYER = "human"
PLAYER_CHOICES = (HUMAN_PLAYER, *BOT_PROFILES)
# Length of the pause between rounds, while the result is on screen.
CORNER_BREAK_SECONDS = 3.0
SINGLE_ENV = 1

ActionSelector = Callable[[np.ndarray], np.ndarray]
TransitionListener = Callable[[np.ndarray, np.ndarray, float, bool, bool, np.ndarray], None]


@dataclass(frozen=True)
class RoundSummary:
    outcome: RoundOutcome
    decisions: int
    total_reward: float
    quit_requested: bool


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    episode_start: bool
    value: torch.Tensor
    log_probability: torch.Tensor


def build_player(name: str, config: GameConfig, seed: int) -> BoxerController:
    """The controller in the opposite corner: a human or a scripted stand-in."""
    assert name in PLAYER_CHOICES, f"unknown player '{name}', choose from {PLAYER_CHOICES}"
    if name == HUMAN_PLAYER:
        return KeyboardController()
    return ScriptedBot(config, BOT_PROFILES[name], seed)


def build_env(config: GameConfig, player: str, show_window: bool, seed: int) -> BoxingEnv:
    assert show_window or player != HUMAN_PLAYER, (
        "a human player needs the game window; remove --headless"
    )
    render_mode = RENDER_MODE_HUMAN if show_window else None
    return BoxingEnv(build_player(player, config, seed), config, render_mode=render_mode)


def play_round(
    env: BoxingEnv, select_action: ActionSelector, on_transition: TransitionListener
) -> RoundSummary:
    """Play one round, in real time if the env has a window."""
    is_live = env.render_mode == RENDER_MODE_HUMAN
    observation, info = env.reset()
    decisions = 0
    total_reward = 0.0
    is_finished = False
    quit_requested = False

    while not (is_finished or quit_requested):
        if is_live:
            # Drawing first also guarantees the window exists before the
            # keyboard is read inside env.step().
            env.render()
            quit_requested = env.renderer.quit_requested()
            env.renderer.wait_for_next_frame(env.config.timing.decisions_per_second)

        action = select_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        on_transition(observation, action, reward, terminated, truncated, next_observation)

        observation = next_observation
        decisions += 1
        total_reward += reward
        is_finished = terminated or truncated

    if is_live:
        env.render()
    return RoundSummary(
        outcome=RoundOutcome(info["outcome"]),
        decisions=decisions,
        total_reward=total_reward,
        quit_requested=quit_requested,
    )


def ignore_transition(*_transition) -> None:
    """Transition listener for modes that do not learn."""


class OnlineLearner:
    """Collects experience during rounds and applies PPO updates between them.

    Stable-Baselines3's own ``learn()`` trains whenever its fixed-size buffer
    fills up, which would freeze the game mid-round. This class instead builds
    a rollout buffer exactly as long as the experience gathered, and calls the
    standard PPO ``train()`` on it when the game loop says it is time.
    """

    def __init__(self, model: PPO) -> None:
        self._model = model
        # train() reports its losses to a logger; we do not need them stored.
        self._model.set_logger(Logger(folder=None, output_formats=[]))
        self._transitions: list[Transition] = []
        self._is_episode_start = True
        self._pending_value: torch.Tensor | None = None
        self._pending_log_probability: torch.Tensor | None = None

    @property
    def pending_decisions(self) -> int:
        return len(self._transitions)

    @property
    def is_ready_to_update(self) -> bool:
        """PPO needs at least one full mini-batch to make a sensible update."""
        return self.pending_decisions >= self._model.batch_size

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor, _ = self._model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            actions, values, log_probabilities = self._model.policy(observation_tensor)
        self._pending_value = values
        self._pending_log_probability = log_probabilities
        return actions[0].cpu().numpy()

    def record(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_observation: np.ndarray,
    ) -> None:
        assert self._pending_value is not None, "record() called before select_action()"
        if truncated:
            # The bell is not a real ending: the fight would have gone on, so
            # the value of what would have come next is still owed.
            reward += self._model.gamma * self._estimate_value(next_observation)
        self._transitions.append(
            Transition(
                observation=observation,
                action=action,
                reward=reward,
                episode_start=self._is_episode_start,
                value=self._pending_value,
                log_probability=self._pending_log_probability,
            )
        )
        self._is_episode_start = terminated or truncated
        self._pending_value = None
        self._pending_log_probability = None

    def _estimate_value(self, observation: np.ndarray) -> float:
        observation_tensor, _ = self._model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            return float(self._model.policy.predict_values(observation_tensor)[0, 0])

    def update(self) -> int:
        """Run one PPO update on everything recorded; return decisions used."""
        assert self.is_ready_to_update, "not enough experience for a PPO update"
        assert self._is_episode_start, "update() must be called between rounds"
        model = self._model
        decisions = self.pending_decisions

        buffer = RolloutBuffer(
            decisions,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=SINGLE_ENV,
        )
        for transition in self._transitions:
            buffer.add(
                transition.observation[np.newaxis],
                transition.action[np.newaxis],
                np.array([transition.reward]),
                np.array([transition.episode_start]),
                transition.value,
                transition.log_probability,
            )
        # Every recorded round is complete, so nothing follows the last step.
        buffer.compute_returns_and_advantage(
            last_values=torch.zeros(SINGLE_ENV), dones=np.array([True])
        )

        model.rollout_buffer = buffer
        model.train()
        model.num_timesteps += decisions
        self._transitions = []
        return decisions


def run_sparring(
    config: GameConfig,
    checkpoint_directory: Path,
    rounds: int,
    player: str,
    show_window: bool,
    seed: int,
) -> list[RoundSummary]:
    """Spar for ``rounds`` rounds, learning and saving a checkpoint after each."""
    assert rounds > 0, "rounds must be > 0"
    env = build_env(config, player, show_window, seed)
    env.reset(seed=seed)
    model = load_latest_or_create(checkpoint_directory, env, config, seed)
    learner = OnlineLearner(model)
    summaries: list[RoundSummary] = []

    for round_number in range(1, rounds + 1):
        env.status_text = f"Round {round_number}/{rounds}   trained {model.num_timesteps}"
        summary = play_round(env, learner.select_action, learner.record)
        summaries.append(summary)
        if summary.quit_requested:
            break

        lines = [f"Round {round_number}: {_describe(summary.outcome)}"]
        if learner.is_ready_to_update:
            _show_message(env, [*lines, "Learning..."])
            trained = learner.update()
            saved = save_checkpoint(model, checkpoint_directory)
            lines.append(f"Learned from {trained} decisions")
            print(f"round {round_number}: {summary.outcome.value}, "
                  f"reward {summary.total_reward:+.1f}, saved {saved}")
        else:
            lines.append("Round too short to learn from yet - keep going")
            print(f"round {round_number}: {summary.outcome.value}, "
                  f"reward {summary.total_reward:+.1f}, no update "
                  f"({learner.pending_decisions}/{model.batch_size} decisions)")

        if _corner_break(env, lines):
            break

    env.close()
    return summaries


def run_watch(
    config: GameConfig,
    checkpoint: Path | None,
    checkpoint_directory: Path,
    rounds: int,
    player: str,
    show_window: bool,
    seed: int,
) -> list[RoundSummary]:
    """Watch (or fight) a saved boxer. Nothing is learned and nothing is saved."""
    assert rounds > 0, "rounds must be > 0"
    if checkpoint is None:
        checkpoints = list_checkpoints(checkpoint_directory)
        assert checkpoints, f"no checkpoints found in {checkpoint_directory}"
        checkpoint = checkpoints[-1]

    env = build_env(config, player, show_window, seed)
    env.reset(seed=seed)
    model = load_model(checkpoint, env)
    summaries: list[RoundSummary] = []

    def select_action(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=False)
        return action

    for round_number in range(1, rounds + 1):
        env.status_text = f"Watching {checkpoint.stem}   round {round_number}/{rounds}"
        summary = play_round(env, select_action, ignore_transition)
        summaries.append(summary)
        print(f"round {round_number}: {summary.outcome.value}, "
              f"reward {summary.total_reward:+.1f}")
        if summary.quit_requested:
            break
        if _corner_break(env, [f"Round {round_number}: {_describe(summary.outcome)}"]):
            break

    env.close()
    return summaries


def with_round_duration(config: GameConfig, seconds: float) -> GameConfig:
    """Copy of ``config`` with a different round length."""
    timing = dataclasses.replace(config.timing, round_duration_seconds=seconds)
    return dataclasses.replace(config, timing=timing)


# ------------------------------------------------------------------ helpers


def _describe(outcome: RoundOutcome) -> str:
    return outcome.value.replace("_", " ").upper()


def _show_message(env: BoxingEnv, lines: list[str]) -> None:
    if env.render_mode == RENDER_MODE_HUMAN:
        env.render()
        env.renderer.show_message(lines)


def _corner_break(env: BoxingEnv, lines: list[str]) -> bool:
    """Show the round result for a few seconds; True if the user quits."""
    if env.render_mode != RENDER_MODE_HUMAN:
        return False
    _show_message(env, lines)
    frames_per_second = env.config.timing.decisions_per_second
    for _ in range(round(CORNER_BREAK_SECONDS * frames_per_second)):
        if env.renderer.quit_requested():
            return True
        env.renderer.wait_for_next_frame(frames_per_second)
    return False
