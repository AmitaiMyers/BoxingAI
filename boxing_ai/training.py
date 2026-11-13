"""Headless "gym grind": fast PPO training against the scripted bot.

This is the optional speed-up between live sparring sessions. It reads and
writes the same checkpoint directory as the sparring loop, so both kinds of
training build on each other.
"""

from __future__ import annotations

import itertools
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env

from boxing_ai.checkpoints import load_latest_or_create, save_checkpoint
from boxing_ai.config import GameConfig
from boxing_ai.controllers import BOT_PROFILES, ScriptedBot
from boxing_ai.env import BoxingEnv


def train_against_bot(
    config: GameConfig,
    checkpoint_directory: Path,
    total_timesteps: int,
    difficulty: str,
    seed: int,
) -> Path:
    """Train for ``total_timesteps`` more decisions; return the last checkpoint."""
    assert total_timesteps > 0, "total_timesteps must be > 0"
    assert difficulty in BOT_PROFILES, (
        f"unknown difficulty '{difficulty}', choose from {sorted(BOT_PROFILES)}"
    )

    # Every parallel ring gets a bot with its own random stream.
    bot_seeds = itertools.count(seed)

    def make_env() -> BoxingEnv:
        bot = ScriptedBot(config, BOT_PROFILES[difficulty], seed=next(bot_seeds))
        return BoxingEnv(bot, config)

    vec_env = make_vec_env(make_env, n_envs=config.training.parallel_envs, seed=seed)
    model = load_latest_or_create(checkpoint_directory, vec_env, config, seed)
    model.verbose = 1

    target_timesteps = model.num_timesteps + total_timesteps
    last_checkpoint = save_checkpoint(model, checkpoint_directory)
    while model.num_timesteps < target_timesteps:
        chunk = min(
            config.training.checkpoint_interval_steps,
            target_timesteps - model.num_timesteps,
        )
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        last_checkpoint = save_checkpoint(model, checkpoint_directory)
        print(f"saved {last_checkpoint}")

    vec_env.close()
    return last_checkpoint
