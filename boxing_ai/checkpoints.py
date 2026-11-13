"""Creating, saving and loading the boxer's brain (PPO ``.zip`` checkpoints).

Checkpoints are named after the number of decisions the boxer has trained on,
e.g. ``boxer_steps_0000250000.zip``, so sorting the file names sorts them from
novice to veteran.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from boxing_ai.config import GameConfig

CHECKPOINT_PREFIX = "boxer_steps_"
CHECKPOINT_SUFFIX = ".zip"
STEP_DIGITS = 10
POLICY_TYPE = "MlpPolicy"
# The network is small; a GPU would spend longer copying data than computing.
DEVICE = "cpu"


def create_model(env: gym.Env | VecEnv, config: GameConfig, seed: int) -> PPO:
    """A brand-new boxer with random weights and zero fighting knowledge."""
    training = config.training
    return PPO(
        POLICY_TYPE,
        env,
        learning_rate=training.learning_rate,
        n_steps=training.rollout_steps_per_env,
        batch_size=training.batch_size,
        n_epochs=training.epochs_per_update,
        gamma=training.discount,
        gae_lambda=training.gae_lambda,
        clip_range=training.clip_range,
        ent_coef=training.entropy_coefficient,
        policy_kwargs={"net_arch": list(training.hidden_layer_sizes)},
        seed=seed,
        device=DEVICE,
    )


def checkpoint_path(directory: Path, timesteps: int) -> Path:
    assert timesteps >= 0, "timesteps must be >= 0"
    return directory / f"{CHECKPOINT_PREFIX}{timesteps:0{STEP_DIGITS}d}{CHECKPOINT_SUFFIX}"


def save_checkpoint(model: PPO, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(directory, model.num_timesteps)
    model.save(path)
    return path


def list_checkpoints(directory: Path) -> list[Path]:
    """All checkpoints in ``directory``, least trained first."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{CHECKPOINT_PREFIX}*{CHECKPOINT_SUFFIX}"))


def load_model(path: Path, env: gym.Env | VecEnv) -> PPO:
    assert path.is_file(), f"checkpoint not found: {path}"
    return PPO.load(path, env=env, device=DEVICE)


def load_latest_or_create(
    directory: Path, env: gym.Env | VecEnv, config: GameConfig, seed: int
) -> PPO:
    """Continue the boxer's career, or start one if it has no checkpoints yet."""
    checkpoints = list_checkpoints(directory)
    if checkpoints:
        return load_model(checkpoints[-1], env)
    return create_model(env, config, seed)
