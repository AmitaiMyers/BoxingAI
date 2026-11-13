"""Gymnasium API, observation/action spaces, checkpoints and the game loops."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from boxing_ai.actions import (
    DEFENSE_CHOICES,
    LOCOMOTION_CHOICES,
    OFFENSE_CHOICES,
    build_action_space,
    decode_action,
)
from boxing_ai.boxer import BoxerCommand, GuardState
from boxing_ai.checkpoints import (
    create_model,
    list_checkpoints,
    load_latest_or_create,
    load_model,
    save_checkpoint,
)
from boxing_ai.config import GameConfig
from boxing_ai.controllers import BOT_PROFILES, ScriptedBot
from boxing_ai.env import RENDER_MODE_RGB_ARRAY, BoxingEnv, RoundOutcome
from boxing_ai.geometry import ring_ray_distance, segment_point_distance
from boxing_ai.rendering import WINDOW_HEIGHT, WINDOW_WIDTH, KeyboardController
from boxing_ai.sparring import OnlineLearner, play_round, run_sparring, run_watch
from boxing_ai.strikes import StrikeType
from boxing_ai.training import train_against_bot

SEED = 7


def short_round_config(seconds: float = 10.0) -> GameConfig:
    """Short rounds and a tiny PPO so the learning tests run in seconds."""
    base = GameConfig()
    return dataclasses.replace(
        base,
        timing=dataclasses.replace(base.timing, round_duration_seconds=seconds),
        training=dataclasses.replace(
            base.training,
            rollout_steps_per_env=64,
            batch_size=32,
            epochs_per_update=1,
            parallel_envs=2,
            checkpoint_interval_steps=128,
        ),
    )


def make_env(config: GameConfig, difficulty: str = "medium", **kwargs) -> BoxingEnv:
    return BoxingEnv(ScriptedBot(config, BOT_PROFILES[difficulty], seed=SEED), config, **kwargs)


class AdvancingJabber:
    """Opponent that walks forward throwing jabs, to force a knockout."""

    def reset(self) -> None:
        return None

    def decide(self, match, boxer_index) -> BoxerCommand:
        return BoxerCommand(forward=1, strike=StrikeType.JAB)


# ------------------------------------------------------------------ geometry


def test_segment_point_distance():
    start, end = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
    assert segment_point_distance(start, end, np.array([1.0, 3.0, 0.0])) == pytest.approx(3.0)
    assert segment_point_distance(start, end, np.array([5.0, 0.0, 4.0])) == pytest.approx(5.0)
    assert segment_point_distance(start, start, np.array([0.0, 2.0, 0.0])) == pytest.approx(2.0)


def test_ring_ray_distance():
    position = np.array([1.0, 0.0])
    assert ring_ray_distance(position, np.array([1.0, 0.0]), 3.0) == pytest.approx(2.0)
    assert ring_ray_distance(position, np.array([-1.0, 0.0]), 3.0) == pytest.approx(4.0)
    assert ring_ray_distance(position, np.array([0.0, 1.0]), 3.0) == pytest.approx(3.0)


# ------------------------------------------------------------------- actions


def test_every_action_decodes_to_a_valid_command():
    space = build_action_space()
    assert list(space.nvec) == [9, 4, 5]
    for locomotion in range(len(LOCOMOTION_CHOICES)):
        for defense in range(len(DEFENSE_CHOICES)):
            for offense in range(len(OFFENSE_CHOICES)):
                command = decode_action(np.array([locomotion, defense, offense]))
                assert isinstance(command, BoxerCommand)


def test_decode_action_maps_heads_to_mechanics():
    command = decode_action(np.array([1, 3, 4]))
    assert (command.forward, command.strafe) == (1, 0)
    assert command.defense is GuardState.SLIP
    assert command.strike is StrikeType.UPPERCUT


def test_decode_action_rejects_wrong_shape():
    with pytest.raises(AssertionError):
        decode_action(np.array([0, 0]))


# --------------------------------------------------------------- environment


def test_env_follows_the_gymnasium_api(config):
    check_env(make_env(config), skip_render_check=True)


def test_observations_stay_inside_the_declared_bounds(config):
    env = make_env(config, "hard")
    observation, _ = env.reset(seed=SEED)
    env.action_space.seed(SEED)
    for _ in range(3000):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            observation, _ = env.reset()


def test_env_is_deterministic_for_a_given_seed(config):
    def rollout() -> list[float]:
        env = make_env(config)
        env.reset(seed=SEED)
        env.action_space.seed(SEED)
        return [env.step(env.action_space.sample())[1] for _ in range(300)]

    assert rollout() == rollout()


def test_observation_flags_an_incoming_windup(config):
    """The agent can see which punch is being wound up before it lands."""
    env = make_env(config, "passive")
    idle_observation, _ = env.reset(seed=SEED)
    opponent = env.match.boxers[1]
    env.match.step((BoxerCommand(), BoxerCommand(strike=StrikeType.CROSS)))
    assert opponent.is_striking

    windup_observation = env._observations.build(env.match, observer_index=0)
    changed = np.flatnonzero(windup_observation != idle_observation)
    newly_set_flags = [
        index
        for index in changed
        if idle_observation[index] == 0.0 and windup_observation[index] == 1.0
    ]
    assert newly_set_flags, "no wind-up flag was raised"


def test_step_requires_reset_and_a_valid_action(config):
    env = make_env(config)
    with pytest.raises(AssertionError):
        env.step(np.array([0, 0, 0]))
    env.reset(seed=SEED)
    with pytest.raises(AssertionError):
        env.step(np.array([99, 0, 0]))


def test_timer_truncates_and_knockout_terminates():
    config = short_round_config(seconds=1.0)
    env = make_env(config, "passive")
    env.reset(seed=SEED)
    idle = np.array([0, 0, 0])
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(idle)
    assert truncated and not terminated
    assert info["outcome"] == RoundOutcome.TIME_UP.value

    config = short_round_config(seconds=20.0)
    knockout_env = BoxingEnv(AdvancingJabber(), config)
    knockout_env.reset(seed=SEED)
    knockout_env.match.boxers[0].health = 0.01
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = knockout_env.step(idle)
    assert terminated and not truncated
    assert info["outcome"] == RoundOutcome.AGENT_KNOCKOUT_LOSS.value
    assert info["reward_terms"]["outcome"] == -config.rewards.knockout_reward


def test_rgb_array_rendering(config):
    env = make_env(config, render_mode=RENDER_MODE_RGB_ARRAY)
    env.reset(seed=SEED)
    frame = env.render()
    assert frame.shape == (WINDOW_HEIGHT, WINDOW_WIDTH, 3)
    assert frame.any()
    env.close()


def test_keyboard_movement_is_screen_relative():
    """Pressing toward the top of the screen moves up whatever the facing."""
    up = np.array([0.0, 1.0])
    assert KeyboardController._to_local_steps(up, facing=np.pi / 2) == (1, 0)
    assert KeyboardController._to_local_steps(up, facing=-np.pi / 2) == (-1, 0)
    assert KeyboardController._to_local_steps(up, facing=0.0) == (0, 1)
    assert KeyboardController._to_local_steps(np.zeros(2), facing=0.0) == (0, 0)


# ------------------------------------------------------- checkpoints & loops


def test_checkpoint_round_trip_preserves_the_policy(tmp_path):
    config = short_round_config()
    env = make_env(config)
    observation, _ = env.reset(seed=SEED)
    model = create_model(env, config, SEED)

    path = save_checkpoint(model, tmp_path)
    assert path.suffix == ".zip"
    assert list_checkpoints(tmp_path) == [path]

    reloaded = load_model(path, env)
    original_action, _ = model.predict(observation, deterministic=True)
    reloaded_action, _ = reloaded.predict(observation, deterministic=True)
    assert np.array_equal(original_action, reloaded_action)


def test_load_latest_picks_the_most_trained_checkpoint(tmp_path):
    config = short_round_config()
    env = make_env(config)
    model = create_model(env, config, SEED)
    save_checkpoint(model, tmp_path)
    model.num_timesteps = 5000
    save_checkpoint(model, tmp_path)

    assert load_latest_or_create(tmp_path, env, config, SEED).num_timesteps == 5000
    assert load_latest_or_create(tmp_path / "empty", env, config, SEED).num_timesteps == 0


def test_online_learner_updates_weights_between_rounds():
    config = short_round_config()
    env = make_env(config)
    env.reset(seed=SEED)
    model = create_model(env, config, SEED)
    learner = OnlineLearner(model)
    weights_before = [parameter.detach().clone() for parameter in model.policy.parameters()]

    summary = play_round(env, learner.select_action, learner.record)
    assert learner.pending_decisions == summary.decisions
    assert learner.is_ready_to_update

    assert learner.update() == summary.decisions
    assert learner.pending_decisions == 0
    assert model.num_timesteps == summary.decisions
    weights_changed = any(
        not np.array_equal(before.numpy(), after.detach().numpy())
        for before, after in zip(weights_before, model.policy.parameters())
    )
    assert weights_changed


def test_sparring_saves_a_checkpoint_per_round_and_resumes(tmp_path):
    config = short_round_config()
    arguments = dict(player="medium", show_window=False, seed=SEED)

    first_session = run_sparring(config, tmp_path, rounds=2, **arguments)
    assert len(first_session) == 2
    assert len(list_checkpoints(tmp_path)) == 2

    run_sparring(config, tmp_path, rounds=1, **arguments)
    checkpoints = list_checkpoints(tmp_path)
    assert len(checkpoints) == 3, "the second session continued from the first"

    watched = run_watch(config, None, tmp_path, rounds=1, **arguments)
    assert len(watched) == 1
    assert list_checkpoints(tmp_path) == checkpoints, "watching never saves"


def test_headless_training_then_sparring_share_checkpoints(tmp_path):
    config = short_round_config()
    last = train_against_bot(config, tmp_path, total_timesteps=128, difficulty="easy", seed=SEED)
    trained_steps = load_model(last, make_env(config)).num_timesteps
    assert trained_steps >= 128

    run_sparring(config, tmp_path, rounds=1, player="easy", show_window=False, seed=SEED)
    newest = list_checkpoints(tmp_path)[-1]
    assert load_model(newest, make_env(config)).num_timesteps > trained_steps


def test_human_player_requires_a_window(tmp_path):
    with pytest.raises(AssertionError):
        run_sparring(
            short_round_config(), tmp_path, rounds=1, player="human",
            show_window=False, seed=SEED,
        )
