"""Every term of the shaping reward, in isolation."""

from __future__ import annotations

import numpy as np
import pytest

from boxing_ai.boxer import BoxerCommand, GuardState
from boxing_ai.rewards import RewardBreakdown, RewardCalculator
from boxing_ai.strikes import StrikeType
from tests.conftest import place_boxers

# In these tests the learning boxer is boxer 0.
AGENT = 0
OPPONENT = 1
CLOSE_RANGE = 0.6
OUT_OF_RANGE = 2.5
IDLE = BoxerCommand()


def total_reward(match, config, frames, agent_command=IDLE, opponent_command=IDLE):
    """Hold two commands for ``frames`` frames and sum the agent's reward."""
    calculator = RewardCalculator(config, AGENT)
    return accumulate(match, calculator, frames, agent_command, opponent_command)


def accumulate(match, calculator, frames, agent_command=IDLE, opponent_command=IDLE):
    reward = RewardBreakdown()
    for _ in range(frames):
        events = match.step((agent_command, opponent_command))
        reward = reward + calculator.frame_reward(match, events)
    return reward


def punch_then_wait(match, config, calculator, attacker_index, strike, other_command=IDLE):
    """One boxer throws ``strike``; return the agent's reward over the punch."""
    punch = BoxerCommand(strike=strike)
    first = (punch, other_command) if attacker_index == AGENT else (other_command, punch)
    events = match.step(first)
    reward = calculator.frame_reward(match, events)
    frames = config.combat.strikes[strike].total_frames - 1
    hold = (IDLE, other_command) if attacker_index == AGENT else (other_command, IDLE)
    return reward + accumulate(match, calculator, frames, *hold)


def test_breakdown_total_is_sum_of_terms():
    breakdown = RewardBreakdown(clean_hit=5.0, damage_taken=-2.0, passivity=-0.5)
    assert breakdown.total == pytest.approx(2.5)
    assert (breakdown + breakdown).total == pytest.approx(5.0)


def test_clean_hit_reward_scales_with_impact_speed(match, config):
    place_boxers(match, CLOSE_RANGE)
    calculator = RewardCalculator(config, AGENT)
    reward = punch_then_wait(match, config, calculator, AGENT, StrikeType.JAB)

    assert 0.0 < reward.clean_hit <= config.rewards.clean_hit_reward
    assert reward.damage_taken == 0.0


def test_blocked_punch_earns_no_clean_hit_reward(match, config):
    place_boxers(match, CLOSE_RANGE)
    calculator = RewardCalculator(config, AGENT)
    guard = BoxerCommand(defense=GuardState.HIGH)
    reward = punch_then_wait(match, config, calculator, AGENT, StrikeType.JAB, guard)
    assert reward.clean_hit == 0.0


def test_damage_taken_is_penalised_per_health_point(match, config):
    place_boxers(match, CLOSE_RANGE)
    calculator = RewardCalculator(config, AGENT)
    reward = punch_then_wait(match, config, calculator, OPPONENT, StrikeType.CROSS)

    cross = config.combat.strikes[StrikeType.CROSS]
    health_points_lost = config.combat.base_strike_damage * cross.damage_multiplier * 100.0
    assert reward.damage_taken == pytest.approx(
        -config.rewards.damage_penalty_per_health_point * health_points_lost
    )


def test_timed_block_is_rewarded(match, config):
    place_boxers(match, CLOSE_RANGE)
    calculator = RewardCalculator(config, AGENT)
    guard = BoxerCommand(defense=GuardState.HIGH)
    reward = punch_then_wait(match, config, calculator, OPPONENT, StrikeType.JAB, guard)
    assert reward.timed_defense == pytest.approx(config.rewards.timed_defense_reward)


def test_slipping_a_punch_is_rewarded(match, config):
    place_boxers(match, CLOSE_RANGE)
    calculator = RewardCalculator(config, AGENT)
    jab = config.combat.strikes[StrikeType.JAB]

    reward = accumulate(
        match, calculator, 1, opponent_command=BoxerCommand(strike=StrikeType.JAB)
    )
    reward = reward + accumulate(match, calculator, jab.windup_frames - 2)
    slip = BoxerCommand(defense=GuardState.SLIP)
    reward = reward + accumulate(match, calculator, jab.active_frames + 1, slip)

    assert reward.timed_defense == pytest.approx(config.rewards.timed_defense_reward)
    assert reward.damage_taken == 0.0


def test_center_control_pays_per_second_only_when_closer_to_center(match, config):
    seconds = 2
    frames = seconds * config.timing.frames_per_second

    match.boxers[AGENT].position = np.array([0.0, 0.0])
    match.boxers[OPPONENT].position = np.array([2.0, 0.0])
    holding = total_reward(match, config, frames)
    assert holding.center_control == pytest.approx(
        config.rewards.center_control_reward_per_second * seconds
    )

    match.boxers[AGENT].position = np.array([-2.5, 0.0])
    match.boxers[OPPONENT].position = np.array([0.0, 0.0])
    losing = total_reward(match, config, frames)
    assert losing.center_control == 0.0


def test_exhaustion_is_penalised_once(match, config):
    place_boxers(match, OUT_OF_RANGE)
    match.boxers[AGENT].stamina = 0.01
    reward = total_reward(match, config, 30, BoxerCommand(strike=StrikeType.CROSS))
    assert reward.exhaustion == pytest.approx(-config.rewards.exhaustion_penalty)


def test_passivity_penalty_starts_after_the_grace_period(match, config):
    place_boxers(match, OUT_OF_RANGE)
    calculator = RewardCalculator(config, AGENT)
    grace_frames = config.timing.seconds_to_frames(config.rewards.passivity_grace_seconds)

    within_grace = accumulate(match, calculator, grace_frames)
    assert within_grace.passivity == 0.0

    one_second = config.timing.frames_per_second
    after_grace = accumulate(match, calculator, one_second)
    assert after_grace.passivity == pytest.approx(
        -config.rewards.passivity_penalty_per_second
    )


def test_punching_in_range_resets_passivity_but_shadow_boxing_does_not(match, config):
    grace_frames = config.timing.seconds_to_frames(config.rewards.passivity_grace_seconds)
    jab = BoxerCommand(strike=StrikeType.JAB)

    place_boxers(match, CLOSE_RANGE)
    calculator = RewardCalculator(config, AGENT)
    accumulate(match, calculator, grace_frames)
    accumulate(match, calculator, 1, jab)
    engaged = accumulate(match, calculator, grace_frames)
    assert engaged.passivity == 0.0

    place_boxers(match, OUT_OF_RANGE)
    calculator = RewardCalculator(config, AGENT)
    accumulate(match, calculator, grace_frames)
    accumulate(match, calculator, 1, jab)
    shadow_boxing = accumulate(match, calculator, grace_frames)
    assert shadow_boxing.passivity < 0.0


def test_guard_spam_is_penalised_after_the_grace_period(match, config):
    place_boxers(match, OUT_OF_RANGE)
    calculator = RewardCalculator(config, AGENT)
    guard = BoxerCommand(defense=GuardState.HIGH)
    grace_frames = config.timing.seconds_to_frames(config.rewards.guard_spam_grace_seconds)

    within_grace = accumulate(match, calculator, grace_frames, guard)
    assert within_grace.guard_spam == 0.0

    one_second = config.timing.frames_per_second
    spamming = accumulate(match, calculator, one_second, guard)
    assert spamming.guard_spam == pytest.approx(
        -config.rewards.guard_spam_penalty_per_second
    )

    # Dropping the guard for a single frame does not wipe the slate clean.
    accumulate(match, calculator, 1)
    flicker = accumulate(match, calculator, one_second, guard)
    assert flicker.guard_spam < 0.0


def test_counter_punching_clears_the_guard_spam_debt(match, config):
    place_boxers(match, CLOSE_RANGE)
    calculator = RewardCalculator(config, AGENT)
    guard = BoxerCommand(defense=GuardState.HIGH)
    grace_frames = config.timing.seconds_to_frames(config.rewards.guard_spam_grace_seconds)

    accumulate(match, calculator, grace_frames + 10, guard)
    accumulate(match, calculator, 1, BoxerCommand(strike=StrikeType.JAB))
    jab_frames = config.combat.strikes[StrikeType.JAB].total_frames
    accumulate(match, calculator, jab_frames)
    after_counter = accumulate(match, calculator, grace_frames, guard)
    assert after_counter.guard_spam == 0.0


@pytest.mark.parametrize("loser, sign", [(OPPONENT, 1.0), (AGENT, -1.0)])
def test_knockout_outcome(match, config, loser, sign):
    place_boxers(match, CLOSE_RANGE)
    match.boxers[loser].health = 0.01
    calculator = RewardCalculator(config, AGENT)
    winner = OPPONENT if loser == AGENT else AGENT
    reward = punch_then_wait_until_over(match, calculator, winner)
    assert reward.outcome == pytest.approx(sign * config.rewards.knockout_reward)


def punch_then_wait_until_over(match, calculator, attacker_index):
    punch = BoxerCommand(strike=StrikeType.JAB)
    commands = (punch, IDLE) if attacker_index == AGENT else (IDLE, punch)
    reward = RewardBreakdown()
    while not match.is_over:
        events = match.step(commands)
        reward = reward + calculator.frame_reward(match, events)
        commands = (IDLE, IDLE)
    return reward


def test_timeout_outcome_rewards_the_health_advantage(match, config):
    place_boxers(match, OUT_OF_RANGE)
    match.boxers[AGENT].health = 0.9
    match.boxers[OPPONENT].health = 0.4
    reward = total_reward(match, config, config.timing.round_frames)
    assert match.is_time_up
    assert reward.outcome == pytest.approx(
        config.rewards.timeout_health_advantage_reward * 0.5
    )
