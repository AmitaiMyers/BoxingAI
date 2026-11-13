"""Stamina, exhaustion and health rules."""

from __future__ import annotations

import pytest

from boxing_ai.boxer import BoxerCommand, GuardState
from boxing_ai.strikes import StrikeType
from tests.conftest import ATTACKER, DEFENDER, all_hits, place_boxers, run_frames

ONE_SECOND_FRAMES = 60
CLOSE_RANGE = 0.6
OUT_OF_RANGE = 2.5


def stamina_after_one_second(match, command: BoxerCommand, start: float = 0.5) -> float:
    place_boxers(match, OUT_OF_RANGE)
    boxer = match.boxers[ATTACKER]
    boxer.stamina = start
    run_frames(match, ONE_SECOND_FRAMES, attacker_command=command)
    return boxer.stamina - start


def test_idle_regenerates_stamina(match, config):
    change = stamina_after_one_second(match, BoxerCommand())
    assert change == pytest.approx(config.stamina.idle_regen_per_second)


def test_retreating_regenerates_more_slowly_than_idling(match, config):
    change = stamina_after_one_second(match, BoxerCommand(forward=-1))
    assert change == pytest.approx(config.stamina.retreat_regen_per_second)
    assert 0.0 < change < config.stamina.idle_regen_per_second


@pytest.mark.parametrize("command", [BoxerCommand(forward=1), BoxerCommand(strafe=1)])
def test_advancing_and_circling_cost_stamina(match, config, command):
    change = stamina_after_one_second(match, command)
    assert change == pytest.approx(-config.stamina.move_cost_per_second)


@pytest.mark.parametrize("guard", [GuardState.HIGH, GuardState.LOW])
def test_holding_a_guard_drains_stamina(match, config, guard):
    change = stamina_after_one_second(match, BoxerCommand(defense=guard))
    assert change == pytest.approx(-config.stamina.guard_cost_per_second)


def test_stamina_never_exceeds_full(match):
    run_frames(match, ONE_SECOND_FRAMES)
    assert match.boxers[ATTACKER].stamina == 1.0


@pytest.mark.parametrize("strike", list(StrikeType))
def test_landed_strike_costs_its_stamina_cost(match, config, strike):
    place_boxers(match, CLOSE_RANGE)
    spec = config.combat.strikes[strike]
    match.step((BoxerCommand(strike=strike), BoxerCommand()))
    run_frames(match, spec.total_frames - 1)
    assert match.boxers[ATTACKER].stamina == pytest.approx(1.0 - spec.stamina_cost)


@pytest.mark.parametrize("strike", list(StrikeType))
def test_missed_strike_costs_extra(match, config, strike):
    place_boxers(match, OUT_OF_RANGE)
    spec = config.combat.strikes[strike]
    match.step((BoxerCommand(strike=strike), BoxerCommand()))
    run_frames(match, spec.total_frames - 1)
    expected_cost = spec.stamina_cost * (
        1.0 + config.stamina.missed_strike_extra_cost_fraction
    )
    assert match.boxers[ATTACKER].stamina == pytest.approx(1.0 - expected_cost)


def exhaust(match) -> None:
    """Empty the attacker's tank by throwing a punch it cannot afford."""
    match.boxers[ATTACKER].stamina = 0.01
    events = match.step((BoxerCommand(strike=StrikeType.CROSS), BoxerCommand()))
    assert events.newly_exhausted == (ATTACKER,)


def test_zero_stamina_triggers_exhaustion_for_n_frames(match, config):
    place_boxers(match, OUT_OF_RANGE)
    boxer = match.boxers[ATTACKER]
    exhaust(match)
    assert boxer.is_exhausted
    assert not boxer.is_striking, "the unaffordable punch is cancelled"

    events = run_frames(match, config.stamina.exhaustion_duration_frames - 2)
    assert boxer.is_exhausted
    assert all(frame.newly_exhausted == () for frame in events), "reported only once"

    run_frames(match, 1)
    assert not boxer.is_exhausted
    assert boxer.stamina > 0.0, "recovers some stamina while stunned"


def test_exhausted_boxer_cannot_punch_guard_or_slip(match):
    place_boxers(match, CLOSE_RANGE)
    boxer = match.boxers[ATTACKER]
    exhaust(match)
    for command in (
        BoxerCommand(strike=StrikeType.JAB),
        BoxerCommand(defense=GuardState.HIGH),
        BoxerCommand(defense=GuardState.SLIP),
    ):
        match.step((command, BoxerCommand()))
        assert not boxer.is_striking
        assert not boxer.is_slipping
        assert boxer.guard is GuardState.NONE


def test_exhausted_boxer_moves_at_half_speed(match, config):
    place_boxers(match, OUT_OF_RANGE)
    boxer = match.boxers[ATTACKER]
    exhaust(match)
    match.step((BoxerCommand(forward=-1), BoxerCommand()))
    speed = float((boxer.velocity @ boxer.velocity) ** 0.5)
    assert speed == pytest.approx(
        config.body.backward_speed * config.body.exhausted_speed_multiplier
    )


def test_exhaustion_disables_damage_mitigation(match, config):
    place_boxers(match, CLOSE_RANGE)
    defender = match.boxers[DEFENDER]
    guard = BoxerCommand(defense=GuardState.HIGH)
    # Drain the defender's stamina to zero by guarding.
    defender.stamina = 0.001
    run_frames(match, 2, defender_command=guard)
    assert defender.is_exhausted

    match.step((BoxerCommand(strike=StrikeType.JAB), guard))
    hits = all_hits(run_frames(match, 20, defender_command=guard))
    jab = config.combat.strikes[StrikeType.JAB]
    assert not hits[0].was_blocked
    assert hits[0].damage == pytest.approx(
        config.combat.base_strike_damage * jab.damage_multiplier
    )


def test_health_reaching_zero_is_a_knockout_and_ends_the_round(match):
    place_boxers(match, CLOSE_RANGE)
    defender = match.boxers[DEFENDER]
    defender.health = 0.01
    match.step((BoxerCommand(strike=StrikeType.JAB), BoxerCommand()))
    while not match.is_over:
        match.step((BoxerCommand(), BoxerCommand()))

    assert defender.health == 0.0
    assert match.knocked_out == (DEFENDER,)
    with pytest.raises(AssertionError):
        match.step((BoxerCommand(), BoxerCommand()))


def test_round_ends_when_the_timer_runs_out(match, config):
    place_boxers(match, OUT_OF_RANGE)
    run_frames(match, config.timing.round_frames)
    assert match.is_time_up
    assert match.is_over
    assert match.knocked_out == ()
