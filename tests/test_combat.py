"""Punch frame data, hit detection, guards and slips."""

from __future__ import annotations

import pytest

from boxing_ai.boxer import BoxerCommand, GuardState
from boxing_ai.controllers import strike_reach
from boxing_ai.strikes import StrikePhase, StrikeType
from tests.conftest import (
    ATTACKER,
    DEFENDER,
    all_evades,
    all_hits,
    place_boxers,
    run_frames,
)

CLOSE_RANGE = 0.6
OUT_OF_RANGE = 2.5


def throw(match, strike: StrikeType, defender_command=BoxerCommand()):
    """Throw one punch and simulate until it has fully recovered."""
    frames = match.config.combat.strikes[strike].total_frames
    first = match.step((BoxerCommand(strike=strike), defender_command))
    rest = run_frames(match, frames - 1, defender_command=defender_command)
    return [first, *rest]


@pytest.mark.parametrize("strike", list(StrikeType))
def test_strike_goes_through_windup_active_recovery(match, config, strike):
    spec = config.combat.strikes[strike]
    attacker = match.boxers[ATTACKER]
    phases = []
    match.step((BoxerCommand(strike=strike), BoxerCommand()))
    phases.append(attacker.strike_phase)
    while attacker.is_striking:
        match.step((BoxerCommand(), BoxerCommand()))
        phases.append(attacker.strike_phase)

    # The first recorded phase is after frame 1, the last is IDLE.
    assert phases.count(StrikePhase.WINDUP) == spec.windup_frames - 1
    assert phases.count(StrikePhase.ACTIVE) == spec.active_frames
    assert phases.count(StrikePhase.RECOVERY) == spec.recovery_frames
    assert phases[-1] is StrikePhase.IDLE


@pytest.mark.parametrize("strike", list(StrikeType))
def test_clean_hit_deals_base_damage_times_multiplier(match, config, strike):
    place_boxers(match, CLOSE_RANGE)
    hits = all_hits(throw(match, strike))

    expected = config.combat.base_strike_damage * config.combat.strikes[strike].damage_multiplier
    assert len(hits) == 1, "a punch can only connect once"
    assert hits[0].damage == pytest.approx(expected)
    assert not hits[0].was_blocked
    assert hits[0].impact_speed > 0.0
    assert match.boxers[DEFENDER].health == pytest.approx(1.0 - expected)


@pytest.mark.parametrize("strike", list(StrikeType))
def test_strike_only_lands_within_its_reach(match, config, strike):
    reach = strike_reach(config, strike)
    margin = 0.05

    place_boxers(match, reach - margin)
    assert len(all_hits(throw(match, strike))) == 1

    place_boxers(match, reach + margin)
    assert all_hits(throw(match, strike)) == []


def test_hit_only_happens_during_active_frames(match, config):
    place_boxers(match, CLOSE_RANGE)
    spec = config.combat.strikes[StrikeType.CROSS]
    events = throw(match, StrikeType.CROSS)
    hit_frames = [index for index, frame in enumerate(events) if frame.hits]
    assert len(hit_frames) == 1
    assert spec.windup_frames <= hit_frames[0] < spec.windup_frames + spec.active_frames


@pytest.mark.parametrize(
    "strike, guard",
    [
        (StrikeType.JAB, GuardState.HIGH),
        (StrikeType.CROSS, GuardState.HIGH),
        (StrikeType.UPPERCUT, GuardState.HIGH),
        (StrikeType.HOOK, GuardState.LOW),
    ],
)
def test_matching_guard_removes_eighty_percent_of_damage(match, config, strike, guard):
    place_boxers(match, CLOSE_RANGE)
    hits = all_hits(throw(match, strike, defender_command=BoxerCommand(defense=guard)))

    full_damage = (
        config.combat.base_strike_damage * config.combat.strikes[strike].damage_multiplier
    )
    assert hits[0].was_blocked
    assert hits[0].damage == pytest.approx(full_damage * 0.2)


@pytest.mark.parametrize(
    "strike, guard",
    [(StrikeType.JAB, GuardState.LOW), (StrikeType.HOOK, GuardState.HIGH)],
)
def test_wrong_guard_does_not_block(match, strike, guard):
    place_boxers(match, CLOSE_RANGE)
    hits = all_hits(throw(match, strike, defender_command=BoxerCommand(defense=guard)))
    assert not hits[0].was_blocked


def test_block_is_timed_only_when_guard_was_raised_recently(match, config):
    place_boxers(match, CLOSE_RANGE)
    guard = BoxerCommand(defense=GuardState.HIGH)

    fresh = all_hits(throw(match, StrikeType.JAB, defender_command=guard))
    assert fresh[0].was_timed_block

    # Keep holding the same guard well past the timed window, then get hit again.
    run_frames(match, config.defense.timed_block_window_frames, defender_command=guard)
    stale = all_hits(throw(match, StrikeType.JAB, defender_command=guard))
    assert stale[0].was_blocked
    assert not stale[0].was_timed_block


def slip_into(match, config, strike: StrikeType):
    """Throw ``strike``; the defender slips just before the active frames."""
    spec = config.combat.strikes[strike]
    events = [match.step((BoxerCommand(strike=strike), BoxerCommand()))]
    events += run_frames(match, spec.windup_frames - 2)
    slip = BoxerCommand(defense=GuardState.SLIP)
    events += run_frames(match, spec.active_frames + 1, defender_command=slip)
    return events


@pytest.mark.parametrize("strike", [StrikeType.JAB, StrikeType.CROSS])
def test_timed_slip_evades_straight_punches(match, config, strike):
    place_boxers(match, CLOSE_RANGE)
    events = slip_into(match, config, strike)

    assert all_hits(events) == []
    evades = all_evades(events)
    assert len(evades) == 1, "one punch is credited as one evasion"
    assert evades[0].defender_index == DEFENDER
    assert match.boxers[DEFENDER].health == 1.0


def test_slip_does_not_evade_body_hook(match, config):
    place_boxers(match, CLOSE_RANGE)
    events = slip_into(match, config, StrikeType.HOOK)
    assert len(all_hits(events)) == 1
    assert all_evades(events) == []


def test_uppercut_punishes_a_slip_with_extra_damage(match, config):
    place_boxers(match, CLOSE_RANGE)
    hits = all_hits(slip_into(match, config, StrikeType.UPPERCUT))

    spec = config.combat.strikes[StrikeType.UPPERCUT]
    expected = (
        config.combat.base_strike_damage
        * spec.damage_multiplier
        * config.defense.slip_uppercut_damage_multiplier
    )
    assert len(hits) == 1
    assert hits[0].damage == pytest.approx(expected)


def test_slip_cannot_be_repeated_during_cooldown(match, config):
    defender = match.boxers[DEFENDER]
    slip = BoxerCommand(defense=GuardState.SLIP)
    run_frames(match, config.defense.slip_duration_frames, defender_command=slip)
    assert not defender.is_slipping

    run_frames(match, config.defense.slip_cooldown_frames - 1, defender_command=slip)
    assert not defender.is_slipping, "still cooling down"

    run_frames(match, 2, defender_command=slip)
    assert defender.is_slipping


def test_out_of_range_punch_is_not_an_evasion(match, config):
    place_boxers(match, OUT_OF_RANGE)
    events = slip_into(match, config, StrikeType.JAB)
    assert all_evades(events) == []


def test_boxers_cannot_walk_through_each_other_or_the_ropes(match, config):
    place_boxers(match, CLOSE_RANGE)
    advance = BoxerCommand(forward=1)
    run_frames(match, 120, attacker_command=advance, defender_command=advance)
    assert match.distance_between_boxers() >= 2.0 * config.body.body_radius - 1e-9

    retreat = BoxerCommand(forward=-1)
    run_frames(match, 600, attacker_command=retreat, defender_command=retreat)
    limit = config.ring.half_size - config.body.body_radius
    for boxer in match.boxers:
        assert abs(boxer.position[0]) <= limit + 1e-9
        assert abs(boxer.position[1]) <= limit + 1e-9
