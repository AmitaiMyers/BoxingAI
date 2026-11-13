"""The shaping reward.

For one physics frame, with ``dt`` the frame duration in seconds, the reward
of the learning boxer is the sum of these terms (weights ``w`` come from
``RewardConfig``):

    clean_hit      = + w_hit * min(impact_speed / reference_speed, 1)
                       for each unblocked punch it lands
    timed_defense  = + w_def   for each punch it slips, or blocks with a guard
                       raised within the "timed block" window
    center_control = + w_ctr * dt   while it is inside the centre radius and
                       closer to the ring centre than its opponent
    damage_taken   = - w_dmg * health points lost (100 points = full health)
    exhaustion     = - w_exh   each time its stamina reaches zero
    passivity      = - w_pas * dt   once it has gone longer than the grace
                       period without throwing a punch within range
    guard_spam     = - w_grd * dt   while it guards after accumulating more
                       than the grace period of guarding without punching
    outcome        = +/- w_ko on a knockout, or
                       w_time * (own health - opponent health) on the bell

Rates are paid per second, not per frame, so the result does not depend on
the frame rate or on how often the policy acts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from boxing_ai.boxer import HELD_GUARDS
from boxing_ai.config import GameConfig
from boxing_ai.simulation import BoxingMatch, FrameEvents, other_index

HEALTH_POINTS_AT_FULL_HEALTH = 100.0


@dataclass(frozen=True)
class RewardBreakdown:
    """One reward, split by cause so training can be inspected and tuned."""

    clean_hit: float = 0.0
    timed_defense: float = 0.0
    center_control: float = 0.0
    damage_taken: float = 0.0
    exhaustion: float = 0.0
    passivity: float = 0.0
    guard_spam: float = 0.0
    outcome: float = 0.0

    @property
    def total(self) -> float:
        return sum(asdict(self).values())

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __add__(self, other: "RewardBreakdown") -> "RewardBreakdown":
        mine, theirs = asdict(self), asdict(other)
        return RewardBreakdown(**{name: mine[name] + theirs[name] for name in mine})


class RewardCalculator:
    """Stateful: it remembers how long the boxer has been passive or guarding."""

    def __init__(self, config: GameConfig, boxer_index: int) -> None:
        self._config = config
        self._weights = config.rewards
        self._boxer_index = boxer_index
        self._opponent_index = other_index(boxer_index)
        self._passivity_grace_frames = config.timing.seconds_to_frames(
            self._weights.passivity_grace_seconds
        )
        self._guard_spam_grace_frames = config.timing.seconds_to_frames(
            self._weights.guard_spam_grace_seconds
        )
        self.reset()

    def reset(self) -> None:
        self._frames_since_engagement = 0
        self._guard_frames_without_offense = 0

    def frame_reward(self, match: BoxingMatch, events: FrameEvents) -> RewardBreakdown:
        """Reward for the frame that ``match.step`` just simulated."""
        self._update_pacing_counters(match, events)
        return RewardBreakdown(
            clean_hit=self._clean_hit(events),
            timed_defense=self._timed_defense(events),
            center_control=self._center_control(match),
            damage_taken=self._damage_taken(events),
            exhaustion=self._exhaustion(events),
            passivity=self._passivity(),
            guard_spam=self._guard_spam(match),
            outcome=self._outcome(match),
        )

    # --------------------------------------------------------------- counters

    def _update_pacing_counters(self, match: BoxingMatch, events: FrameEvents) -> None:
        boxer = match.boxers[self._boxer_index]
        threw_in_range = (
            self._boxer_index in events.strikes_started
            and match.distance_between_boxers() <= self._weights.engagement_range
        )
        if threw_in_range:
            self._frames_since_engagement = 0
            self._guard_frames_without_offense = 0
            return

        self._frames_since_engagement += 1
        if boxer.guard in HELD_GUARDS:
            self._guard_frames_without_offense += 1
        else:
            # The debt drains slowly, so flickering the guard does not clear it.
            self._guard_frames_without_offense = max(
                0, self._guard_frames_without_offense - 1
            )

    # ------------------------------------------------------------------ terms

    def _clean_hit(self, events: FrameEvents) -> float:
        reward = 0.0
        for hit in events.hits:
            if hit.attacker_index == self._boxer_index and not hit.was_blocked:
                impact = min(hit.impact_speed / self._weights.impact_speed_reference, 1.0)
                reward += self._weights.clean_hit_reward * impact
        return reward

    def _timed_defense(self, events: FrameEvents) -> float:
        timed_blocks = sum(
            1
            for hit in events.hits
            if hit.attacker_index == self._opponent_index and hit.was_timed_block
        )
        slips = sum(
            1 for evade in events.evades if evade.defender_index == self._boxer_index
        )
        return self._weights.timed_defense_reward * (timed_blocks + slips)

    def _center_control(self, match: BoxingMatch) -> float:
        own_distance = float(np.linalg.norm(match.boxers[self._boxer_index].position))
        opponent_distance = float(
            np.linalg.norm(match.boxers[self._opponent_index].position)
        )
        holds_center = (
            own_distance <= self._weights.center_control_radius
            and own_distance < opponent_distance
        )
        return self._per_frame(self._weights.center_control_reward_per_second) * holds_center

    def _damage_taken(self, events: FrameEvents) -> float:
        health_lost = sum(
            hit.damage for hit in events.hits if hit.attacker_index == self._opponent_index
        )
        return (
            -self._weights.damage_penalty_per_health_point
            * health_lost
            * HEALTH_POINTS_AT_FULL_HEALTH
        )

    def _exhaustion(self, events: FrameEvents) -> float:
        if self._boxer_index in events.newly_exhausted:
            return -self._weights.exhaustion_penalty
        return 0.0

    def _passivity(self) -> float:
        if self._frames_since_engagement <= self._passivity_grace_frames:
            return 0.0
        return -self._per_frame(self._weights.passivity_penalty_per_second)

    def _guard_spam(self, match: BoxingMatch) -> float:
        is_guarding = match.boxers[self._boxer_index].guard in HELD_GUARDS
        over_budget = self._guard_frames_without_offense > self._guard_spam_grace_frames
        if not (is_guarding and over_budget):
            return 0.0
        return -self._per_frame(self._weights.guard_spam_penalty_per_second)

    def _outcome(self, match: BoxingMatch) -> float:
        if not match.is_over:
            return 0.0
        knocked_out = match.knocked_out
        if knocked_out:
            won = self._opponent_index in knocked_out
            lost = self._boxer_index in knocked_out
            # A double knockout is a draw: the two cancel out.
            return self._weights.knockout_reward * (float(won) - float(lost))
        health_advantage = (
            match.boxers[self._boxer_index].health
            - match.boxers[self._opponent_index].health
        )
        return self._weights.timeout_health_advantage_reward * health_advantage

    def _per_frame(self, rate_per_second: float) -> float:
        return rate_per_second * self._config.timing.frame_seconds
