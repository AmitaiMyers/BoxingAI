"""Export what the mobile app needs from the Python reference implementation.

Run from the repository root after changing any game rule or number:

    python -m tools.export_for_mobile

It writes two files into ``mobile/src/sim``:

* ``gameConfig.json`` - every tunable number, so the TypeScript simulation
  reads the same values as the Python one instead of keeping a second copy.
* ``__fixtures__/parityTrace.json`` - recorded fights (commands in, boxer
  state and events out). The TypeScript tests replay the commands and must
  reproduce the states, which proves both implementations follow the same
  rules.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from boxing_ai.boxer import Boxer, BoxerCommand, GuardState
from boxing_ai.config import GameConfig
from boxing_ai.simulation import BoxingMatch, FrameEvents, StartPositions
from boxing_ai.strikes import Hand, StrikeType

MOBILE_SIM_DIRECTORY = Path("mobile/src/sim")
CONFIG_FILE = MOBILE_SIM_DIRECTORY / "gameConfig.json"
TRACE_FILE = MOBILE_SIM_DIRECTORY / "__fixtures__" / "parityTrace.json"

# The phone never trains, so PPO hyper-parameters are not exported.
EXPORTED_SECTIONS = (
    "timing", "ring", "body", "defense", "stamina", "combat", "rewards", "observation",
)

TRACE_SEED = 2026
TRACE_EPISODES = 4
TRACE_EPISODE_FRAMES = 1000
# Full boxer state is stored every N frames (and on the last one) to keep the
# fixture small; events are stored for every frame.
TRACE_SNAPSHOT_INTERVAL_FRAMES = 12
# A random command is held for 1..N frames, like a real controller would.
MAX_COMMAND_HOLD_FRAMES = 8
STRIKE_PROBABILITY = 0.35
DEFENSE_PROBABILITY = 0.4
# Starting close together makes sure punches actually land in the trace.
TRACE_START_CENTER_RANGE = 1.5
JSON_INDENT = 1


# ------------------------------------------------------------------- config


def _to_camel_case(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(word.capitalize() for word in tail)


def _to_json(value: Any) -> Any:
    """Convert config values to JSON, with camelCase keys for TypeScript."""
    if dataclasses.is_dataclass(value):
        return {
            _to_camel_case(field.name): _to_json(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {key.name: _to_json(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value if isinstance(value.value, str) else value.name
    if isinstance(value, (tuple, list)):
        return [_to_json(item) for item in value]
    assert isinstance(value, (int, float, str)), f"cannot export {value!r}"
    return value


def export_config(config: GameConfig) -> dict[str, Any]:
    return {name: _to_json(getattr(config, name)) for name in EXPORTED_SECTIONS}


# -------------------------------------------------------------------- trace


def _random_command(rng: np.random.Generator) -> BoxerCommand:
    strike = None
    if rng.random() < STRIKE_PROBABILITY:
        strike = StrikeType(int(rng.integers(len(StrikeType))))
    defense = GuardState.NONE
    if rng.random() < DEFENSE_PROBABILITY:
        defense = GuardState(int(rng.integers(1, len(GuardState))))
    return BoxerCommand(
        forward=int(rng.integers(-1, 2)),
        strafe=int(rng.integers(-1, 2)),
        defense=defense,
        strike=strike,
    )


def _command_json(command: BoxerCommand) -> dict[str, Any]:
    return {
        "forward": command.forward,
        "strafe": command.strafe,
        "defense": command.defense.name,
        "strike": None if command.strike is None else command.strike.name,
    }


def _boxer_json(boxer: Boxer) -> dict[str, Any]:
    return {
        "position": boxer.position.tolist(),
        "velocity": boxer.velocity.tolist(),
        "facing": boxer.facing,
        "health": boxer.health,
        "stamina": boxer.stamina,
        "guard": boxer.guard.name,
        "strikeType": None if boxer.strike_type is None else boxer.strike_type.name,
        "strikeFrame": boxer.strike_frame,
        "strikePhase": boxer.strike_phase.name,
        "slipFramesRemaining": boxer.slip_frames_remaining,
        "slipCooldownFramesRemaining": boxer.slip_cooldown_frames_remaining,
        "exhaustedFramesRemaining": boxer.exhausted_frames_remaining,
        "gloves": {
            hand.value: {
                "position": boxer.gloves[hand].position.tolist(),
                "velocity": boxer.gloves[hand].velocity.tolist(),
                "acceleration": boxer.gloves[hand].acceleration.tolist(),
            }
            for hand in Hand
        },
    }


def _events_json(events: FrameEvents) -> dict[str, Any]:
    return {
        "hits": [
            {
                "attackerIndex": hit.attacker_index,
                "strikeType": hit.strike_type.name,
                "damage": hit.damage,
                "impactSpeed": hit.impact_speed,
                "wasBlocked": hit.was_blocked,
                "wasTimedBlock": hit.was_timed_block,
            }
            for hit in events.hits
        ],
        "evades": [
            {"defenderIndex": evade.defender_index, "strikeType": evade.strike_type.name}
            for evade in events.evades
        ],
        "newlyExhausted": list(events.newly_exhausted),
        "strikesStarted": list(events.strikes_started),
    }


def _record_episode(match: BoxingMatch, rng: np.random.Generator) -> dict[str, Any]:
    start = StartPositions(
        center=rng.uniform(-TRACE_START_CENTER_RANGE, TRACE_START_CENTER_RANGE, size=2),
        axis_angle=float(rng.uniform(-np.pi, np.pi)),
    )
    match.reset(start)
    # Half the default distance, so the boxers begin within punching range.
    for boxer in match.boxers:
        boxer.position = start.center + 0.5 * (boxer.position - start.center)

    frames = []
    commands = [_random_command(rng), _random_command(rng)]
    hold = [0, 0]
    while not match.is_over and len(frames) < TRACE_EPISODE_FRAMES:
        for index in range(len(commands)):
            if hold[index] == 0:
                commands[index] = _random_command(rng)
                hold[index] = int(rng.integers(1, MAX_COMMAND_HOLD_FRAMES + 1))
            hold[index] -= 1
        events = match.step((commands[0], commands[1]))
        is_snapshot_frame = (
            len(frames) % TRACE_SNAPSHOT_INTERVAL_FRAMES == 0 or match.is_over
        )
        frames.append(
            {
                "commands": [_command_json(command) for command in commands],
                "events": _events_json(events),
                "boxers": (
                    [_boxer_json(boxer) for boxer in match.boxers]
                    if is_snapshot_frame
                    else None
                ),
            }
        )
    return {
        "start": {"center": start.center.tolist(), "axisAngle": start.axis_angle},
        "initialPositions": _initial_positions(start, match),
        "frames": frames,
        "knockedOut": list(match.knocked_out),
    }


def _initial_positions(start: StartPositions, match: BoxingMatch) -> list[list[float]]:
    """Positions the episode started from (half the default separation)."""
    reference = BoxingMatch(match.config)
    reference.reset(start)
    return [
        (start.center + 0.5 * (boxer.position - start.center)).tolist()
        for boxer in reference.boxers
    ]


def export_trace(config: GameConfig, seed: int = TRACE_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    match = BoxingMatch(config)
    return {"episodes": [_record_episode(match, rng) for _ in range(TRACE_EPISODES)]}


def missing_coverage(trace: dict[str, Any]) -> list[str]:
    """Rules the recorded fights never exercise (empty when all are covered)."""
    frames = [frame for episode in trace["episodes"] for frame in episode["frames"]]
    hits = [hit for frame in frames for hit in frame["events"]["hits"]]
    covered = {
        "clean hit": any(not hit["wasBlocked"] for hit in hits),
        "blocked hit": any(hit["wasBlocked"] for hit in hits),
        "timed block": any(hit["wasTimedBlock"] for hit in hits),
        "slipped punch": any(frame["events"]["evades"] for frame in frames),
        "exhaustion": any(frame["events"]["newlyExhausted"] for frame in frames),
        "knockout": any(episode["knockedOut"] for episode in trace["episodes"]),
        "round timer": any(not episode["knockedOut"] for episode in trace["episodes"]),
    }
    covered.update(
        {
            f"{strike.name} landing": any(hit["strikeType"] == strike.name for hit in hits)
            for strike in StrikeType
        }
    )
    return [rule for rule, is_covered in covered.items() if not is_covered]


# --------------------------------------------------------------------- main


def _write_json(path: Path, content: dict[str, Any], indent: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=indent) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    config = GameConfig()
    _write_json(CONFIG_FILE, export_config(config), indent=JSON_INDENT)
    # The trace is machine-read only, so it is written compactly.
    trace = export_trace(config)
    missing = missing_coverage(trace)
    assert not missing, f"trace does not exercise {missing}; choose another TRACE_SEED"
    _write_json(TRACE_FILE, trace, indent=None)


if __name__ == "__main__":
    main()
