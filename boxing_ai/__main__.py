"""Command line: ``python -m boxing_ai {spar,train,watch} ...``"""

from __future__ import annotations

import argparse
from pathlib import Path

from boxing_ai.checkpoints import list_checkpoints
from boxing_ai.config import GameConfig
from boxing_ai.controllers import BOT_PROFILES
from boxing_ai.sparring import (
    HUMAN_PLAYER,
    PLAYER_CHOICES,
    run_sparring,
    run_watch,
    with_round_duration,
)
from boxing_ai.training import train_against_bot

DEFAULT_CHECKPOINT_DIRECTORY = Path("checkpoints")
DEFAULT_SEED = 0
DEFAULT_SPARRING_ROUNDS = 5
DEFAULT_SPARRING_ROUND_SECONDS = 180.0
DEFAULT_TRAINING_ROUND_SECONDS = 60.0
DEFAULT_TRAINING_TIMESTEPS = 500_000
DEFAULT_TRAINING_DIFFICULTY = "easy"


def _add_common_arguments(parser: argparse.ArgumentParser, round_seconds: float) -> None:
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
        help="directory holding this boxer's .zip checkpoints",
    )
    parser.add_argument("--round-seconds", type=float, default=round_seconds)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def _add_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rounds", type=int, default=DEFAULT_SPARRING_ROUNDS)
    parser.add_argument(
        "--player",
        choices=PLAYER_CHOICES,
        default=HUMAN_PLAYER,
        help="who is in the other corner: you, or a scripted stand-in",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="no window, as fast as possible (scripted players only)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boxing_ai", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    spar = commands.add_parser("spar", help="spar live; the boxer learns between rounds")
    _add_common_arguments(spar, DEFAULT_SPARRING_ROUND_SECONDS)
    _add_live_arguments(spar)

    train = commands.add_parser("train", help="fast headless training against the bot")
    _add_common_arguments(train, DEFAULT_TRAINING_ROUND_SECONDS)
    train.add_argument("--timesteps", type=int, default=DEFAULT_TRAINING_TIMESTEPS)
    train.add_argument(
        "--difficulty", choices=sorted(BOT_PROFILES), default=DEFAULT_TRAINING_DIFFICULTY
    )

    watch = commands.add_parser("watch", help="watch or fight a saved boxer; no learning")
    _add_common_arguments(watch, DEFAULT_SPARRING_ROUND_SECONDS)
    _add_live_arguments(watch)
    watch.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="specific .zip to load (default: the most trained one)",
    )

    listing = commands.add_parser("list", help="list saved checkpoints, novice first")
    listing.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()

    if arguments.command == "list":
        for checkpoint in list_checkpoints(arguments.checkpoints):
            print(checkpoint)
        return

    config = with_round_duration(GameConfig(), arguments.round_seconds)
    if arguments.command == "spar":
        run_sparring(
            config,
            arguments.checkpoints,
            arguments.rounds,
            arguments.player,
            show_window=not arguments.headless,
            seed=arguments.seed,
        )
    elif arguments.command == "train":
        train_against_bot(
            config,
            arguments.checkpoints,
            arguments.timesteps,
            arguments.difficulty,
            arguments.seed,
        )
    elif arguments.command == "watch":
        run_watch(
            config,
            arguments.checkpoint,
            arguments.checkpoints,
            arguments.rounds,
            arguments.player,
            show_window=not arguments.headless,
            seed=arguments.seed,
        )


if __name__ == "__main__":
    main()
