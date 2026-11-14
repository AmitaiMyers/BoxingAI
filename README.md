# Boxing AI

Train a boxer from scratch by sparring with it. The boxer is a PPO policy
(PyTorch + Stable-Baselines3) that starts with random weights. You fight it in
real time; between rounds it learns from what just happened and saves a
checkpoint, so you can watch it grow from a flailing novice into a counter-puncher.

## Setup

Windows (Command Prompt):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Run each line separately. After activating, the prompt starts with `(.venv)`;
the packages are installed inside that folder, not into the system Python.

## Commands

```bash
# Spar live. The boxer learns in the break after every round.
python -m boxing_ai spar --rounds 5

# Optional gym grind: fast headless training against the scripted bot.
python -m boxing_ai train --timesteps 500000 --difficulty easy

# Watch (or fight) a saved boxer. Nothing is learned or saved.
python -m boxing_ai watch --player medium
python -m boxing_ai watch --checkpoint checkpoints/boxer_steps_0000000000.zip

# List checkpoints, least trained first.
python -m boxing_ai list
```

All commands share one checkpoint directory (`--checkpoints`, default
`checkpoints/`), so live sparring and headless training build on each other.
Use a different directory per boxer. Other options:

| Option | Meaning |
| --- | --- |
| `--player human\|passive\|easy\|medium\|hard` | Who is in the other corner (`spar`, `watch`). Default `human`. |
| `--headless` | No window, as fast as possible. Scripted players only. |
| `--round-seconds N` | Round length. Default 180 for `spar`/`watch`, 60 for `train`. |
| `--difficulty` | Bot used by `train`: `passive`, `easy`, `medium`, `hard`. |

### Controls (you are the orange boxer)

| Key | Action |
| --- | --- |
| `W` `A` `S` `D` | Move up / left / down / right on screen |
| `J` | Jab (fast, long, weak, head) |
| `K` | Cross (straight power punch, head) |
| `L` | Hook (body) |
| `I` | Uppercut (short, slow, heavy; punishes a slip) |
| `Q` (hold) | High guard |
| `E` (hold) | Low guard |
| `Space` | Slip / duck |
| `Esc` | Quit |

A glove turns yellow while a punch is winding up and white while it can hit.
A green outline means a guard is up; a grey body means exhausted.

## How it works

```
boxing_ai/
  config.py       every tunable number, as named and validated fields
  geometry.py     vector helpers, swept hit test, ring ray casting
  strikes.py      frame data and trajectory of each punch
  boxer.py        one boxer: stamina, health, guards, punch state machine, gloves
  simulation.py   the match: movement, ropes, hit resolution, events
  actions.py      action space and decoding            (module 2)
  observation.py  observation vector                   (module 1)
  rewards.py      shaping reward                       (module 4)
  env.py          the Gymnasium environment
  controllers.py  scripted bot and the controller interface
  rendering.py    Pygame window and keyboard controller
  checkpoints.py  create / save / load the PPO model
  training.py     headless training against the bot
  sparring.py     real-time loop: live sparring and watch mode (module 5)
```

### Simulation

A deterministic kinematic simulation at 60 frames per second. Boxers move on
the ring floor; height exists where it matters (head level, body level,
ducking), so positions and glove motion are 3D vectors. Both corners use the
same `Boxer` class and the same rules. Boxers automatically turn to face each
other at a limited rate, except during a punch, when the direction is locked.

A punch moves its glove rest -> wind-up pose -> full extension -> rest. It can
only hit during the active frames, and hits are a swept test of the glove's
path against the target's head or body.

| Punch | Wind-up | Active | Recovery | Damage x | Stamina | Target | Reach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jab | 6 | 3 | 8 | 1.0 | 0.04 | head | 1.24 m |
| Cross | 10 | 3 | 14 | 2.0 | 0.07 | head | 1.29 m |
| Hook | 12 | 4 | 16 | 2.5 | 0.09 | body | 1.10 m |
| Uppercut | 14 | 4 | 20 | 3.25 | 0.11 | head | 0.99 m |

Base damage is 0.04 of full health, multiplied by the punch's damage factor.

* **High guard / low guard** remove 80% of the damage of punches to the head /
  body, and drain stamina while held.
* **Slip** is a committed 18-frame duck followed by a 24-frame cooldown. Jabs
  and crosses pass over a ducked head. Hooks still hit the body, and an
  uppercut rises through the ducked head for 1.5x damage.
* **Stamina** regenerates when standing still, more slowly when backing away,
  and is spent by advancing, circling, guarding, slipping, blocking and
  punching. A punch that hits nothing costs 50% extra.
* **Exhaustion**: at zero stamina the boxer is stunned for 120 frames. It
  cannot punch, guard or slip, takes full damage and moves at half speed.
* **Knockout** at zero health ends the round; otherwise the timer does.

### Observation (62 values, the agent's sensors)

Everything is relative to the boxer's own forward/left/up axes and scaled to
[-1, 1].

| Group | Size | Contents |
| --- | --- | --- |
| Self | 13 | health, stamina, guard (one-hot of 4), exhausted, punch phase (one-hot of 4), punch progress, slip readiness |
| Opponent | 15 | relative 3D position and velocity, heading vector, health, stamina, guard (one-hot), exhausted |
| Threat | 24 | per opponent glove: relative 3D position, velocity, acceleration; one wind-up flag per punch type; active-hitbox flag; punch progress |
| Ring | 10 | rope distance along 8 rays (lidar-style), distance to nearest corner, round time remaining |

### Action (`MultiDiscrete([9, 4, 5])`)

Three independent heads, chosen 30 times per second: locomotion (stand or one
of 8 directions), defense (none, high guard, low guard, slip) and offense
(none, jab, cross, hook, uppercut).

### Reward

| Term | Value |
| --- | --- |
| Clean punch landed | +5.0 x (impact speed / 20 m/s, capped at 1) |
| Punch slipped, or blocked by a guard raised in the last 0.5 s | +3.0 |
| Holding the centre (closer to it than the opponent, within 2 m) | +0.25 per second |
| Damage taken | -0.5 per health point (100 points = full health) |
| Stamina reaching zero | -10 |
| Passivity (no punch thrown within 1.5 m for more than 3 s) | -0.5 per second |
| Guard spam (more than 2 s of guarding without punching back) | -0.5 per second |
| Knockout | +20 for a win, -20 for a loss |
| Round timer runs out | +10 x (own health - opponent health) |

All weights are fields of `RewardConfig`. Each step's `info["reward_terms"]`
contains the value of every term, for tuning.

### Learning during live sparring

Stable-Baselines3's `learn()` updates the network whenever its buffer fills,
which would freeze the game mid-round. `OnlineLearner` (in `sparring.py`)
records every decision during the round and runs one standard PPO update in
the break, then a checkpoint is written. Rounds shorter than one mini-batch
(256 decisions, about 8.5 s) are carried over into the next update.
