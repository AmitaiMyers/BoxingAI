/**
 * Rule-based sparring partner. Mirrors `ScriptedBot` in
 * `boxing_ai/controllers.py`. It is the opponent until the trained neural
 * network is wired in, and stays useful afterwards as a practice dummy.
 */

import { Boxer, BoxerCommand, StepDirection } from './boxer';
import {
  GUARD_FOR_ZONE,
  GameConfig,
  GuardState,
  STRIKE_TYPES,
  StrikeType,
  decisionsPerSecond,
  strikeReach,
} from './config';
import { assert } from './geometry';
import { BoxerIndex, BoxingMatch, otherIndex } from './match';

/** Anything that can decide what a boxer does: a script, a human, a network. */
export interface BoxerController {
  /** Called at the start of every round. */
  reset(): void;
  /** Choose the command for `match.boxers[boxerIndex]`. */
  decide(match: BoxingMatch, boxerIndex: BoxerIndex): BoxerCommand;
}

/** Personality of the scripted bot. */
export interface BotProfile {
  /** How often it punches while the opponent is in range. */
  readonly strikesPerSecond: number;
  /** Chance that it defends against any given incoming punch. */
  readonly defenseProbability: number;
  /** Of the head punches it defends, the fraction it slips instead of blocks. */
  readonly slipPreference: number;
  /** Frames of the opponent's punch it needs to see before it can react. */
  readonly reactionDelayFrames: number;
  /** Distance it tries to keep from the opponent. */
  readonly preferredDistance: number;
  /** It backs off to recover when stamina falls below this... */
  readonly restBelowStamina: number;
  /** ...and comes back once stamina has climbed above this. */
  readonly resumeAboveStamina: number;
}

export const BOT_DIFFICULTIES = ['passive', 'easy', 'medium', 'hard'] as const;
export type BotDifficulty = (typeof BOT_DIFFICULTIES)[number];

export const BOT_PROFILES: Record<BotDifficulty, BotProfile> = {
  // A walking heavy bag: approaches, rarely punches, never defends.
  passive: {
    strikesPerSecond: 0.3,
    defenseProbability: 0,
    slipPreference: 0,
    reactionDelayFrames: 0,
    preferredDistance: 0.9,
    restBelowStamina: 0.15,
    resumeAboveStamina: 0.5,
  },
  easy: {
    strikesPerSecond: 0.8,
    defenseProbability: 0.2,
    slipPreference: 0.2,
    reactionDelayFrames: 10,
    preferredDistance: 0.9,
    restBelowStamina: 0.2,
    resumeAboveStamina: 0.55,
  },
  medium: {
    strikesPerSecond: 1.5,
    defenseProbability: 0.5,
    slipPreference: 0.3,
    reactionDelayFrames: 6,
    preferredDistance: 0.9,
    restBelowStamina: 0.25,
    resumeAboveStamina: 0.6,
  },
  hard: {
    strikesPerSecond: 2.5,
    defenseProbability: 0.8,
    slipPreference: 0.4,
    reactionDelayFrames: 3,
    preferredDistance: 0.85,
    restBelowStamina: 0.3,
    resumeAboveStamina: 0.65,
  },
};

/** The bot only reacts to punches thrown from within reach plus this margin. */
const THREAT_RANGE_MARGIN = 0.3;
/** It is happy anywhere within this distance of its preferred range. */
const DISTANCE_TOLERANCE = 0.15;
/** How often, on average, it changes the direction it circles in. */
const STRAFE_CHANGES_PER_SECOND = 0.7;
const STRAFE_CHOICES: readonly StepDirection[] = [-1, 0, 1];

/** Returns a number in [0, 1). */
export type RandomSource = () => number;

/**
 * Small seedable random generator (mulberry32), so a fight can be replayed
 * exactly from its seed.
 */
export function seededRandom(seed: number): RandomSource {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(choices: readonly T[], random: RandomSource): T {
  assert(choices.length > 0, 'cannot pick from an empty list');
  return choices[Math.floor(random() * choices.length)];
}

export class ScriptedBot implements BoxerController {
  private readonly reach: Record<StrikeType, number>;
  private isResting = false;
  private strafe: StepDirection = 0;
  private reactedToCurrentStrike = false;
  private plannedDefense: GuardState | null = null;

  constructor(
    private readonly config: GameConfig,
    private readonly profile: BotProfile,
    private readonly random: RandomSource,
  ) {
    assert(
      profile.restBelowStamina < profile.resumeAboveStamina,
      'resume threshold must be above the rest threshold',
    );
    this.reach = {
      JAB: strikeReach(config, 'JAB'),
      CROSS: strikeReach(config, 'CROSS'),
      HOOK: strikeReach(config, 'HOOK'),
      UPPERCUT: strikeReach(config, 'UPPERCUT'),
    };
  }

  reset(): void {
    this.isResting = false;
    this.strafe = 0;
    this.reactedToCurrentStrike = false;
    this.plannedDefense = null;
  }

  decide(match: BoxingMatch, boxerIndex: BoxerIndex): BoxerCommand {
    const me = match.boxers[boxerIndex];
    const opponent = match.boxers[otherIndex(boxerIndex)];
    const distance = match.distanceBetweenBoxers();

    if (me.isExhausted) {
      return { forward: -1, strafe: 0, defense: 'NONE', strike: null };
    }

    const defense = this.reactToThreat(opponent, distance);
    if (defense !== null) {
      return { forward: 0, strafe: 0, defense, strike: null };
    }

    this.updateResting(me);
    if (this.isResting) {
      const isCrowded = distance < this.profile.preferredDistance + DISTANCE_TOLERANCE;
      return { forward: isCrowded ? -1 : 0, strafe: 0, defense: 'NONE', strike: null };
    }

    return {
      forward: this.approachDirection(distance),
      strafe: this.circleDirection(),
      defense: 'NONE',
      strike: this.chooseStrike(distance),
    };
  }

  // ---------------------------------------------------------------- defense

  private reactToThreat(opponent: Boxer, distance: number): GuardState | null {
    const phase = opponent.strikePhase;
    const isThreatening = phase === 'WINDUP' || phase === 'ACTIVE';
    if (!isThreatening) {
      this.reactedToCurrentStrike = false;
      this.plannedDefense = null;
      return null;
    }

    const incoming = opponent.strikeType as StrikeType;
    const hasSeenEnough = opponent.strikeFrame >= this.profile.reactionDelayFrames;
    const inDanger = distance <= this.reach[incoming] + THREAT_RANGE_MARGIN;
    if (!this.reactedToCurrentStrike && hasSeenEnough && inDanger) {
      // One roll per incoming punch, not one per decision.
      this.reactedToCurrentStrike = true;
      this.plannedDefense = this.pickDefense(incoming);
    }
    return this.plannedDefense;
  }

  private pickDefense(incoming: StrikeType): GuardState | null {
    if (this.random() >= this.profile.defenseProbability) {
      return null;
    }
    const zone = this.config.combat.strikes[incoming].targetZone;
    const canSlip = zone === 'head' && incoming !== 'UPPERCUT';
    if (canSlip && this.random() < this.profile.slipPreference) {
      return 'SLIP';
    }
    return GUARD_FOR_ZONE[zone];
  }

  // ----------------------------------------------------------------- pacing

  private updateResting(me: Boxer): void {
    if (me.stamina < this.profile.restBelowStamina) {
      this.isResting = true;
    } else if (me.stamina > this.profile.resumeAboveStamina) {
      this.isResting = false;
    }
  }

  // ---------------------------------------------------------------- offense

  private chooseStrike(distance: number): StrikeType | null {
    const inRange = STRIKE_TYPES.filter((strike) => distance <= this.reach[strike]);
    if (inRange.length === 0) {
      return null;
    }
    if (this.random() >= this.perDecision(this.profile.strikesPerSecond)) {
      return null;
    }
    return pick(inRange, this.random);
  }

  // --------------------------------------------------------------- movement

  private approachDirection(distance: number): StepDirection {
    if (distance > this.profile.preferredDistance + DISTANCE_TOLERANCE) {
      return 1;
    }
    if (distance < this.profile.preferredDistance - DISTANCE_TOLERANCE) {
      return -1;
    }
    return 0;
  }

  private circleDirection(): StepDirection {
    if (this.random() < this.perDecision(STRAFE_CHANGES_PER_SECOND)) {
      this.strafe = pick(STRAFE_CHOICES, this.random);
    }
    return this.strafe;
  }

  /** Probability of an event with the given rate on a single decision. */
  private perDecision(ratePerSecond: number): number {
    return ratePerSecond / decisionsPerSecond(this.config);
  }
}
