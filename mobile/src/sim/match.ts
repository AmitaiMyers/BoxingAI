/**
 * The match: two boxers, a ring, and the rules for when a punch lands.
 * Mirrors `boxing_ai/simulation.py`.
 *
 * `BoxingMatch` knows nothing about rendering, touch input or who controls
 * the boxers. It takes two commands per frame and reports what happened as a
 * `FrameEvents` record, which the 3D view turns into animation and effects.
 */

import { Boxer, BoxerCommand } from './boxer';
import {
  GUARD_FOR_ZONE,
  GameConfig,
  StrikeType,
  TargetZone,
  frameSeconds,
  hurtboxRadius,
  roundFrames,
} from './config';
import {
  Vec2,
  Vec3,
  add2,
  assert,
  clamp,
  headingVector,
  length2,
  length3,
  scale2,
  segmentPointDistance,
  subtract2,
  subtract3,
} from './geometry';

export const BOXER_COUNT = 2;
export type BoxerIndex = 0 | 1;

export function otherIndex(boxerIndex: BoxerIndex): BoxerIndex {
  return boxerIndex === 0 ? 1 : 0;
}

/** A punch made contact (it may still have been blocked). */
export interface HitEvent {
  readonly attackerIndex: BoxerIndex;
  readonly strikeType: StrikeType;
  readonly damage: number;
  /** Speed of the glove relative to the defender at contact, in m/s. */
  readonly impactSpeed: number;
  readonly wasBlocked: boolean;
  /** Blocked by a guard that was raised in reaction, not held indefinitely. */
  readonly wasTimedBlock: boolean;
}

/** A punch that would have hit a standing head was slipped. */
export interface EvadeEvent {
  readonly defenderIndex: BoxerIndex;
  readonly strikeType: StrikeType;
}

export interface FrameEvents {
  readonly hits: readonly HitEvent[];
  readonly evades: readonly EvadeEvent[];
  readonly newlyExhausted: readonly BoxerIndex[];
  readonly strikesStarted: readonly BoxerIndex[];
}

/** Where a round begins: boxers stand on opposite sides of `center`. */
export interface StartPositions {
  readonly center: Vec2;
  /** Direction from the centre toward boxer 1, in radians. */
  readonly axisAngle: number;
}

const BOXER_INDICES: readonly BoxerIndex[] = [0, 1];

export class BoxingMatch {
  readonly boxers: readonly [Boxer, Boxer];
  frameCount = 0;

  constructor(readonly config: GameConfig) {
    this.boxers = [new Boxer(config), new Boxer(config)];
    this.reset({ center: [0, 0], axisAngle: 0 });
  }

  reset(start: StartPositions): void {
    const offset = scale2(headingVector(start.axisAngle), this.config.ring.startOffset);
    // Each boxer starts facing the other.
    this.boxers[0].reset(subtract2(start.center, offset), start.axisAngle);
    this.boxers[1].reset(add2(start.center, offset), start.axisAngle + Math.PI);
    this.frameCount = 0;
  }

  // ------------------------------------------------------------ round state

  get knockedOut(): readonly BoxerIndex[] {
    return BOXER_INDICES.filter((index) => this.boxers[index].isKnockedOut);
  }

  get isTimeUp(): boolean {
    return this.frameCount >= roundFrames(this.config);
  }

  get isOver(): boolean {
    return this.knockedOut.length > 0 || this.isTimeUp;
  }

  get timeRemainingFraction(): number {
    return Math.max(0, 1 - this.frameCount / roundFrames(this.config));
  }

  distanceBetweenBoxers(): number {
    return length2(subtract2(this.boxers[1].position, this.boxers[0].position));
  }

  // ------------------------------------------------------------------- step

  step(commands: readonly [BoxerCommand, BoxerCommand]): FrameEvents {
    assert(!this.isOver, 'the round is over; call reset() first');

    // Positions are read before anyone moves so that turn order is fair.
    const positions: readonly [Vec2, Vec2] = [this.boxers[0].position, this.boxers[1].position];
    for (const index of BOXER_INDICES) {
      this.boxers[index].beginFrame(commands[index], positions[otherIndex(index)]);
    }

    this.moveBoxers();
    for (const boxer of this.boxers) {
      boxer.updateGloves();
    }

    const hits: HitEvent[] = [];
    const evades: EvadeEvent[] = [];
    for (const attackerIndex of BOXER_INDICES) {
      this.resolveStrike(attackerIndex, hits, evades);
    }

    const strikesStarted = BOXER_INDICES.filter(
      (index) => this.boxers[index].strikeStartedThisFrame,
    );
    for (const boxer of this.boxers) {
      boxer.endFrame();
    }
    this.frameCount += 1;

    return {
      hits,
      evades,
      // Read last: blocking and missing can both empty the tank.
      newlyExhausted: BOXER_INDICES.filter((index) => this.boxers[index].justExhausted),
      strikesStarted,
    };
  }

  // --------------------------------------------------------------- movement

  private moveBoxers(): void {
    const seconds = frameSeconds(this.config);
    for (const boxer of this.boxers) {
      boxer.position = add2(boxer.position, scale2(boxer.velocity, seconds));
    }
    this.keepInsideRopes();
    this.separateBodies();
    // Separation can push a cornered boxer back out; the ropes win.
    this.keepInsideRopes();
  }

  private keepInsideRopes(): void {
    const limit = this.config.ring.halfSize - this.config.body.bodyRadius;
    for (const boxer of this.boxers) {
      boxer.position = [
        clamp(boxer.position[0], -limit, limit),
        clamp(boxer.position[1], -limit, limit),
      ];
    }
  }

  private separateBodies(): void {
    const [first, second] = this.boxers;
    const minimumDistance = 2 * this.config.body.bodyRadius;
    const offset = subtract2(second.position, first.position);
    const distance = length2(offset);
    if (distance >= minimumDistance) {
      return;
    }
    const direction = distance > 0 ? scale2(offset, 1 / distance) : headingVector(first.facing);
    const correction = scale2(direction, 0.5 * (minimumDistance - distance));
    first.position = subtract2(first.position, correction);
    second.position = add2(second.position, correction);
  }

  // --------------------------------------------------------- hit resolution

  private resolveStrike(attackerIndex: BoxerIndex, hits: HitEvent[], evades: EvadeEvent[]): void {
    const attacker = this.boxers[attackerIndex];
    if (attacker.strikePhase !== 'ACTIVE' || attacker.strikeConnected) {
      return;
    }

    const defenderIndex = otherIndex(attackerIndex);
    const defender = this.boxers[defenderIndex];
    const zone = attacker.strikeSpec.targetZone;

    if (this.gloveReaches(attacker, defender.hurtboxCenter(zone), zone)) {
      hits.push(this.landStrike(attackerIndex));
      return;
    }

    const slippedIt =
      zone === 'head' &&
      defender.isSlipping &&
      !attacker.strikeEvaded &&
      this.gloveReaches(attacker, defender.standingHeadCenter, zone);
    if (slippedIt) {
      attacker.strikeEvaded = true;
      evades.push({ defenderIndex, strikeType: attacker.strikeType as StrikeType });
    }
  }

  /** Swept test: did the glove pass within contact distance this frame? */
  private gloveReaches(attacker: Boxer, target: Vec3, zone: TargetZone): boolean {
    const glove = attacker.strikingGlove;
    const distance = segmentPointDistance(glove.previousPosition, glove.position, target);
    return distance <= this.config.body.gloveRadius + hurtboxRadius(this.config, zone);
  }

  private landStrike(attackerIndex: BoxerIndex): HitEvent {
    const attacker = this.boxers[attackerIndex];
    const defender = this.boxers[otherIndex(attackerIndex)];
    const spec = attacker.strikeSpec;
    const strikeType = attacker.strikeType as StrikeType;
    const defense = this.config.defense;

    // An exhausted boxer has no guard, so mitigation is off automatically.
    const wasBlocked = defender.guard === GUARD_FOR_ZONE[spec.targetZone];
    const wasTimedBlock = wasBlocked && defender.guardHeldFrames < defense.timedBlockWindowFrames;

    let damage = this.config.combat.baseStrikeDamage * spec.damageMultiplier;
    if (wasBlocked) {
      damage *= 1 - defense.guardDamageReduction;
    }
    if (defender.isSlipping && strikeType === 'UPPERCUT') {
      damage *= defense.slipUppercutDamageMultiplier;
    }

    const defenderVelocity: Vec3 = [defender.velocity[0], defender.velocity[1], 0];
    const impactSpeed = length3(subtract3(attacker.strikingGlove.velocity, defenderVelocity));

    attacker.strikeConnected = true;
    defender.takeDamage(damage);
    if (wasBlocked) {
      defender.payBlockCost();
    }

    return { attackerIndex, strikeType, damage, impactSpeed, wasBlocked, wasTimedBlock };
  }
}
