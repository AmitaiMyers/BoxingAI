/**
 * One boxer: resources, stance, punch state machine and glove kinematics.
 * Mirrors `boxing_ai/boxer.py`; the same class drives the player and the AI.
 *
 * A frame is processed in three calls, driven by `BoxingMatch`:
 * 1. `beginFrame`   - turn the controller's command into state changes.
 * 2. `updateGloves` - after the match has moved the bodies, place the gloves
 *                     and derive their velocity and acceleration.
 * 3. `endFrame`     - settle the stamina bill and advance timers.
 * Hits are resolved by the match between steps 2 and 3.
 */

import {
  GameConfig,
  GloveOffset,
  GuardState,
  HAND_SIDE_SIGN,
  HANDS,
  Hand,
  StrikePhase,
  StrikeSpec,
  StrikeType,
  TargetZone,
  frameSeconds,
  strikePhaseAt,
  strikeTotalFrames,
} from './config';
import {
  Vec2,
  Vec3,
  ZERO_VEC2,
  ZERO_VEC3,
  add2,
  assert,
  headingVector,
  leftVector,
  lerp3,
  localToWorld,
  rotateTowards,
  scale2,
  scale3,
  subtract2,
  subtract3,
} from './geometry';

const FULL_RESOURCE = 1;
const EMPTY_RESOURCE = 0;

export type StepDirection = -1 | 0 | 1;

/**
 * What a controller asks its boxer to do this frame. `forward` and `strafe`
 * are in the boxer's local frame (+forward is where it faces, +strafe is to
 * its left).
 */
export interface BoxerCommand {
  readonly forward: StepDirection;
  readonly strafe: StepDirection;
  readonly defense: GuardState;
  readonly strike: StrikeType | null;
}

export const IDLE_COMMAND: BoxerCommand = { forward: 0, strafe: 0, defense: 'NONE', strike: null };

/** World-space kinematics of one glove. */
export interface GloveState {
  position: Vec3;
  previousPosition: Vec3;
  velocity: Vec3;
  acceleration: Vec3;
}

function isHeldGuard(guard: GuardState): boolean {
  return guard === 'HIGH' || guard === 'LOW';
}

/** [forward, left, height] of a glove offset for the hand on `sideSign`. */
function offsetToLocal(offset: GloveOffset, sideSign: number): Vec3 {
  return [offset.forward, offset.outward * sideSign, offset.height];
}

export class Boxer {
  position: Vec2 = ZERO_VEC2;
  velocity: Vec2 = ZERO_VEC2;
  facing = 0;

  health = FULL_RESOURCE;
  stamina = FULL_RESOURCE;

  guard: GuardState = 'NONE';
  guardHeldFrames = 0;

  strikeType: StrikeType | null = null;
  strikeFrame = 0;
  strikeConnected = false;
  strikeEvaded = false;

  slipFramesRemaining = 0;
  slipCooldownFramesRemaining = 0;
  exhaustedFramesRemaining = 0;

  /** One-frame flags read by the match to build its event list. */
  justExhausted = false;
  strikeStartedThisFrame = false;

  gloves!: Record<Hand, GloveState>;

  private locomotion: readonly [StepDirection, StepDirection] = [0, 0];
  private strikeStartOffset: Vec3 = ZERO_VEC3;
  private gloveLocal!: Record<Hand, Vec3>;

  constructor(private readonly config: GameConfig) {
    this.reset(ZERO_VEC2, 0);
  }

  // ------------------------------------------------------------------ setup

  reset(position: Vec2, facing: number): void {
    this.position = position;
    this.velocity = ZERO_VEC2;
    this.facing = facing;

    this.health = FULL_RESOURCE;
    this.stamina = FULL_RESOURCE;

    this.guard = 'NONE';
    this.guardHeldFrames = 0;

    this.strikeType = null;
    this.strikeFrame = 0;
    this.strikeConnected = false;
    this.strikeEvaded = false;

    this.slipFramesRemaining = 0;
    this.slipCooldownFramesRemaining = 0;
    this.exhaustedFramesRemaining = 0;

    this.justExhausted = false;
    this.strikeStartedThisFrame = false;

    this.locomotion = [0, 0];
    this.strikeStartOffset = ZERO_VEC3;
    this.gloveLocal = {
      lead: offsetToLocal(this.restingOffset(), HAND_SIDE_SIGN.lead),
      rear: offsetToLocal(this.restingOffset(), HAND_SIDE_SIGN.rear),
    };
    this.gloves = { lead: this.stillGlove('lead'), rear: this.stillGlove('rear') };
  }

  private stillGlove(hand: Hand): GloveState {
    const position = this.gloveWorldPosition(hand);
    return {
      position,
      previousPosition: position,
      velocity: ZERO_VEC3,
      acceleration: ZERO_VEC3,
    };
  }

  // ------------------------------------------------------------- properties

  get isStriking(): boolean {
    return this.strikeType !== null;
  }

  get isSlipping(): boolean {
    return this.slipFramesRemaining > 0;
  }

  get isExhausted(): boolean {
    return this.exhaustedFramesRemaining > 0;
  }

  get isKnockedOut(): boolean {
    return this.health <= EMPTY_RESOURCE;
  }

  /** True when the hands are free to start a punch or a slip. */
  get canStartAction(): boolean {
    return !(this.isExhausted || this.isStriking || this.isSlipping);
  }

  get strikeSpec(): StrikeSpec {
    assert(this.strikeType !== null, 'no strike is in progress');
    return this.config.combat.strikes[this.strikeType];
  }

  get strikePhase(): StrikePhase {
    if (this.strikeType === null) {
      return 'IDLE';
    }
    return strikePhaseAt(this.strikeSpec, this.strikeFrame);
  }

  /** Fraction of the current punch already performed (0 when idle). */
  get strikeProgress(): number {
    if (this.strikeType === null) {
      return 0;
    }
    return this.strikeFrame / strikeTotalFrames(this.strikeSpec);
  }

  get strikingGlove(): GloveState {
    return this.gloves[this.strikeSpec.hand];
  }

  get headHeight(): number {
    const body = this.config.body;
    return this.isSlipping ? body.duckedHeadHeight : body.standingHeadHeight;
  }

  get headCenter(): Vec3 {
    return [this.position[0], this.position[1], this.headHeight];
  }

  /** Where the head would be without the slip (used to credit evasions). */
  get standingHeadCenter(): Vec3 {
    return [this.position[0], this.position[1], this.config.body.standingHeadHeight];
  }

  get torsoCenter(): Vec3 {
    return [this.position[0], this.position[1], this.config.body.torsoHeight];
  }

  hurtboxCenter(zone: TargetZone): Vec3 {
    return zone === 'head' ? this.headCenter : this.torsoCenter;
  }

  // ---------------------------------------------------------- frame: step 1

  beginFrame(command: BoxerCommand, opponentPosition: Vec2): void {
    this.justExhausted = false;
    this.strikeStartedThisFrame = false;
    this.locomotion = [command.forward, command.strafe];

    if (this.canStartAction) {
      if (command.strike !== null) {
        this.startStrike(command.strike);
      } else if (command.defense === 'SLIP' && this.slipCooldownFramesRemaining === 0) {
        this.startSlip();
      }
    }

    this.updateGuard(command.defense);
    this.updateFacing(opponentPosition);
    this.velocity = this.commandedVelocity(command);
  }

  private startStrike(strikeType: StrikeType): void {
    this.strikeType = strikeType;
    this.strikeFrame = 0;
    this.strikeConnected = false;
    this.strikeEvaded = false;
    this.strikeStartOffset = this.gloveLocal[this.strikeSpec.hand];
    // Paying may empty the tank, which cancels the punch (see exhaust).
    this.spendStamina(this.strikeSpec.staminaCost);
    this.strikeStartedThisFrame = this.isStriking;
  }

  private startSlip(): void {
    this.slipFramesRemaining = this.config.defense.slipDurationFrames;
    this.spendStamina(this.config.stamina.slipCost);
  }

  private updateGuard(requested: GuardState): void {
    let newGuard: GuardState;
    if (this.isSlipping) {
      newGuard = 'SLIP';
    } else if (this.isExhausted || this.isStriking) {
      newGuard = 'NONE';
    } else if (isHeldGuard(requested)) {
      newGuard = requested;
    } else {
      newGuard = 'NONE';
    }
    this.guardHeldFrames = newGuard === this.guard ? this.guardHeldFrames + 1 : 0;
    this.guard = newGuard;
  }

  private updateFacing(opponentPosition: Vec2): void {
    // A boxer commits to the direction of a punch; otherwise it keeps
    // turning to face its opponent at a limited rate.
    if (this.isStriking) {
      return;
    }
    const toOpponent = subtract2(opponentPosition, this.position);
    if (toOpponent[0] === 0 && toOpponent[1] === 0) {
      return;
    }
    const targetFacing = Math.atan2(toOpponent[1], toOpponent[0]);
    const maxStep = this.config.body.turnRate * frameSeconds(this.config);
    this.facing = rotateTowards(this.facing, targetFacing, maxStep);
  }

  private commandedVelocity(command: BoxerCommand): Vec2 {
    const body = this.config.body;
    const forwardSpeed = command.forward > 0 ? body.forwardSpeed : body.backwardSpeed;
    let localVelocity: Vec2 = [command.forward * forwardSpeed, command.strafe * body.strafeSpeed];
    if (command.forward !== 0 && command.strafe !== 0) {
      localVelocity = scale2(localVelocity, 1 / Math.sqrt(2));
    }
    return scale2(localToWorld(localVelocity, this.facing), this.speedMultiplier());
  }

  private speedMultiplier(): number {
    const body = this.config.body;
    if (this.isExhausted) {
      return body.exhaustedSpeedMultiplier;
    }
    if (this.isSlipping) {
      return body.slippingSpeedMultiplier;
    }
    if (this.isStriking) {
      return body.strikingSpeedMultiplier;
    }
    if (isHeldGuard(this.guard)) {
      return body.guardingSpeedMultiplier;
    }
    return 1;
  }

  // ---------------------------------------------------------- frame: step 2

  updateGloves(): void {
    const seconds = frameSeconds(this.config);
    for (const hand of HANDS) {
      this.gloveLocal[hand] = this.nextGloveLocal(hand);
      const glove = this.gloves[hand];
      const newPosition = this.gloveWorldPosition(hand);
      const newVelocity = scale3(subtract3(newPosition, glove.position), 1 / seconds);
      glove.acceleration = scale3(subtract3(newVelocity, glove.velocity), 1 / seconds);
      glove.previousPosition = glove.position;
      glove.position = newPosition;
      glove.velocity = newVelocity;
    }
  }

  private nextGloveLocal(hand: Hand): Vec3 {
    if (this.isStriking && this.strikeSpec.hand === hand) {
      return this.strikeGloveLocal();
    }
    const target = offsetToLocal(this.restingOffset(), HAND_SIDE_SIGN[hand]);
    return lerp3(this.gloveLocal[hand], target, this.config.body.gloveBlendPerFrame);
  }

  /** Glove offset at the end of the current frame of the punch. */
  private strikeGloveLocal(): Vec3 {
    const spec = this.strikeSpec;
    const sideSign = HAND_SIDE_SIGN[spec.hand];
    const windup = offsetToLocal(spec.windupOffset, sideSign);
    const extended = offsetToLocal(spec.extendedOffset, sideSign);
    const rest = offsetToLocal(this.config.body.neutralGloveOffset, sideSign);

    const framesDone = this.strikeFrame + 1;
    if (framesDone <= spec.windupFrames) {
      return lerp3(this.strikeStartOffset, windup, framesDone / spec.windupFrames);
    }
    if (framesDone <= spec.windupFrames + spec.activeFrames) {
      return lerp3(windup, extended, (framesDone - spec.windupFrames) / spec.activeFrames);
    }
    const recoveryFramesDone = framesDone - spec.windupFrames - spec.activeFrames;
    return lerp3(extended, rest, recoveryFramesDone / spec.recoveryFrames);
  }

  private restingOffset(): GloveOffset {
    const body = this.config.body;
    if (this.isExhausted) {
      return body.exhaustedGloveOffset;
    }
    if (this.isSlipping) {
      return body.slipGloveOffset;
    }
    if (this.guard === 'HIGH') {
      return body.highGuardGloveOffset;
    }
    if (this.guard === 'LOW') {
      return body.lowGuardGloveOffset;
    }
    return body.neutralGloveOffset;
  }

  private gloveWorldPosition(hand: Hand): Vec3 {
    const [forward, left, height] = this.gloveLocal[hand];
    const planar = add2(
      add2(this.position, scale2(headingVector(this.facing), forward)),
      scale2(leftVector(this.facing), left),
    );
    return [planar[0], planar[1], height];
  }

  // ---------------------------------------------------- called by the match

  takeDamage(damage: number): void {
    assert(damage >= 0, 'damage must be >= 0');
    this.health = Math.max(EMPTY_RESOURCE, this.health - damage);
  }

  payBlockCost(): void {
    this.spendStamina(this.config.stamina.blockCost);
  }

  // ---------------------------------------------------------- frame: step 3

  endFrame(): void {
    // Settled first, while the state still describes the frame just played.
    this.settleStamina();
    this.advanceStrike();
    this.advanceSlip();
    this.advanceExhaustion();
  }

  private advanceStrike(): void {
    if (!this.isStriking) {
      return;
    }
    this.strikeFrame += 1;
    if (this.strikeFrame < strikeTotalFrames(this.strikeSpec)) {
      return;
    }
    const staminaCost = this.strikeSpec.staminaCost;
    const wasMissed = !this.strikeConnected;
    this.strikeType = null;
    this.strikeFrame = 0;
    if (wasMissed) {
      this.spendStamina(staminaCost * this.config.stamina.missedStrikeExtraCostFraction);
    }
  }

  private advanceSlip(): void {
    if (this.slipCooldownFramesRemaining > 0) {
      this.slipCooldownFramesRemaining -= 1;
    }
    if (!this.isSlipping) {
      return;
    }
    this.slipFramesRemaining -= 1;
    if (!this.isSlipping) {
      this.slipCooldownFramesRemaining = this.config.defense.slipCooldownFrames;
    }
  }

  private advanceExhaustion(): void {
    if (this.isExhausted) {
      this.exhaustedFramesRemaining -= 1;
    }
  }

  /** Per-frame stamina flow. Exactly one rule applies each frame. */
  private settleStamina(): void {
    const stamina = this.config.stamina;
    const seconds = frameSeconds(this.config);
    const [forward, strafe] = this.locomotion;
    if (this.isExhausted) {
      // Always recovers while stunned so a boxer can never get stuck at 0.
      this.regainStamina(stamina.idleRegenPerSecond * seconds);
    } else if (this.isStriking || this.isSlipping) {
      return; // Already paid up front; no recovery while committed.
    } else if (isHeldGuard(this.guard)) {
      this.spendStamina(stamina.guardCostPerSecond * seconds);
    } else if (forward === 0 && strafe === 0) {
      this.regainStamina(stamina.idleRegenPerSecond * seconds);
    } else if (forward < 0) {
      this.regainStamina(stamina.retreatRegenPerSecond * seconds);
    } else {
      this.spendStamina(stamina.moveCostPerSecond * seconds);
    }
  }

  // ---------------------------------------------------------------- stamina

  private regainStamina(amount: number): void {
    this.stamina = Math.min(FULL_RESOURCE, this.stamina + amount);
  }

  private spendStamina(amount: number): void {
    assert(amount >= 0, 'stamina cost must be >= 0');
    if (this.isExhausted) {
      return;
    }
    this.stamina = Math.max(EMPTY_RESOURCE, this.stamina - amount);
    if (this.stamina === EMPTY_RESOURCE) {
      this.exhaust();
    }
  }

  /** Stamina hit zero: hands drop and whatever was in progress stops. */
  private exhaust(): void {
    this.exhaustedFramesRemaining = this.config.stamina.exhaustionDurationFrames;
    this.justExhausted = true;
    this.strikeType = null;
    this.strikeFrame = 0;
    this.slipFramesRemaining = 0;
    this.guard = 'NONE';
    this.guardHeldFrames = 0;
  }
}
