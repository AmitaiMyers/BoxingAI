/**
 * Game rules and numbers.
 *
 * The values are NOT written here: they are loaded from `gameConfig.json`,
 * which is generated from the Python reference implementation by
 * `python -m tools.export_for_mobile`. Change a number in
 * `boxing_ai/config.py`, re-run the export, and both sides stay in sync.
 */

import exportedConfig from './gameConfig.json';
import { assert } from './geometry';

export const STRIKE_TYPES = ['JAB', 'CROSS', 'HOOK', 'UPPERCUT'] as const;
export type StrikeType = (typeof STRIKE_TYPES)[number];

export const GUARD_STATES = ['NONE', 'HIGH', 'LOW', 'SLIP'] as const;
export type GuardState = (typeof GUARD_STATES)[number];

export type StrikePhase = 'IDLE' | 'WINDUP' | 'ACTIVE' | 'RECOVERY';
export type TargetZone = 'head' | 'body';

/** LEAD is the left hand, REAR the right hand (orthodox stance). */
export const HANDS = ['lead', 'rear'] as const;
export type Hand = (typeof HANDS)[number];

/** +1 for the left side of the body, -1 for the right side. */
export const HAND_SIDE_SIGN: Record<Hand, number> = { lead: 1, rear: -1 };

/** Which guard protects which target zone. */
export const GUARD_FOR_ZONE: Record<TargetZone, GuardState> = { head: 'HIGH', body: 'LOW' };

/**
 * Where a glove sits relative to its owner, in the owner's local frame.
 * `outward` is the distance from the body's centre line toward the side of
 * the hand that owns the glove, so one offset describes both hands.
 */
export interface GloveOffset {
  readonly forward: number;
  readonly outward: number;
  readonly height: number;
}

/**
 * The glove travels rest -> `windupOffset` during the wind-up frames,
 * `windupOffset` -> `extendedOffset` during the active frames (the only
 * frames in which it can hit), then back to rest during the recovery frames.
 */
export interface StrikeSpec {
  readonly windupFrames: number;
  readonly activeFrames: number;
  readonly recoveryFrames: number;
  readonly damageMultiplier: number;
  readonly staminaCost: number;
  readonly targetZone: TargetZone;
  readonly hand: Hand;
  readonly windupOffset: GloveOffset;
  readonly extendedOffset: GloveOffset;
}

export interface TimingConfig {
  readonly framesPerSecond: number;
  readonly decisionIntervalFrames: number;
  readonly roundDurationSeconds: number;
}

export interface RingConfig {
  readonly halfSize: number;
  readonly startOffset: number;
  readonly startCenterJitter: number;
}

export interface BodyConfig {
  readonly bodyRadius: number;
  readonly headRadius: number;
  readonly torsoRadius: number;
  readonly gloveRadius: number;
  readonly standingHeadHeight: number;
  readonly duckedHeadHeight: number;
  readonly torsoHeight: number;
  readonly forwardSpeed: number;
  readonly backwardSpeed: number;
  readonly strafeSpeed: number;
  readonly turnRate: number;
  readonly strikingSpeedMultiplier: number;
  readonly guardingSpeedMultiplier: number;
  readonly slippingSpeedMultiplier: number;
  readonly exhaustedSpeedMultiplier: number;
  readonly neutralGloveOffset: GloveOffset;
  readonly highGuardGloveOffset: GloveOffset;
  readonly lowGuardGloveOffset: GloveOffset;
  readonly slipGloveOffset: GloveOffset;
  readonly exhaustedGloveOffset: GloveOffset;
  readonly gloveBlendPerFrame: number;
}

export interface DefenseConfig {
  readonly guardDamageReduction: number;
  readonly slipDurationFrames: number;
  readonly slipCooldownFrames: number;
  readonly slipUppercutDamageMultiplier: number;
  readonly timedBlockWindowFrames: number;
}

export interface StaminaConfig {
  readonly idleRegenPerSecond: number;
  readonly retreatRegenPerSecond: number;
  readonly moveCostPerSecond: number;
  readonly guardCostPerSecond: number;
  readonly slipCost: number;
  readonly blockCost: number;
  readonly missedStrikeExtraCostFraction: number;
  readonly exhaustionDurationFrames: number;
}

export interface CombatConfig {
  readonly baseStrikeDamage: number;
  readonly strikes: Record<StrikeType, StrikeSpec>;
}

export interface GameConfig {
  readonly timing: TimingConfig;
  readonly ring: RingConfig;
  readonly body: BodyConfig;
  readonly defense: DefenseConfig;
  readonly stamina: StaminaConfig;
  readonly combat: CombatConfig;
}

function validate(config: GameConfig): GameConfig {
  for (const strikeType of STRIKE_TYPES) {
    assert(strikeType in config.combat.strikes, `gameConfig.json has no spec for ${strikeType}`);
  }
  assert(config.timing.framesPerSecond > 0, 'framesPerSecond must be > 0');
  assert(config.ring.halfSize > config.body.bodyRadius, 'the ring is smaller than a boxer');
  return config;
}

export const GAME_CONFIG: GameConfig = validate(exportedConfig as GameConfig);

// ------------------------------------------------------------ derived values

export function frameSeconds(config: GameConfig): number {
  return 1 / config.timing.framesPerSecond;
}

export function decisionsPerSecond(config: GameConfig): number {
  return config.timing.framesPerSecond / config.timing.decisionIntervalFrames;
}

export function roundFrames(config: GameConfig): number {
  return Math.round(config.timing.roundDurationSeconds * config.timing.framesPerSecond);
}

export function strikeTotalFrames(spec: StrikeSpec): number {
  return spec.windupFrames + spec.activeFrames + spec.recoveryFrames;
}

/** Phase of a punch on its `frame`-th frame (0-based). */
export function strikePhaseAt(spec: StrikeSpec, frame: number): StrikePhase {
  assert(frame >= 0 && frame < strikeTotalFrames(spec), 'frame is outside the strike');
  if (frame < spec.windupFrames) {
    return 'WINDUP';
  }
  if (frame < spec.windupFrames + spec.activeFrames) {
    return 'ACTIVE';
  }
  return 'RECOVERY';
}

export function hurtboxRadius(config: GameConfig, zone: TargetZone): number {
  return zone === 'head' ? config.body.headRadius : config.body.torsoRadius;
}

/** Greatest distance between two boxers at which this punch can land. */
export function strikeReach(config: GameConfig, strikeType: StrikeType): number {
  const spec = config.combat.strikes[strikeType];
  return (
    spec.extendedOffset.forward + config.body.gloveRadius + hurtboxRadius(config, spec.targetZone)
  );
}
