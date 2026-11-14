/**
 * One sparring round between the player and the AI boxer.
 *
 * The session owns the simulation and paces it: the renderer calls
 * `advance(elapsedSeconds)` once per drawn frame and the session runs as many
 * fixed 60 Hz simulation frames as that time covers. Input arrives through
 * the `press...`/`set...` methods from touch controls or the keyboard.
 *
 * No React and no rendering here, so the same class runs in tests.
 */

import { BoxerController } from '../sim/bot';
import { Boxer, BoxerCommand, IDLE_COMMAND, StepDirection } from '../sim/boxer';
import { GameConfig, GuardState, StrikeType, frameSeconds } from '../sim/config';
import { assert } from '../sim/geometry';
import { BoxingMatch, FrameEvents } from '../sim/match';

/** Same seats as the Python reference: the learning boxer is 0, the player 1. */
export const AI_INDEX = 0;
export const PLAYER_INDEX = 1;

/**
 * A tapped punch or slip is remembered this long, so a tap made just before
 * the hands are free still comes out (standard fighting-game input buffer).
 */
const INPUT_BUFFER_FRAMES = 10;

/** After a stall (app in background, slow frame) never simulate more than this at once. */
const MAX_FRAMES_PER_ADVANCE = 6;

export type HeldGuard = Extract<GuardState, 'NONE' | 'HIGH' | 'LOW'>;

export type RoundOutcome = 'ongoing' | 'player_knockout_win' | 'ai_knockout_win' | 'double_knockout' | 'time_up';

export class FightSession {
  readonly match: BoxingMatch;

  private accumulatedSeconds = 0;
  private aiCommand: BoxerCommand = IDLE_COMMAND;

  private moveForward: StepDirection = 0;
  private moveStrafe: StepDirection = 0;
  private heldGuard: HeldGuard = 'NONE';
  private bufferedStrike: StrikeType | null = null;
  private strikeBufferFramesLeft = 0;
  private slipBufferFramesLeft = 0;

  constructor(
    private readonly config: GameConfig,
    private readonly opponent: BoxerController,
  ) {
    this.match = new BoxingMatch(config);
    // The player starts at the bottom of the ring looking "up" it.
    this.match.reset({ center: [0, 0], axisAngle: -Math.PI / 2 });
    this.opponent.reset();
  }

  get player(): Boxer {
    return this.match.boxers[PLAYER_INDEX];
  }

  get ai(): Boxer {
    return this.match.boxers[AI_INDEX];
  }

  get outcome(): RoundOutcome {
    const knockedOut = this.match.knockedOut;
    if (knockedOut.length === 2) {
      return 'double_knockout';
    }
    if (knockedOut.includes(AI_INDEX)) {
      return 'player_knockout_win';
    }
    if (knockedOut.includes(PLAYER_INDEX)) {
      return 'ai_knockout_win';
    }
    return this.match.isTimeUp ? 'time_up' : 'ongoing';
  }

  get secondsRemaining(): number {
    return this.match.timeRemainingFraction * this.config.timing.roundDurationSeconds;
  }

  // ------------------------------------------------------------------ input

  /** Footwork in the player's own frame: +forward is toward where they face, +strafe is left. */
  setMove(forward: StepDirection, strafe: StepDirection): void {
    this.moveForward = forward;
    this.moveStrafe = strafe;
  }

  /** A guard is held for as long as its button is down. */
  setGuard(guard: HeldGuard): void {
    this.heldGuard = guard;
  }

  pressStrike(strike: StrikeType): void {
    this.bufferedStrike = strike;
    this.strikeBufferFramesLeft = INPUT_BUFFER_FRAMES;
  }

  pressSlip(): void {
    this.slipBufferFramesLeft = INPUT_BUFFER_FRAMES;
  }

  // ---------------------------------------------------------------- pacing

  /**
   * Simulate the time that passed since the last call. Returns the events of
   * every frame simulated, for the renderer's effects and sounds.
   */
  advance(elapsedSeconds: number): FrameEvents[] {
    assert(elapsedSeconds >= 0, 'elapsedSeconds must be >= 0');
    const seconds = frameSeconds(this.config);
    this.accumulatedSeconds = Math.min(
      this.accumulatedSeconds + elapsedSeconds,
      MAX_FRAMES_PER_ADVANCE * seconds,
    );

    const events: FrameEvents[] = [];
    while (this.accumulatedSeconds >= seconds && !this.match.isOver) {
      this.accumulatedSeconds -= seconds;
      events.push(this.stepFrame());
    }
    return events;
  }

  private stepFrame(): FrameEvents {
    // The AI picks a new action every few frames, as it was trained to.
    if (this.match.frameCount % this.config.timing.decisionIntervalFrames === 0) {
      this.aiCommand = this.opponent.decide(this.match, AI_INDEX);
    }
    const events = this.match.step([this.aiCommand, this.playerCommand()]);
    this.consumeBufferedInput(events);
    return events;
  }

  private playerCommand(): BoxerCommand {
    return {
      forward: this.moveForward,
      strafe: this.moveStrafe,
      defense: this.slipBufferFramesLeft > 0 ? 'SLIP' : this.heldGuard,
      strike: this.strikeBufferFramesLeft > 0 ? this.bufferedStrike : null,
    };
  }

  private consumeBufferedInput(events: FrameEvents): void {
    const strikeCameOut = events.strikesStarted.includes(PLAYER_INDEX);
    this.strikeBufferFramesLeft = strikeCameOut ? 0 : Math.max(0, this.strikeBufferFramesLeft - 1);
    this.slipBufferFramesLeft = this.player.isSlipping ? 0 : Math.max(0, this.slipBufferFramesLeft - 1);
  }
}
