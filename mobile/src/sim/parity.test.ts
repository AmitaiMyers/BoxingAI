/**
 * Proves the TypeScript simulation follows exactly the same rules as the
 * Python reference: recorded Python fights are replayed command by command
 * and every event and boxer state must match.
 *
 * Regenerate the recording with `python -m tools.export_for_mobile`.
 */

import trace from './__fixtures__/parityTrace.json';
import { Boxer, BoxerCommand } from './boxer';
import { GAME_CONFIG, HANDS } from './config';
import { BoxingMatch, FrameEvents } from './match';

/** Decimal places that must agree (float maths differs in the last bits). */
const NUMBER_PRECISION = 9;

interface RecordedGlove {
  position: number[];
  velocity: number[];
  acceleration: number[];
}

interface RecordedBoxer {
  position: number[];
  velocity: number[];
  facing: number;
  health: number;
  stamina: number;
  guard: string;
  strikeType: string | null;
  strikeFrame: number;
  strikePhase: string;
  slipFramesRemaining: number;
  slipCooldownFramesRemaining: number;
  exhaustedFramesRemaining: number;
  gloves: Record<string, RecordedGlove>;
}

interface RecordedFrame {
  commands: BoxerCommand[];
  events: FrameEvents;
  boxers: RecordedBoxer[] | null;
}

interface RecordedEpisode {
  start: { center: number[]; axisAngle: number };
  initialPositions: number[][];
  frames: RecordedFrame[];
  knockedOut: number[];
}

const episodes = (trace as unknown as { episodes: RecordedEpisode[] }).episodes;

function expectVector(actual: readonly number[], expected: number[], label: string): void {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, axis) => {
    // Accelerations reach thousands of m/s^2, so compare relative to size.
    const scale = Math.max(1, Math.abs(value));
    expect({ label, axis, value: actual[axis] / scale }).toEqual({
      label,
      axis,
      value: expect.closeTo(value / scale, NUMBER_PRECISION),
    });
  });
}

function expectBoxer(actual: Boxer, expected: RecordedBoxer, label: string): void {
  expect({
    guard: actual.guard,
    strikeType: actual.strikeType,
    strikeFrame: actual.strikeFrame,
    strikePhase: actual.strikePhase,
    slipFramesRemaining: actual.slipFramesRemaining,
    slipCooldownFramesRemaining: actual.slipCooldownFramesRemaining,
    exhaustedFramesRemaining: actual.exhaustedFramesRemaining,
  }).toEqual({
    guard: expected.guard,
    strikeType: expected.strikeType,
    strikeFrame: expected.strikeFrame,
    strikePhase: expected.strikePhase,
    slipFramesRemaining: expected.slipFramesRemaining,
    slipCooldownFramesRemaining: expected.slipCooldownFramesRemaining,
    exhaustedFramesRemaining: expected.exhaustedFramesRemaining,
  });
  expect(actual.health).toBeCloseTo(expected.health, NUMBER_PRECISION);
  expect(actual.stamina).toBeCloseTo(expected.stamina, NUMBER_PRECISION);
  expect(actual.facing).toBeCloseTo(expected.facing, NUMBER_PRECISION);
  expectVector(actual.position, expected.position, `${label} position`);
  expectVector(actual.velocity, expected.velocity, `${label} velocity`);
  for (const hand of HANDS) {
    const glove = actual.gloves[hand];
    const recorded = expected.gloves[hand];
    expectVector(glove.position, recorded.position, `${label} ${hand} glove position`);
    expectVector(glove.velocity, recorded.velocity, `${label} ${hand} glove velocity`);
    expectVector(glove.acceleration, recorded.acceleration, `${label} ${hand} glove acceleration`);
  }
}

function expectEvents(actual: FrameEvents, expected: FrameEvents): void {
  expect(actual.newlyExhausted).toEqual(expected.newlyExhausted);
  expect(actual.strikesStarted).toEqual(expected.strikesStarted);
  expect(actual.evades).toEqual(expected.evades);
  expect(actual.hits.length).toBe(expected.hits.length);
  expected.hits.forEach((hit, index) => {
    const { damage, impactSpeed, ...discrete } = actual.hits[index];
    expect(discrete).toEqual({
      attackerIndex: hit.attackerIndex,
      strikeType: hit.strikeType,
      wasBlocked: hit.wasBlocked,
      wasTimedBlock: hit.wasTimedBlock,
    });
    expect(damage).toBeCloseTo(hit.damage, NUMBER_PRECISION);
    expect(impactSpeed).toBeCloseTo(hit.impactSpeed, NUMBER_PRECISION);
  });
}

describe('TypeScript simulation matches the Python reference', () => {
  it('has recorded fights that exercise the interesting rules', () => {
    const frames = episodes.flatMap((episode) => episode.frames);
    const hits = frames.flatMap((frame) => frame.events.hits);
    expect(hits.some((hit) => hit.wasBlocked)).toBe(true);
    expect(hits.some((hit) => !hit.wasBlocked)).toBe(true);
    expect(frames.some((frame) => frame.events.evades.length > 0)).toBe(true);
    expect(frames.some((frame) => frame.events.newlyExhausted.length > 0)).toBe(true);
    expect(episodes.some((episode) => episode.knockedOut.length > 0)).toBe(true);
  });

  episodes.forEach((episode, episodeNumber) => {
    it(`replays episode ${episodeNumber} identically (${episode.frames.length} frames)`, () => {
      const match = new BoxingMatch(GAME_CONFIG);
      match.reset({
        center: [episode.start.center[0], episode.start.center[1]],
        axisAngle: episode.start.axisAngle,
      });
      episode.initialPositions.forEach((position, index) => {
        match.boxers[index].position = [position[0], position[1]];
      });

      episode.frames.forEach((frame, frameNumber) => {
        const events = match.step([frame.commands[0], frame.commands[1]]);
        expectEvents(events, frame.events);
        if (frame.boxers !== null) {
          frame.boxers.forEach((recorded, index) => {
            expectBoxer(match.boxers[index], recorded, `frame ${frameNumber} boxer ${index}`);
          });
        }
      });

      expect(match.knockedOut).toEqual(episode.knockedOut);
    });
  });
});
