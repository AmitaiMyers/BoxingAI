import { describe, expect, it } from 'vitest';

import { BoxerController } from '../sim/bot';
import { BoxerCommand, IDLE_COMMAND } from '../sim/boxer';
import { GAME_CONFIG, frameSeconds, roundFrames, strikeTotalFrames } from '../sim/config';
import { FightSession, PLAYER_INDEX } from './fightSession';

const FRAME = frameSeconds(GAME_CONFIG);

/** An opponent that always does the same thing. */
function fixedOpponent(command: BoxerCommand): BoxerController {
  return { reset: () => undefined, decide: () => command };
}

function newSession(command: BoxerCommand = IDLE_COMMAND): FightSession {
  return new FightSession(GAME_CONFIG, fixedOpponent(command));
}

/** Advance exactly `frames` simulation frames. */
function run(session: FightSession, frames: number): void {
  for (let frame = 0; frame < frames; frame += 1) {
    // A hair over one frame, so rounding never drops a frame.
    session.advance(FRAME * 1.0001);
  }
}

describe('FightSession', () => {
  it('simulates as many fixed frames as the elapsed time covers', () => {
    const session = newSession();
    expect(session.advance(FRAME * 3.5)).toHaveLength(3);
    // The leftover half frame is kept and completes with the next half.
    expect(session.advance(FRAME * 0.6)).toHaveLength(1);
    expect(session.match.frameCount).toBe(4);
  });

  it('limits catch-up after a long stall', () => {
    const session = newSession();
    expect(session.advance(5).length).toBeLessThanOrEqual(6);
  });

  it('a tapped punch comes out once, not repeatedly', () => {
    const session = newSession();
    session.pressStrike('JAB');
    run(session, 1);
    expect(session.player.strikeType).toBe('JAB');

    run(session, strikeTotalFrames(GAME_CONFIG.combat.strikes.JAB) + 5);
    expect(session.player.isStriking).toBe(false);
  });

  it('buffers a punch tapped while the hands are still busy', () => {
    const session = newSession();
    const jabFrames = strikeTotalFrames(GAME_CONFIG.combat.strikes.JAB);
    session.pressStrike('JAB');
    run(session, jabFrames - 3);
    session.pressStrike('CROSS');
    run(session, 4);
    expect(session.player.strikeType).toBe('CROSS');
  });

  it('forgets a buffered punch that could not come out in time', () => {
    const session = newSession();
    session.pressStrike('UPPERCUT');
    run(session, 2);
    session.pressStrike('JAB');
    run(session, strikeTotalFrames(GAME_CONFIG.combat.strikes.UPPERCUT) + 2);
    expect(session.player.isStriking).toBe(false);
  });

  it('holds a guard while set and slips on a tap', () => {
    const session = newSession();
    session.setGuard('HIGH');
    run(session, 3);
    expect(session.player.guard).toBe('HIGH');

    session.setGuard('NONE');
    session.pressSlip();
    run(session, 1);
    expect(session.player.isSlipping).toBe(true);
    run(session, GAME_CONFIG.defense.slipDurationFrames + 2);
    expect(session.player.isSlipping).toBe(false);
  });

  it('moves the player toward the opponent on forward input', () => {
    const session = newSession();
    const before = session.match.distanceBetweenBoxers();
    session.setMove(1, 0);
    run(session, 20);
    expect(session.match.distanceBetweenBoxers()).toBeLessThan(before);
  });

  it('ends on the bell and reports the outcome', () => {
    const session = newSession();
    expect(session.outcome).toBe('ongoing');
    run(session, roundFrames(GAME_CONFIG));
    expect(session.outcome).toBe('time_up');
    expect(session.advance(1)).toHaveLength(0);
  });

  it('reports a knockout of the player', () => {
    const session = newSession({ forward: 1, strafe: 0, defense: 'NONE', strike: 'JAB' });
    session.match.boxers[PLAYER_INDEX].health = 0.01;
    run(session, 600);
    expect(session.outcome).toBe('ai_knockout_win');
  });
});
