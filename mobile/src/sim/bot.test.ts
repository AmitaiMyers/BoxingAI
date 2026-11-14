import { BOT_DIFFICULTIES, BOT_PROFILES, ScriptedBot, seededRandom } from './bot';
import { GAME_CONFIG, roundFrames } from './config';
import { BoxingMatch } from './match';

const SEED = 7;

interface FightResult {
  frames: number;
  hits: number;
  health: [number, number];
}

/** Two scripted bots fight one full round. */
function fight(firstDifficulty: keyof typeof BOT_PROFILES, seed: number): FightResult {
  const match = new BoxingMatch(GAME_CONFIG);
  const first = new ScriptedBot(GAME_CONFIG, BOT_PROFILES[firstDifficulty], seededRandom(seed));
  const second = new ScriptedBot(GAME_CONFIG, BOT_PROFILES.medium, seededRandom(seed + 1));
  let hits = 0;
  while (!match.isOver) {
    const commands = [first.decide(match, 0), second.decide(match, 1)] as const;
    hits += match.step(commands).hits.length;
  }
  return {
    frames: match.frameCount,
    hits,
    health: [match.boxers[0].health, match.boxers[1].health],
  };
}

describe('ScriptedBot', () => {
  it('seededRandom is repeatable and stays in [0, 1)', () => {
    const first = seededRandom(SEED);
    const second = seededRandom(SEED);
    for (let draw = 0; draw < 1000; draw += 1) {
      const value = first();
      expect(value).toBe(second());
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it.each(BOT_DIFFICULTIES)('%s bot fights a full round and lands punches', (difficulty) => {
    const result = fight(difficulty, SEED);
    expect(result.frames).toBeLessThanOrEqual(roundFrames(GAME_CONFIG));
    expect(result.hits).toBeGreaterThan(0);
  });

  it('the same seed replays the same fight', () => {
    expect(fight('hard', SEED)).toEqual(fight('hard', SEED));
  });

  it('a harder bot does better against the same opponent', () => {
    const seeds = [1, 2, 3, 4, 5, 6, 7, 8];
    const margin = (difficulty: keyof typeof BOT_PROFILES) =>
      seeds
        .map((seed) => fight(difficulty, seed))
        .reduce((total, result) => total + result.health[0] - result.health[1], 0);
    expect(margin('hard')).toBeGreaterThan(margin('passive'));
  });
});
