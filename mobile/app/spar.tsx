import { router, useLocalSearchParams } from 'expo-router';
import { useMemo, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { FightSession, RoundOutcome } from '../src/engine/fightSession';
import { FightScene } from '../src/render/FightScene';
import { COLORS } from '../src/render/palette';
import { BOT_DIFFICULTIES, BOT_PROFILES, BotDifficulty, ScriptedBot, seededRandom } from '../src/sim/bot';
import { GAME_CONFIG } from '../src/sim/config';
import { useFightStore } from '../src/state/fightStore';
import { FightControls } from '../src/ui/FightControls';
import { FightHud } from '../src/ui/FightHud';

const RESULT_TEXT: Record<Exclude<RoundOutcome, 'ongoing'>, string> = {
  player_knockout_win: 'KNOCKOUT! YOU WIN',
  ai_knockout_win: 'KNOCKED OUT',
  double_knockout: 'DOUBLE KNOCKOUT',
  time_up: 'TIME!',
};

function parseDifficulty(value: string | string[] | undefined): BotDifficulty {
  const found = BOT_DIFFICULTIES.find((difficulty) => difficulty === value);
  if (found === undefined) {
    throw new Error(`Unknown sparring difficulty: ${String(value)}`);
  }
  return found;
}

/** One sparring round in the gym. */
export default function Spar() {
  const difficulty = parseDifficulty(useLocalSearchParams().difficulty);
  const [round, setRound] = useState(1);
  const outcome = useFightStore((state) => state.hud.outcome);

  // A new session per round; the round number also seeds the opponent so every round differs.
  const session = useMemo(
    () => new FightSession(GAME_CONFIG, new ScriptedBot(GAME_CONFIG, BOT_PROFILES[difficulty], seededRandom(round))),
    [difficulty, round],
  );
  const isOver = outcome !== 'ongoing' && session.match.isOver;

  return (
    <View style={styles.screen}>
      <FightScene key={round} session={session} config={GAME_CONFIG} />
      <FightHud />
      {isOver ? (
        <View style={styles.result}>
          <Text style={styles.resultText}>{RESULT_TEXT[outcome]}</Text>
          <View style={styles.resultButtons}>
            <Pressable accessibilityRole="button" style={styles.button} onPress={() => setRound(round + 1)}>
              <Text style={styles.buttonText}>NEXT ROUND</Text>
            </Pressable>
            <Pressable accessibilityRole="button" style={[styles.button, styles.secondary]} onPress={() => router.back()}>
              <Text style={styles.buttonText}>LEAVE</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <FightControls session={session} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  result: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 18,
  },
  resultText: {
    color: COLORS.text,
    fontSize: 40,
    fontWeight: '900',
    letterSpacing: 2,
  },
  resultButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    paddingHorizontal: 22,
    paddingVertical: 14,
    borderRadius: 14,
    backgroundColor: COLORS.accent,
  },
  secondary: {
    backgroundColor: COLORS.surfaceRaised,
  },
  buttonText: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '800',
  },
});
