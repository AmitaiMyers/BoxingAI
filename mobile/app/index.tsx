import { router } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { COLORS } from '../src/render/palette';
import { BOT_DIFFICULTIES, BotDifficulty } from '../src/sim/bot';

/** Home: pick a sparring partner. (The player's own learning boxer replaces this list in the next stage.) */
export default function Home() {
  const spar = (difficulty: BotDifficulty) => router.push({ pathname: '/spar', params: { difficulty } });

  return (
    <View style={styles.screen}>
      <Text style={styles.title}>BOXING AI</Text>
      <Text style={styles.subtitle}>Step into the gym and spar</Text>
      <View style={styles.row}>
        {BOT_DIFFICULTIES.map((difficulty) => (
          <Pressable
            key={difficulty}
            accessibilityRole="button"
            onPress={() => spar(difficulty)}
            style={({ pressed }) => [styles.button, pressed ? styles.pressed : null]}
          >
            <Text style={styles.buttonText}>{difficulty.toUpperCase()}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: COLORS.background,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 14,
  },
  title: {
    color: COLORS.text,
    fontSize: 44,
    fontWeight: '900',
    letterSpacing: 4,
  },
  subtitle: {
    color: COLORS.textMuted,
    fontSize: 16,
  },
  row: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 12,
  },
  button: {
    paddingHorizontal: 22,
    paddingVertical: 14,
    borderRadius: 14,
    backgroundColor: COLORS.accent,
  },
  pressed: {
    opacity: 0.7,
  },
  buttonText: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '800',
  },
});
