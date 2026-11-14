/** Health and stamina bars, the round clock and the camera switch. */

import { Pressable, StyleSheet, Text, View } from 'react-native';

import { COLORS } from '../render/palette';
import { BoxerHud, useFightStore } from '../state/fightStore';

const BAR_WIDTH = 190;
const SECONDS_PER_MINUTE = 60;

function Bar({ fraction, color }: { fraction: number; color: string }) {
  return (
    <View style={styles.barTrack}>
      <View style={[styles.barFill, { width: `${Math.round(fraction * 100)}%`, backgroundColor: color }]} />
    </View>
  );
}

function BoxerPanel({ name, color, hud, alignRight }: { name: string; color: string; hud: BoxerHud; alignRight: boolean }) {
  return (
    <View style={alignRight ? styles.panelRight : null}>
      <Text style={[styles.name, { color }]}>{name}</Text>
      <Bar fraction={hud.health} color={COLORS.health} />
      <Bar fraction={hud.stamina} color={COLORS.stamina} />
      <Text style={styles.status}>{hud.status}</Text>
    </View>
  );
}

function formatClock(seconds: number): string {
  const whole = Math.ceil(seconds);
  const minutes = Math.floor(whole / SECONDS_PER_MINUTE);
  return `${minutes}:${String(whole % SECONDS_PER_MINUTE).padStart(2, '0')}`;
}

export function FightHud() {
  const hud = useFightStore((state) => state.hud);
  const cameraMode = useFightStore((state) => state.cameraMode);
  const toggleCamera = useFightStore((state) => state.toggleCamera);

  return (
    <View pointerEvents="box-none" style={styles.layer}>
      <BoxerPanel name="YOU" color={COLORS.playerBody} hud={hud.player} alignRight={false} />
      <View style={styles.center}>
        <Text style={styles.clock}>{formatClock(hud.secondsRemaining)}</Text>
        <Pressable accessibilityRole="button" onPress={toggleCamera} style={styles.cameraButton}>
          <Text style={styles.cameraText}>{cameraMode === 'first' ? 'CAMERA: EYES' : 'CAMERA: SHOULDER'}</Text>
        </Pressable>
      </View>
      <BoxerPanel name="AI BOXER" color={COLORS.aiBody} hud={hud.ai} alignRight />
    </View>
  );
}

const styles = StyleSheet.create({
  layer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 28,
    paddingTop: 14,
  },
  panelRight: {
    alignItems: 'flex-end',
  },
  name: {
    fontSize: 14,
    fontWeight: '800',
    marginBottom: 4,
  },
  barTrack: {
    width: BAR_WIDTH,
    height: 10,
    borderRadius: 5,
    backgroundColor: COLORS.barTrack,
    marginBottom: 4,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
  },
  status: {
    color: COLORS.textMuted,
    fontSize: 12,
  },
  center: {
    alignItems: 'center',
  },
  clock: {
    color: COLORS.text,
    fontSize: 26,
    fontWeight: '800',
  },
  cameraButton: {
    marginTop: 4,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
    backgroundColor: 'rgba(16, 19, 27, 0.55)',
  },
  cameraText: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '700',
  },
});
