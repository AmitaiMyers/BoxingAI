/**
 * On-screen fight controls: a movement stick under the left thumb and punch,
 * guard and slip buttons under the right. (Swipe gestures are the next
 * control scheme; this one stays available as a setting.)
 *
 * On the web preview the keyboard works too: W/A/S/D move, J/K/L/I punch,
 * Q and E guard high and low, Space slips, C switches camera.
 */

import { useEffect, useRef, useState } from 'react';
import { GestureResponderEvent, Platform, Pressable, StyleSheet, Text, View } from 'react-native';

import { FightSession, HeldGuard } from '../engine/fightSession';
import { StepDirection } from '../sim/boxer';
import { StrikeType } from '../sim/config';
import { COLORS } from '../render/palette';
import { useFightStore } from '../state/fightStore';

const STICK_SIZE = 132;
const KNOB_SIZE = 54;
const STICK_RADIUS = STICK_SIZE / 2;
/** The thumb must move this far from the centre (as a fraction of the radius) before the boxer steps. */
const STICK_DEAD_ZONE = 0.3;
/** A direction counts toward an axis when it is within 67.5 degrees of it, giving 8 even sectors. */
const AXIS_THRESHOLD = 0.38;
const BUTTON_SIZE = 62;

function toStep(component: number): StepDirection {
  if (component > AXIS_THRESHOLD) {
    return 1;
  }
  if (component < -AXIS_THRESHOLD) {
    return -1;
  }
  return 0;
}

function MoveStick({ session }: { session: FightSession }) {
  const [knob, setKnob] = useState({ x: 0, y: 0 });

  function follow(event: GestureResponderEvent): void {
    const dx = event.nativeEvent.locationX - STICK_RADIUS;
    const dy = event.nativeEvent.locationY - STICK_RADIUS;
    const length = Math.sqrt(dx * dx + dy * dy);
    const clamped = Math.min(length, STICK_RADIUS);
    const unitX = length === 0 ? 0 : dx / length;
    const unitY = length === 0 ? 0 : dy / length;
    setKnob({ x: unitX * clamped, y: unitY * clamped });
    if (length < STICK_RADIUS * STICK_DEAD_ZONE) {
      session.setMove(0, 0);
      return;
    }
    // Up on the stick is toward the opponent; right on the stick is a step to the right (strafe -1).
    session.setMove(toStep(-unitY), toStep(-unitX));
  }

  function release(): void {
    setKnob({ x: 0, y: 0 });
    session.setMove(0, 0);
  }

  return (
    <View
      style={styles.stick}
      onStartShouldSetResponder={() => true}
      onMoveShouldSetResponder={() => true}
      onResponderGrant={follow}
      onResponderMove={follow}
      onResponderRelease={release}
      onResponderTerminate={release}
    >
      <View pointerEvents="none" style={[styles.knob, { transform: [{ translateX: knob.x }, { translateY: knob.y }] }]} />
    </View>
  );
}

function ActionButton({
  label,
  color,
  onPressIn,
  onPressOut,
}: {
  label: string;
  color: string;
  onPressIn: () => void;
  onPressOut?: () => void;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      onPressIn={onPressIn}
      onPressOut={onPressOut}
      style={({ pressed }) => [styles.action, { borderColor: color }, pressed ? styles.actionPressed : null]}
    >
      <Text style={styles.actionText}>{label}</Text>
    </Pressable>
  );
}

const STRIKE_KEYS: Record<string, StrikeType> = { j: 'JAB', k: 'CROSS', l: 'HOOK', i: 'UPPERCUT' };
const GUARD_KEYS: Record<string, HeldGuard> = { q: 'HIGH', e: 'LOW' };
const MOVE_KEYS = ['w', 'a', 's', 'd'];

/** Keyboard control for the web preview (and for checking the game on a PC). */
function useKeyboard(session: FightSession): void {
  const held = useRef(new Set<string>());

  useEffect(() => {
    if (Platform.OS !== 'web') {
      return undefined;
    }
    function applyMovement(): void {
      const keys = held.current;
      const forward = (keys.has('w') ? 1 : 0) - (keys.has('s') ? 1 : 0);
      const strafe = (keys.has('a') ? 1 : 0) - (keys.has('d') ? 1 : 0);
      session.setMove(forward as StepDirection, strafe as StepDirection);
    }
    function onKeyDown(event: KeyboardEvent): void {
      const key = event.key.toLowerCase();
      if (event.repeat) {
        return;
      }
      if (key in STRIKE_KEYS) {
        session.pressStrike(STRIKE_KEYS[key] as StrikeType);
      } else if (key in GUARD_KEYS) {
        session.setGuard(GUARD_KEYS[key] as HeldGuard);
      } else if (key === ' ') {
        session.pressSlip();
      } else if (key === 'c') {
        useFightStore.getState().toggleCamera();
      } else if (MOVE_KEYS.includes(key)) {
        held.current.add(key);
        applyMovement();
      }
    }
    function onKeyUp(event: KeyboardEvent): void {
      const key = event.key.toLowerCase();
      if (key in GUARD_KEYS) {
        session.setGuard('NONE');
      } else if (MOVE_KEYS.includes(key)) {
        held.current.delete(key);
        applyMovement();
      }
    }
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [session]);
}

export function FightControls({ session }: { session: FightSession }) {
  useKeyboard(session);
  const releaseGuard = () => session.setGuard('NONE');

  return (
    <View pointerEvents="box-none" style={styles.layer}>
      <MoveStick session={session} />

      <View pointerEvents="box-none" style={styles.actions}>
        <View style={styles.row}>
          <ActionButton label="HIGH" color={COLORS.success} onPressIn={() => session.setGuard('HIGH')} onPressOut={releaseGuard} />
          <ActionButton label="UPPER" color={COLORS.gold} onPressIn={() => session.pressStrike('UPPERCUT')} />
          <ActionButton label="CROSS" color={COLORS.danger} onPressIn={() => session.pressStrike('CROSS')} />
        </View>
        <View style={styles.row}>
          <ActionButton label="LOW" color={COLORS.success} onPressIn={() => session.setGuard('LOW')} onPressOut={releaseGuard} />
          <ActionButton label="HOOK" color={COLORS.gold} onPressIn={() => session.pressStrike('HOOK')} />
          <ActionButton label="JAB" color={COLORS.danger} onPressIn={() => session.pressStrike('JAB')} />
        </View>
        <View style={styles.row}>
          <ActionButton label="SLIP" color={COLORS.neonCyan} onPressIn={() => session.pressSlip()} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  layer: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    paddingHorizontal: 28,
    paddingBottom: 22,
  },
  stick: {
    width: STICK_SIZE,
    height: STICK_SIZE,
    borderRadius: STICK_RADIUS,
    borderWidth: 2,
    borderColor: COLORS.border,
    backgroundColor: 'rgba(16, 19, 27, 0.45)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  knob: {
    width: KNOB_SIZE,
    height: KNOB_SIZE,
    borderRadius: KNOB_SIZE / 2,
    backgroundColor: 'rgba(242, 244, 248, 0.55)',
  },
  actions: {
    alignItems: 'flex-end',
    gap: 8,
  },
  row: {
    flexDirection: 'row',
    gap: 8,
  },
  action: {
    width: BUTTON_SIZE,
    height: BUTTON_SIZE,
    borderRadius: BUTTON_SIZE / 2,
    borderWidth: 2,
    backgroundColor: 'rgba(16, 19, 27, 0.55)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionPressed: {
    backgroundColor: 'rgba(242, 244, 248, 0.35)',
  },
  actionText: {
    color: COLORS.text,
    fontSize: 12,
    fontWeight: '700',
  },
});
