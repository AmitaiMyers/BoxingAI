import { create } from 'zustand';

import type { RoundOutcome } from '../engine/fightSession';

/**
 * What the fight screen's React parts (bars, timer, result) show.
 * The 3D scene writes here a few times per second, never per frame, so React
 * does not re-render at frame rate (same rule as the racer's stores).
 */

export type CameraMode = 'first' | 'shoulder';

export interface BoxerHud {
  readonly health: number;
  readonly stamina: number;
  readonly status: string;
}

export interface FightHud {
  readonly ai: BoxerHud;
  readonly player: BoxerHud;
  readonly secondsRemaining: number;
  readonly outcome: RoundOutcome;
}

const FRESH_BOXER: BoxerHud = { health: 1, stamina: 1, status: 'ready' };

interface FightState {
  hud: FightHud;
  cameraMode: CameraMode;
  setHud: (hud: FightHud) => void;
  toggleCamera: () => void;
}

export const useFightStore = create<FightState>((set) => ({
  hud: { ai: FRESH_BOXER, player: FRESH_BOXER, secondsRemaining: 0, outcome: 'ongoing' },
  cameraMode: 'first',
  setHud: (hud) => set({ hud }),
  toggleCamera: () => set((state) => ({ cameraMode: state.cameraMode === 'first' ? 'shoulder' : 'first' })),
}));
