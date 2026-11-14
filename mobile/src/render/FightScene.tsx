/**
 * The 3D view of a sparring round: gym, both robots, camera and hit effects.
 *
 * `FightDirector` runs inside the canvas once per drawn frame: it advances the
 * session, moves the camera, starts effects and publishes HUD values a few
 * times per second.
 */

import { Canvas, useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

import { AI_INDEX, FightSession } from '../engine/fightSession';
import { Boxer } from '../sim/boxer';
import { GameConfig } from '../sim/config';
import { FrameEvents, HitEvent, otherIndex } from '../sim/match';
import { CameraMode, useFightStore } from '../state/fightStore';
import { Gym } from './Gym';
import { COLORS } from './palette';
import { HitSparks } from './hitSparks';
import { RobotModel } from './robotModel';
import { ScenePoint, addPoints, approach, bodyDirection, bodyPoint, toScene } from './scenePose';

// -------------------------------------------------------------------- camera
const FIELD_OF_VIEW = 72;
const NEAR_PLANE = 0.04;
const FAR_PLANE = 60;
/** First person: the eye sits a little behind the head centre and looks slightly down, so the gloves are in view. */
const EYE_BACK_OFFSET = 0.28;
const EYE_UP_OFFSET = 0.08;
const EYE_LOOK_DROP = 0.16;
/** Over the shoulder: behind, to the right of and above the player. */
const SHOULDER_BACK = 1.5;
const SHOULDER_SIDE = -0.5;
const SHOULDER_HEIGHT = 1.95;
/** Knockout camera: a slow circle around the two robots. */
const KNOCKOUT_RADIUS = 3.4;
const KNOCKOUT_HEIGHT = 1.5;
const KNOCKOUT_TURN_PER_SECOND = 0.35;
/** How fast the camera follows its target position, per second. */
const CAMERA_FOLLOW_RATE = 18;
const EYE_HEIGHT_RATE = 22;

// ------------------------------------------------------------------- effects
/** Camera shake in metres per unit of health lost, and how fast it dies away. */
const SHAKE_PER_DAMAGE = 1.6;
const BLOCKED_SHAKE = 0.015;
const SHAKE_DECAY_RATE = 9;
const HUD_UPDATES_PER_SECOND = 10;

function describe(boxer: Boxer): string {
  if (boxer.isKnockedOut) {
    return 'knocked out';
  }
  if (boxer.isExhausted) {
    return 'exhausted';
  }
  if (boxer.isStriking) {
    return (boxer.strikeType as string).toLowerCase();
  }
  if (boxer.isSlipping) {
    return 'slipping';
  }
  if (boxer.guard === 'HIGH') {
    return 'high guard';
  }
  if (boxer.guard === 'LOW') {
    return 'low guard';
  }
  return 'ready';
}

function FightDirector({ session, config }: { session: FightSession; config: GameConfig }) {
  const cameraMode = useFightStore((state) => state.cameraMode);
  // The robots and sparks are plain three.js objects, built once per round and posed every frame.
  const stage = useMemo(
    () => ({
      aiRobot: new RobotModel(config, { body: COLORS.aiBody, trim: COLORS.aiTrim }),
      playerRobot: new RobotModel(config, { body: COLORS.playerBody, trim: COLORS.playerTrim }),
      sparks: new HitSparks(),
    }),
    [config],
  );
  useEffect(
    () => () => {
      stage.aiRobot.dispose();
      stage.playerRobot.dispose();
      stage.sparks.dispose();
    },
    [stage],
  );
  const shake = useRef(0);
  const eyeHeight = useRef(config.body.standingHeadHeight);
  const knockoutAngle = useRef(0);
  const secondsSinceHud = useRef(1);
  const cameraTarget = useRef(new THREE.Vector3());
  const lookTarget = useRef(new THREE.Vector3());

  function startSpark(hit: HitEvent): void {
    // A flash on the player's own head would fill the screen in first person; the camera shake says it instead.
    const isOwnHead = hit.attackerIndex === AI_INDEX && cameraMode === 'first' && session.match.knockedOut.length === 0;
    if (isOwnHead) {
      return;
    }
    const defender = session.match.boxers[otherIndex(hit.attackerIndex)];
    const zone = config.combat.strikes[hit.strikeType].targetZone;
    stage.sparks.start(toScene(defender.hurtboxCenter(zone)), hit.wasBlocked);
  }

  function applyEvents(frames: FrameEvents[]): void {
    for (const frame of frames) {
      for (const hit of frame.hits) {
        startSpark(hit);
        if (hit.attackerIndex === AI_INDEX) {
          shake.current += hit.wasBlocked ? BLOCKED_SHAKE : hit.damage * SHAKE_PER_DAMAGE;
        }
      }
    }
  }

  function cameraPose(mode: CameraMode | 'knockout', deltaSeconds: number): { eye: ScenePoint; look: ScenePoint } {
    const { player, ai } = session;
    if (mode === 'knockout') {
      knockoutAngle.current += KNOCKOUT_TURN_PER_SECOND * deltaSeconds;
      const middle = bodyPoint(
        [(player.position[0] + ai.position[0]) / 2, (player.position[1] + ai.position[1]) / 2],
        0,
        0,
        0,
        config.body.torsoHeight,
      );
      const around = bodyDirection(knockoutAngle.current, KNOCKOUT_RADIUS, 0, KNOCKOUT_HEIGHT - config.body.torsoHeight);
      return { eye: addPoints(middle, around), look: middle };
    }
    if (mode === 'shoulder') {
      return {
        eye: bodyPoint(player.position, player.facing, -SHOULDER_BACK, SHOULDER_SIDE, SHOULDER_HEIGHT),
        look: bodyPoint(ai.position, ai.facing, 0, 0, config.body.torsoHeight + 0.2),
      };
    }
    eyeHeight.current = approach(eyeHeight.current, player.headHeight, EYE_HEIGHT_RATE, deltaSeconds);
    const eyeY = eyeHeight.current + EYE_UP_OFFSET;
    return {
      eye: bodyPoint(player.position, player.facing, -EYE_BACK_OFFSET, 0, eyeY),
      look: bodyPoint(player.position, player.facing, 1, 0, eyeY - EYE_LOOK_DROP),
    };
  }

  function updateCamera(camera: THREE.Camera, deltaSeconds: number): void {
    const isKnockout = session.match.knockedOut.length > 0;
    const { eye, look } = cameraPose(isKnockout ? 'knockout' : cameraMode, deltaSeconds);
    cameraTarget.current.set(eye[0], eye[1], eye[2]);
    lookTarget.current.set(look[0], look[1], look[2]);

    // First person must stay glued to the head; the outside cameras glide.
    const follow = cameraMode === 'first' && !isKnockout ? 1 : 1 - Math.exp(-CAMERA_FOLLOW_RATE * deltaSeconds);
    camera.position.lerp(cameraTarget.current, follow);

    shake.current = approach(shake.current, 0, SHAKE_DECAY_RATE, deltaSeconds);
    // Two different rhythms so the shake does not move along a straight line.
    const time = session.match.frameCount;
    camera.position.x += Math.sin(time * 1.7) * shake.current;
    camera.position.y += Math.cos(time * 2.3) * shake.current;
    camera.lookAt(lookTarget.current);
  }

  function publishHud(deltaSeconds: number): void {
    secondsSinceHud.current += deltaSeconds;
    if (secondsSinceHud.current < 1 / HUD_UPDATES_PER_SECOND) {
      return;
    }
    secondsSinceHud.current = 0;
    const { player, ai } = session;
    useFightStore.getState().setHud({
      ai: { health: ai.health, stamina: ai.stamina, status: describe(ai) },
      player: { health: player.health, stamina: player.stamina, status: describe(player) },
      secondsRemaining: session.secondsRemaining,
      outcome: session.outcome,
    });
  }

  useFrame((frame, deltaSeconds) => {
    applyEvents(session.advance(deltaSeconds));
    const isOutsideCamera = cameraMode !== 'first' || session.match.knockedOut.length > 0;
    stage.aiRobot.pose(session.ai, true, deltaSeconds);
    stage.playerRobot.pose(session.player, isOutsideCamera, deltaSeconds);
    stage.sparks.update(deltaSeconds);
    updateCamera(frame.camera, deltaSeconds);
    publishHud(deltaSeconds);
  });

  return (
    <>
      <primitive object={stage.aiRobot.group} />
      <primitive object={stage.playerRobot.group} />
      <primitive object={stage.sparks.group} />
    </>
  );
}

export function FightScene({ session, config }: { session: FightSession; config: GameConfig }) {
  return (
    <Canvas
      shadows
      camera={{ fov: FIELD_OF_VIEW, near: NEAR_PLANE, far: FAR_PLANE }}
      gl={{ antialias: true }}
      onCreated={({ scene }) => {
        scene.background = new THREE.Color(COLORS.gymCeiling);
        scene.fog = new THREE.Fog(COLORS.gymCeiling, 12, 30);
      }}
    >
      <Gym config={config} />
      <FightDirector session={session} config={config} />
    </Canvas>
  );
}
