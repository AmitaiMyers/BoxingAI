/**
 * A sparring robot built from simple shapes and posed straight from the
 * simulation. Plain three.js, no React.
 *
 * Nothing is animated from clips: the gloves are drawn exactly where the
 * simulation's hit detection has them, the head is exactly as high as the
 * simulated head (so a slip visibly ducks under a punch), and elbows and
 * knees are solved from those positions. What you see is what can hit you.
 */

import * as THREE from 'three';

import { Boxer } from '../sim/boxer';
import { GameConfig, HAND_SIDE_SIGN, HANDS, Hand, StrikePhase } from '../sim/config';
import { COLORS } from './palette';
import { ScenePoint, approach, bodyDirection, bodyPoint, limbJoint, toScene } from './scenePose';

// --------------------------------------------------------------- proportions
const HEAD_SIZE = 0.24;
const NECK_DROP = 0.17;
const SHOULDER_DROP = 0.27;
const SHOULDER_HALF_WIDTH = 0.23;
const TORSO_WIDTH = 0.4;
const TORSO_DEPTH = 0.24;
const HIP_DROP = 0.66;
const HIP_HALF_WIDTH = 0.11;
const FOOT_HALF_WIDTH = 0.14;
const FOOT_HEIGHT = 0.06;
const FOOT_LENGTH = 0.26;
const FOOT_WIDTH = 0.11;
const UPPER_ARM_LENGTH = 0.3;
const FOREARM_LENGTH = 0.32;
const THIGH_LENGTH = 0.5;
const SHIN_LENGTH = 0.5;
const LIMB_RADIUS = 0.045;
const JOINT_RADIUS = 0.06;
const VISOR_WIDTH = 0.19;
const VISOR_HEIGHT = 0.06;
const ANTENNA_LENGTH = 0.1;
const GUARD_RING_RADIUS = 0.2;
const GUARD_RING_FORWARD = 0.12;

/** How fast the body follows a change of head height (ducking), per second. */
const CROUCH_RATE = 22;
/** Glow strength of the visor and of a glove that can hit right now. */
const BRIGHT_GLOW = 2.2;
const SOFT_GLOW = 0.5;
const DIM_GLOW = 0.2;

const GLOVE_COLORS: Record<StrikePhase, string> = {
  IDLE: COLORS.glove,
  RECOVERY: COLORS.glove,
  WINDUP: COLORS.gloveWindup,
  ACTIVE: COLORS.gloveActive,
};

const UP = new THREE.Vector3(0, 1, 0);
const scratchStart = new THREE.Vector3();
const scratchEnd = new THREE.Vector3();
const scratchDirection = new THREE.Vector3();

function placePoint(object: THREE.Object3D, point: ScenePoint): void {
  object.position.set(point[0], point[1], point[2]);
}

/** Stretch a unit-height mesh so that it joins two points. */
function placeSegment(mesh: THREE.Mesh, start: ScenePoint, end: ScenePoint): void {
  scratchStart.set(start[0], start[1], start[2]);
  scratchEnd.set(end[0], end[1], end[2]);
  scratchDirection.subVectors(scratchEnd, scratchStart);
  mesh.scale.set(1, scratchDirection.length(), 1);
  mesh.position.addVectors(scratchStart, scratchEnd).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(UP, scratchDirection.normalize());
}

function metal(color: string, metalness: number, roughness: number): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, metalness, roughness });
}

function shadowMesh(geometry: THREE.BufferGeometry, material: THREE.Material): THREE.Mesh {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  return mesh;
}

/** An arm or a leg: two segments and the ball joint between them. */
class Limb {
  private readonly upper: THREE.Mesh;
  private readonly lower: THREE.Mesh;
  private readonly joint: THREE.Mesh;

  constructor(
    parent: THREE.Group,
    color: string,
    private readonly upperLength: number,
    private readonly lowerLength: number,
  ) {
    this.upper = shadowMesh(new THREE.CylinderGeometry(LIMB_RADIUS, LIMB_RADIUS, 1, 10), metal(color, 0.55, 0.4));
    this.lower = shadowMesh(
      new THREE.CylinderGeometry(LIMB_RADIUS * 0.8, LIMB_RADIUS, 1, 10),
      metal(COLORS.metal, 0.85, 0.3),
    );
    this.joint = new THREE.Mesh(new THREE.SphereGeometry(JOINT_RADIUS, 12, 10), metal(COLORS.joint, 0.6, 0.5));
    parent.add(this.upper, this.lower, this.joint);
  }

  pose(root: ScenePoint, tip: ScenePoint, bendHint: ScenePoint): void {
    const middle = limbJoint(root, tip, this.upperLength, this.lowerLength, bendHint);
    placeSegment(this.upper, root, middle);
    placeSegment(this.lower, middle, tip);
    placePoint(this.joint, middle);
  }
}

export interface RobotColors {
  readonly body: string;
  readonly trim: string;
}

export class RobotModel {
  /** Add this to the scene. */
  readonly group = new THREE.Group();

  /** Everything except arms and gloves; hidden in first person so the camera can sit in the head. */
  private readonly body = new THREE.Group();
  private readonly torso: THREE.Mesh;
  private readonly head = new THREE.Group();
  private readonly visor: THREE.MeshStandardMaterial;
  private readonly guardRing: THREE.Mesh;
  private readonly gloves: Record<Hand, THREE.Mesh>;
  private readonly gloveMaterials: Record<Hand, THREE.MeshStandardMaterial>;
  private readonly arms: Record<Hand, Limb>;
  private readonly legs: Record<Hand, Limb>;
  private readonly feet: Record<Hand, THREE.Mesh>;
  private headHeight: number;

  constructor(
    private readonly config: GameConfig,
    colors: RobotColors,
  ) {
    this.headHeight = config.body.standingHeadHeight;

    this.torso = shadowMesh(new THREE.BoxGeometry(TORSO_DEPTH, 1, TORSO_WIDTH), metal(colors.body, 0.5, 0.4));

    this.visor = new THREE.MeshStandardMaterial({ color: COLORS.visor, emissive: COLORS.visor });
    const skull = shadowMesh(new THREE.BoxGeometry(HEAD_SIZE, HEAD_SIZE, HEAD_SIZE), metal(colors.trim, 0.6, 0.35));
    // The visor is on the robot's front, which is +X in its own frame.
    const visorMesh = new THREE.Mesh(new THREE.BoxGeometry(0.01, VISOR_HEIGHT, VISOR_WIDTH), this.visor);
    visorMesh.position.set(HEAD_SIZE / 2 + 0.002, 0.02, 0);
    const antenna = new THREE.Mesh(
      new THREE.CylinderGeometry(0.008, 0.008, ANTENNA_LENGTH, 6),
      metal(COLORS.metal, 0.9, 0.3),
    );
    antenna.position.set(0, HEAD_SIZE / 2 + ANTENNA_LENGTH / 2, 0);
    this.head.add(skull, visorMesh, antenna);

    this.guardRing = new THREE.Mesh(
      new THREE.TorusGeometry(GUARD_RING_RADIUS, 0.012, 8, 28),
      new THREE.MeshBasicMaterial({ color: COLORS.guardGlow }),
    );

    const armGroup = new THREE.Group();
    this.arms = {
      lead: new Limb(armGroup, colors.body, UPPER_ARM_LENGTH, FOREARM_LENGTH),
      rear: new Limb(armGroup, colors.body, UPPER_ARM_LENGTH, FOREARM_LENGTH),
    };
    this.legs = {
      lead: new Limb(this.body, colors.trim, THIGH_LENGTH, SHIN_LENGTH),
      rear: new Limb(this.body, colors.trim, THIGH_LENGTH, SHIN_LENGTH),
    };
    this.gloveMaterials = { lead: this.gloveMaterial(), rear: this.gloveMaterial() };
    this.gloves = { lead: this.gloveMesh('lead'), rear: this.gloveMesh('rear') };
    this.feet = { lead: this.footMesh(), rear: this.footMesh() };

    this.body.add(this.torso, this.head, this.feet.lead, this.feet.rear);
    this.group.add(this.body, armGroup, this.gloves.lead, this.gloves.rear, this.guardRing);
  }

  private gloveMaterial(): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({ color: COLORS.glove, roughness: 0.45 });
  }

  private gloveMesh(hand: Hand): THREE.Mesh {
    return shadowMesh(new THREE.SphereGeometry(this.config.body.gloveRadius, 18, 14), this.gloveMaterials[hand]);
  }

  private footMesh(): THREE.Mesh {
    return shadowMesh(new THREE.BoxGeometry(FOOT_LENGTH, FOOT_HEIGHT, FOOT_WIDTH), metal(COLORS.joint, 0.5, 0.5));
  }

  /** Pose the robot to match `boxer`. Call once per drawn frame. */
  pose(boxer: Boxer, showBody: boolean, deltaSeconds: number): void {
    // The simulation changes head height in one frame; the body follows quickly but smoothly.
    this.headHeight = approach(this.headHeight, boxer.headHeight, CROUCH_RATE, deltaSeconds);
    const headY = this.headHeight;
    const { position, facing } = boxer;

    this.body.visible = showBody;
    placePoint(this.head, bodyPoint(position, facing, 0, 0, headY));
    this.head.rotation.y = facing;
    const neck = bodyPoint(position, facing, 0, 0, headY - NECK_DROP);
    const hips = bodyPoint(position, facing, 0, 0, headY - HIP_DROP);
    placeSegment(this.torso, hips, neck);
    // The torso stands upright; turn its front toward where the boxer faces.
    this.torso.rotation.set(0, facing, 0);

    const forward = bodyDirection(facing, 1, 0, 0);
    for (const hand of HANDS) {
      const side = HAND_SIDE_SIGN[hand];
      const glove = toScene(boxer.gloves[hand].position);
      placePoint(this.gloves[hand], glove);
      this.colorGlove(boxer, hand);

      const shoulder = bodyPoint(position, facing, 0, side * SHOULDER_HALF_WIDTH, headY - SHOULDER_DROP);
      // Elbows point down and away from the body.
      this.arms[hand].pose(shoulder, glove, bodyDirection(facing, -0.2, side * 0.6, -1));

      const hip = bodyPoint(position, facing, 0, side * HIP_HALF_WIDTH, headY - HIP_DROP);
      const ankle = bodyPoint(position, facing, 0, side * FOOT_HALF_WIDTH, FOOT_HEIGHT);
      // Knees bend forward, which is what lowers the body in a slip.
      this.legs[hand].pose(hip, ankle, forward);
      placePoint(this.feet[hand], bodyPoint(position, facing, FOOT_LENGTH / 4, side * FOOT_HALF_WIDTH, FOOT_HEIGHT / 2));
      this.feet[hand].rotation.y = facing;
    }

    const visorColor = boxer.isExhausted ? COLORS.visorExhausted : COLORS.visor;
    this.visor.color.set(visorColor);
    this.visor.emissive.set(visorColor);
    this.visor.emissiveIntensity = boxer.isExhausted ? DIM_GLOW : BRIGHT_GLOW;

    const isGuarding = boxer.guard === 'HIGH' || boxer.guard === 'LOW';
    // In first person the player's own ring would sit right in front of the camera.
    this.guardRing.visible = isGuarding && showBody;
    const ringHeight = boxer.guard === 'HIGH' ? headY : this.config.body.torsoHeight;
    placePoint(this.guardRing, bodyPoint(position, facing, GUARD_RING_FORWARD, 0, ringHeight));
    this.guardRing.rotation.set(0, facing + Math.PI / 2, 0);
  }

  private colorGlove(boxer: Boxer, hand: Hand): void {
    const isThisHandStriking = boxer.isStriking && boxer.strikeSpec.hand === hand;
    const phase: StrikePhase = isThisHandStriking ? boxer.strikePhase : 'IDLE';
    const material = this.gloveMaterials[hand];
    const isLit = phase === 'WINDUP' || phase === 'ACTIVE';
    material.color.set(GLOVE_COLORS[phase]);
    material.emissive.set(isLit ? GLOVE_COLORS[phase] : '#000000');
    material.emissiveIntensity = phase === 'ACTIVE' ? BRIGHT_GLOW : SOFT_GLOW;
  }

  /** Free the GPU memory of every shape and material. Call when the scene is removed. */
  dispose(): void {
    this.group.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.geometry.dispose();
        (object.material as THREE.Material).dispose();
      }
    });
  }
}
