/**
 * Pure maths that turns simulation state into poses for the 3D scene.
 * No three.js objects here, so it is unit-tested in Node.
 *
 * Simulation space: X/Y on the ring floor, Z up.
 * Scene space (three.js): X right, Y up, Z toward the viewer.
 * The mapping is (x, y, z) -> (x, z, -y), which keeps "counter-clockwise from
 * +X" facing angles equal to a rotation about the scene's Y axis.
 */

import { Vec2, Vec3, headingVector, leftVector } from '../sim/geometry';

export type ScenePoint = readonly [number, number, number];

export function toScene(point: Vec3): ScenePoint {
  return [point[0], point[2], -point[1]];
}

/**
 * Scene position of a point given in a boxer's own frame: `forward` and
 * `left` metres from where it stands, `up` metres above the floor.
 */
export function bodyPoint(
  position: Vec2,
  facing: number,
  forward: number,
  left: number,
  up: number,
): ScenePoint {
  const heading = headingVector(facing);
  const side = leftVector(facing);
  return toScene([
    position[0] + heading[0] * forward + side[0] * left,
    position[1] + heading[1] * forward + side[1] * left,
    up,
  ]);
}

/** A direction in a boxer's own frame (forward, left, up) as a scene vector. */
export function bodyDirection(facing: number, forward: number, left: number, up: number): ScenePoint {
  return bodyPoint([0, 0], facing, forward, left, up);
}

export function addPoints(a: ScenePoint, b: ScenePoint): ScenePoint {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a: ScenePoint, b: ScenePoint): ScenePoint {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(vector: ScenePoint, factor: number): ScenePoint {
  return [vector[0] * factor, vector[1] * factor, vector[2] * factor];
}

function dot(a: ScenePoint, b: ScenePoint): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function distance(a: ScenePoint, b: ScenePoint): number {
  const offset = subtract(a, b);
  return Math.sqrt(dot(offset, offset));
}

/** Smallest length treated as non-zero when normalising. */
const EPSILON = 1e-6;

/**
 * Where the middle joint (elbow or knee) of a two-segment limb sits.
 *
 * `root` is the shoulder or hip and `tip` the glove or foot. The joint bends
 * toward `bendHint`. When the tip is further away than the limb is long, the
 * limb is drawn straight and stretched: the robots have telescopic arms, and
 * a punch in the simulation reaches further than a rigid arm would.
 */
export function limbJoint(
  root: ScenePoint,
  tip: ScenePoint,
  upperLength: number,
  lowerLength: number,
  bendHint: ScenePoint,
): ScenePoint {
  const span = subtract(tip, root);
  const reach = Math.sqrt(dot(span, span));
  if (reach < EPSILON) {
    return addPoints(root, scale(bendHint, upperLength));
  }
  const along = scale(span, 1 / reach);
  if (reach >= upperLength + lowerLength) {
    return addPoints(root, scale(along, reach * (upperLength / (upperLength + lowerLength))));
  }

  // Distance from the root to the joint's projection on the root-tip line.
  const projected = (upperLength * upperLength - lowerLength * lowerLength + reach * reach) / (2 * reach);
  const bendDistance = Math.sqrt(Math.max(0, upperLength * upperLength - projected * projected));

  // The bend direction is the hint with its component along the limb removed.
  const sideways = subtract(bendHint, scale(along, dot(bendHint, along)));
  const sidewaysLength = Math.sqrt(dot(sideways, sideways));
  const bend = sidewaysLength < EPSILON ? ([0, -1, 0] as ScenePoint) : scale(sideways, 1 / sidewaysLength);

  return addPoints(addPoints(root, scale(along, projected)), scale(bend, bendDistance));
}

/** Move `current` a fraction of the way to `target` (frame-rate independent smoothing). */
export function approach(current: number, target: number, ratePerSecond: number, seconds: number): number {
  return target + (current - target) * Math.exp(-ratePerSecond * seconds);
}
