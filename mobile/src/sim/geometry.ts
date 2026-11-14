/**
 * Pure geometry helpers. Mirrors `boxing_ai/geometry.py`.
 *
 * Conventions:
 * - The ring floor is the world XY plane, centred on the origin. Z is height.
 * - A boxer's facing is an angle in radians, counter-clockwise from world +X.
 * - A boxer's local frame has +forward along its facing and +left 90 degrees
 *   counter-clockwise from it.
 */

export type Vec2 = readonly [number, number];
export type Vec3 = readonly [number, number, number];

export const ZERO_VEC2: Vec2 = [0, 0];
export const ZERO_VEC3: Vec3 = [0, 0, 0];

const FULL_TURN = 2 * Math.PI;

export function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

// ----------------------------------------------------------------- vectors

export function add2(a: Vec2, b: Vec2): Vec2 {
  return [a[0] + b[0], a[1] + b[1]];
}

export function subtract2(a: Vec2, b: Vec2): Vec2 {
  return [a[0] - b[0], a[1] - b[1]];
}

export function scale2(vector: Vec2, factor: number): Vec2 {
  return [vector[0] * factor, vector[1] * factor];
}

export function dot2(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}

export function length2(vector: Vec2): number {
  return Math.sqrt(dot2(vector, vector));
}

export function add3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function subtract3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale3(vector: Vec3, factor: number): Vec3 {
  return [vector[0] * factor, vector[1] * factor, vector[2] * factor];
}

export function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function length3(vector: Vec3): number {
  return Math.sqrt(dot3(vector, vector));
}

/** Point `fraction` of the way from `start` to `end`. */
export function lerp3(start: Vec3, end: Vec3, fraction: number): Vec3 {
  return add3(start, scale3(subtract3(end, start), fraction));
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

// ------------------------------------------------------------------ angles

/** Unit vector pointing along `facing` on the ring floor. */
export function headingVector(facing: number): Vec2 {
  return [Math.cos(facing), Math.sin(facing)];
}

/** Unit vector pointing to the left of `facing` on the ring floor. */
export function leftVector(facing: number): Vec2 {
  return [-Math.sin(facing), Math.cos(facing)];
}

/** Express a planar world vector as [forward, left] for `facing`. */
export function worldToLocal(vector: Vec2, facing: number): Vec2 {
  return [dot2(vector, headingVector(facing)), dot2(vector, leftVector(facing))];
}

/** Convert a planar [forward, left] vector back to world axes. */
export function localToWorld(local: Vec2, facing: number): Vec2 {
  return add2(scale2(headingVector(facing), local[0]), scale2(leftVector(facing), local[1]));
}

/** Wrap an angle to the interval [-pi, pi). */
export function wrapAngle(angle: number): number {
  // JavaScript's % keeps the sign of the dividend, so wrap twice.
  const shifted = (((angle + Math.PI) % FULL_TURN) + FULL_TURN) % FULL_TURN;
  return shifted - Math.PI;
}

/** Turn `current` toward `target` by at most `maxStep` radians. */
export function rotateTowards(current: number, target: number, maxStep: number): number {
  assert(maxStep >= 0, 'maxStep must be >= 0');
  const difference = wrapAngle(target - current);
  return wrapAngle(current + clamp(difference, -maxStep, maxStep));
}

// ---------------------------------------------------------------- distance

/**
 * Shortest distance between `point` and a line segment.
 *
 * Used for swept hit detection: a glove that moves from `segmentStart` to
 * `segmentEnd` during one frame hits everything close to that path, so a fast
 * punch can never tunnel through its target between two frames.
 */
export function segmentPointDistance(segmentStart: Vec3, segmentEnd: Vec3, point: Vec3): number {
  const segment = subtract3(segmentEnd, segmentStart);
  const squaredLength = dot3(segment, segment);
  if (squaredLength === 0) {
    return length3(subtract3(point, segmentStart));
  }
  const fraction = dot3(subtract3(point, segmentStart), segment) / squaredLength;
  const closest = add3(segmentStart, scale3(segment, clamp(fraction, 0, 1)));
  return length3(subtract3(point, closest));
}
