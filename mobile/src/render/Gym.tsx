/**
 * The training gym: floor, ring, ropes, corner posts, walls, heavy bags and
 * lights. Everything is static, built once from simple shapes.
 */

import { useMemo } from 'react';

import { GameConfig } from '../sim/config';
import { COLORS } from './palette';

const ROOM_HALF_SIZE = 9;
const ROOM_HEIGHT = 5.5;
const APRON_MARGIN = 0.55;
const APRON_THICKNESS = 0.9;
const POST_HEIGHT = 1.55;
const POST_RADIUS = 0.07;
const ROPE_RADIUS = 0.022;
const ROPE_HEIGHTS = [0.55, 0.9, 1.25];
const CORNER_PAD_HEIGHT = 0.9;
const BAG_RADIUS = 0.22;
const BAG_LENGTH = 1.2;
const BAG_CENTER_HEIGHT = 1.5;
const LAMP_HEIGHT = 4.6;
const RING_LIGHT_HEIGHT = 5;
const SHADOW_MAP_SIZE = 1024;

/** Heavy bags hang outside the ring; [x, z] in metres. */
const BAG_SPOTS: readonly (readonly [number, number])[] = [
  [-6.2, -5.5],
  [-6.8, 1.5],
  [6.4, -4.8],
  [6.6, 3.2],
];

/** Ceiling lamps over the ring's corners and centre; [x, z] in metres. */
const LAMP_SPOTS: readonly (readonly [number, number])[] = [
  [0, 0],
  [-3.5, -3.5],
  [3.5, -3.5],
  [-3.5, 3.5],
  [3.5, 3.5],
];

interface Rope {
  readonly key: string;
  readonly position: readonly [number, number, number];
  /** Rotation that lays the cylinder along the X or Z axis. */
  readonly rotation: readonly [number, number, number];
}

function buildRopes(halfSize: number): Rope[] {
  const ropes: Rope[] = [];
  for (const height of ROPE_HEIGHTS) {
    for (const side of [-1, 1]) {
      ropes.push({
        key: `x${side}h${height}`,
        position: [0, height, side * halfSize],
        rotation: [0, 0, Math.PI / 2],
      });
      ropes.push({
        key: `z${side}h${height}`,
        position: [side * halfSize, height, 0],
        rotation: [Math.PI / 2, 0, 0],
      });
    }
  }
  return ropes;
}

const CORNERS: readonly { x: number; z: number; color: string }[] = [
  { x: -1, z: -1, color: COLORS.cornerBlue },
  { x: 1, z: 1, color: COLORS.cornerRed },
  { x: -1, z: 1, color: COLORS.cornerNeutral },
  { x: 1, z: -1, color: COLORS.cornerNeutral },
];

export function Gym({ config }: { config: GameConfig }) {
  const halfSize = config.ring.halfSize;
  const ropes = useMemo(() => buildRopes(halfSize), [halfSize]);
  const apronSize = 2 * (halfSize + APRON_MARGIN);

  return (
    <group>
      {/* Lights: soft fill everywhere, one shadow-casting lamp over the ring. */}
      <hemisphereLight args={[COLORS.lampGlow, COLORS.gymFloor, 0.9]} />
      <spotLight
        position={[0, RING_LIGHT_HEIGHT, 0]}
        angle={0.95}
        penumbra={0.6}
        intensity={90}
        color={COLORS.lampGlow}
        castShadow
        shadow-mapSize-width={SHADOW_MAP_SIZE}
        shadow-mapSize-height={SHADOW_MAP_SIZE}
      />
      <pointLight position={[-6, 3.5, 5]} intensity={18} color={COLORS.lampGlow} />
      <pointLight position={[6, 3.5, -5]} intensity={18} color={COLORS.lampGlow} />

      {/* Room */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -APRON_THICKNESS, 0]} receiveShadow>
        <planeGeometry args={[2 * ROOM_HALF_SIZE, 2 * ROOM_HALF_SIZE]} />
        <meshStandardMaterial color={COLORS.gymFloor} roughness={0.9} />
      </mesh>
      {[0, 1, 2, 3].map((wall) => (
        <mesh
          key={wall}
          rotation={[0, (wall * Math.PI) / 2, 0]}
          position={[
            Math.sin((wall * Math.PI) / 2) * -ROOM_HALF_SIZE,
            ROOM_HEIGHT / 2 - APRON_THICKNESS,
            Math.cos((wall * Math.PI) / 2) * -ROOM_HALF_SIZE,
          ]}
        >
          <planeGeometry args={[2 * ROOM_HALF_SIZE, ROOM_HEIGHT]} />
          <meshStandardMaterial color={COLORS.gymWall} roughness={0.95} />
        </mesh>
      ))}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, ROOM_HEIGHT - APRON_THICKNESS, 0]}>
        <planeGeometry args={[2 * ROOM_HALF_SIZE, 2 * ROOM_HALF_SIZE]} />
        <meshStandardMaterial color={COLORS.gymCeiling} roughness={1} />
      </mesh>

      {/* Ring: raised platform with the canvas on top (the canvas is height 0). */}
      <mesh position={[0, -APRON_THICKNESS / 2, 0]} receiveShadow>
        <boxGeometry args={[apronSize, APRON_THICKNESS, apronSize]} />
        <meshStandardMaterial color={COLORS.ringApron} roughness={0.8} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]} receiveShadow>
        <planeGeometry args={[2 * halfSize, 2 * halfSize]} />
        <meshStandardMaterial color={COLORS.ringCanvas} roughness={0.85} />
      </mesh>

      {ropes.map((rope) => (
        <mesh key={rope.key} position={rope.position} rotation={rope.rotation}>
          <cylinderGeometry args={[ROPE_RADIUS, ROPE_RADIUS, 2 * halfSize, 8]} />
          <meshStandardMaterial color={COLORS.rope} roughness={0.6} />
        </mesh>
      ))}

      {CORNERS.map((corner) => (
        <group key={`${corner.x}${corner.z}`} position={[corner.x * halfSize, 0, corner.z * halfSize]}>
          <mesh position={[0, POST_HEIGHT / 2, 0]} castShadow>
            <cylinderGeometry args={[POST_RADIUS, POST_RADIUS, POST_HEIGHT, 12]} />
            <meshStandardMaterial color={COLORS.metal} metalness={0.8} roughness={0.35} />
          </mesh>
          <mesh position={[-corner.x * 0.1, CORNER_PAD_HEIGHT, -corner.z * 0.1]} rotation={[0, Math.PI / 4, 0]}>
            <boxGeometry args={[0.34, 0.95, 0.12]} />
            <meshStandardMaterial color={corner.color} roughness={0.7} />
          </mesh>
        </group>
      ))}

      {BAG_SPOTS.map(([x, z]) => (
        <group key={`${x}${z}`} position={[x, BAG_CENTER_HEIGHT - APRON_THICKNESS, z]}>
          <mesh castShadow>
            <cylinderGeometry args={[BAG_RADIUS, BAG_RADIUS, BAG_LENGTH, 16]} />
            <meshStandardMaterial color={COLORS.heavyBag} roughness={0.75} />
          </mesh>
          <mesh position={[0, BAG_LENGTH, 0]}>
            <cylinderGeometry args={[0.012, 0.012, BAG_LENGTH, 6]} />
            <meshStandardMaterial color={COLORS.metal} metalness={0.9} roughness={0.3} />
          </mesh>
        </group>
      ))}

      {LAMP_SPOTS.map(([x, z]) => (
        <mesh key={`${x}${z}`} position={[x, LAMP_HEIGHT - APRON_THICKNESS, z]} rotation={[Math.PI / 2, 0, 0]}>
          <circleGeometry args={[0.45, 20]} />
          <meshBasicMaterial color={COLORS.lampGlow} />
        </mesh>
      ))}
    </group>
  );
}
