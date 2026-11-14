/** Short flashes where punches land. Plain three.js, no React. */

import * as THREE from 'three';

import { COLORS } from './palette';
import { ScenePoint } from './scenePose';

const SPARK_COUNT = 6;
const SPARK_SECONDS = 0.22;
const SPARK_START_SIZE = 0.05;
const SPARK_GROWTH = 0.13;

interface Spark {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.MeshBasicMaterial;
  secondsLeft: number;
}

export class HitSparks {
  /** Add this to the scene. */
  readonly group = new THREE.Group();

  private readonly sparks: Spark[] = [];
  private nextSpark = 0;

  constructor() {
    const geometry = new THREE.SphereGeometry(1, 10, 8);
    for (let index = 0; index < SPARK_COUNT; index += 1) {
      const material = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.visible = false;
      this.group.add(mesh);
      this.sparks.push({ mesh, material, secondsLeft: 0 });
    }
  }

  /** Flash at `at`: warm for a clean hit, cyan for a blocked one. */
  start(at: ScenePoint, wasBlocked: boolean): void {
    const spark = this.sparks[this.nextSpark];
    this.nextSpark = (this.nextSpark + 1) % SPARK_COUNT;
    spark.mesh.position.set(at[0], at[1], at[2]);
    spark.material.color.set(wasBlocked ? COLORS.blockSpark : COLORS.hitSpark);
    spark.secondsLeft = SPARK_SECONDS;
  }

  update(deltaSeconds: number): void {
    for (const spark of this.sparks) {
      spark.secondsLeft = Math.max(0, spark.secondsLeft - deltaSeconds);
      const life = spark.secondsLeft / SPARK_SECONDS;
      spark.mesh.visible = life > 0;
      spark.mesh.scale.setScalar(SPARK_START_SIZE + (1 - life) * SPARK_GROWTH);
      spark.material.opacity = life;
    }
  }

  dispose(): void {
    this.sparks[0].mesh.geometry.dispose();
    for (const spark of this.sparks) {
      spark.material.dispose();
    }
  }
}
