/**
 * Mapping obstacles onto vibration motors.
 *
 * Port of `src/safestep/haptics.py`. Strength varies continuously with
 * distance rather than stepping between bands, so a wearer feels an obstacle
 * approaching. In the browser demo this drives the on-screen meters and, where
 * the device supports it, the Vibration API.
 */

import { PROXIMITY_NAME, Proximity, Zone } from './spatial.mjs';

export const FULL_POWER_M = 0.5;
export const MAX_RANGE_M = 5.0;

const INTENSITY = { 0: 0.0, 1: 0.35, 2: 0.65, 3: 1.0 };
const INTENSITY_FLOOR = { 0: 0.0, 1: 0.20, 2: 0.45, 3: 0.75 };
const PULSE_HZ = { 0: 0.0, 1: 1.5, 2: 3.0, 3: 6.0 };

/** Left/right weighting. The off-side is damped, not silenced. */
const BALANCE = {
  [Zone.LEFT]: [1.0, 0.2],
  [Zone.CENTER]: [1.0, 1.0],
  [Zone.RIGHT]: [0.2, 1.0],
};

export function intensityForDistance(distanceM) {
  if (distanceM <= FULL_POWER_M) return 1.0;
  if (distanceM >= MAX_RANGE_M) return 0.0;
  return (MAX_RANGE_M - distanceM) / (MAX_RANGE_M - FULL_POWER_M);
}

export function pulseHzForDistance(distanceM) {
  return 1.0 + 6.0 * intensityForDistance(distanceM);
}

export const IDLE = Object.freeze({
  left: 0, right: 0, pulseHz: 0, intensity: 0, strength: 0,
  proximity: 'FAR', zone: Zone.CENTER, active: false,
});

export function cueFor(proximity, zone, distanceM = null) {
  let intensity;
  let pulse;

  if (distanceM === null || distanceM === undefined) {
    intensity = INTENSITY[proximity];
    pulse = PULSE_HZ[proximity];
  } else if (proximity === Proximity.FAR) {
    intensity = 0; pulse = 0;
  } else {
    intensity = Math.max(intensityForDistance(distanceM), INTENSITY_FLOOR[proximity]);
    pulse = pulseHzForDistance(distanceM);
  }

  const [lw, rw] = BALANCE[zone] ?? BALANCE[Zone.CENTER];
  const left = intensity * lw;
  const right = intensity * rw;

  return {
    left, right, pulseHz: pulse, intensity,
    strength: Math.max(left, right),
    proximity: PROXIMITY_NAME[proximity],
    zone,
    active: intensity > 0,
  };
}

export function cueForDetection(detection) {
  if (!detection) return IDLE;
  return cueFor(detection.proximity, detection.zone, detection.distanceM);
}

/** Only one obstacle drives the motors: superimposed patterns cannot be decoded. */
export function strongestCue(detections) {
  let best = null;
  for (const d of detections) {
    if (best === null || d.proximity > best.proximity) {
      best = d;
    } else if (d.proximity === best.proximity) {
      const bestCentre = best.zone === Zone.CENTER;
      const thisCentre = d.zone === Zone.CENTER;
      if ((thisCentre && !bestCentre) || (thisCentre === bestCentre && d.bbox.area > best.bbox.area)) {
        best = d;
      }
    }
  }
  return cueForDetection(best);
}
