/**
 * Turning a pixel box into something a walking person can act on.
 *
 * A direct port of `src/safestep/spatial.py`; see that file for the reasoning
 * behind each rule. The lookup tables are generated (`tables.mjs`) and the
 * logic is checked against Python by `scripts/check_parity.mjs`.
 */

import { BBox } from './geometry.mjs';
import {
  IMMEDIATE_M, KNOWN_HEIGHTS_M, KNOWN_WIDTHS_M, MODERATE_M, NEAR_M, UNKNOWN_LABEL,
} from './tables.mjs';

export { UNKNOWN_LABEL };

export const Zone = Object.freeze({ LEFT: 'left', CENTER: 'center', RIGHT: 'right' });

export const ZONE_SPOKEN = Object.freeze({
  left: 'on your left', center: 'ahead', right: 'on your right',
});

/** Ordered so a larger value is more urgent. */
export const Proximity = Object.freeze({ FAR: 0, MODERATE: 1, NEAR: 2, IMMEDIATE: 3 });
export const PROXIMITY_NAME = Object.freeze(['FAR', 'MODERATE', 'NEAR', 'IMMEDIATE']);
export const PROXIMITY_SPOKEN = Object.freeze({
  FAR: 'far', MODERATE: 'ahead of you', NEAR: 'close', IMMEDIATE: 'very close',
});

export function focalLengthPx(frameWidth, horizontalFovDeg) {
  if (frameWidth <= 0) throw new RangeError('frameWidth must be positive');
  if (!(horizontalFovDeg > 0 && horizontalFovDeg < 180)) {
    throw new RangeError('horizontalFovDeg must be in (0, 180)');
  }
  return (frameWidth / 2) / Math.tan((horizontalFovDeg * Math.PI / 180) / 2);
}

export function isVerticallyTruncated(bbox, frameHeight, margin = 2) {
  if (frameHeight <= 0) return false;
  return bbox.y <= margin || bbox.y2 >= frameHeight - margin;
}

export function isHorizontallyTruncated(bbox, frameWidth, margin = 2) {
  if (frameWidth <= 0) return false;
  return bbox.x <= margin || bbox.x2 >= frameWidth - margin;
}

function aspectSaysVerticallyIncomplete(bbox, realH, realW) {
  if (bbox.w <= 0 || bbox.h <= 0 || realW <= 0) return false;
  return (bbox.h / bbox.w) < (realH / realW) * 0.6;
}

function aspectSaysHorizontallyIncomplete(bbox, realH, realW) {
  if (bbox.w <= 0 || bbox.h <= 0 || realW <= 0) return false;
  return (bbox.h / bbox.w) > (realH / realW) * 1.7;
}

/**
 * Estimate distance, choosing whichever box dimension is trustworthy.
 * Returns `{metres, method, isUpperBound}`; `metres` is null when unknown.
 */
export function estimateDistance(bbox, label, focalPx, frameWidth, frameHeight) {
  const key = String(label).toLowerCase();
  const realH = KNOWN_HEIGHTS_M[key];
  const realW = KNOWN_WIDTHS_M[key];
  const none = { metres: null, method: '', isUpperBound: false };
  if (focalPx <= 0) return none;

  const byHeight = (realH !== undefined && bbox.h > 0) ? realH * focalPx / bbox.h : null;
  const byWidth = (realW !== undefined && bbox.w > 0) ? realW * focalPx / bbox.w : null;

  if (byHeight === null && byWidth === null) return none;
  if (byWidth === null) {
    return { metres: byHeight, method: 'height', isUpperBound: isVerticallyTruncated(bbox, frameHeight) };
  }
  if (byHeight === null) {
    return { metres: byWidth, method: 'width', isUpperBound: isHorizontallyTruncated(bbox, frameWidth) };
  }

  const vCut = isVerticallyTruncated(bbox, frameHeight);
  const hCut = isHorizontallyTruncated(bbox, frameWidth);

  if (vCut && !hCut) return { metres: byWidth, method: 'width', isUpperBound: false };
  if (hCut && !vCut) return { metres: byHeight, method: 'height', isUpperBound: false };

  if (!vCut && !hCut) {
    if (aspectSaysVerticallyIncomplete(bbox, realH, realW)) {
      return { metres: byWidth, method: 'width', isUpperBound: false };
    }
    if (aspectSaysHorizontallyIncomplete(bbox, realH, realW)) {
      return { metres: byHeight, method: 'height', isUpperBound: false };
    }
    return { metres: byHeight, method: 'height', isUpperBound: false };
  }

  // Cut off in both axes: report the nearer reading as an upper bound.
  const nearer = Math.min(byHeight, byWidth);
  return { metres: nearer, method: nearer === byHeight ? 'height' : 'width', isUpperBound: true };
}

export function proximityFromDistance(distanceM) {
  if (distanceM < IMMEDIATE_M) return Proximity.IMMEDIATE;
  if (distanceM < NEAR_M) return Proximity.NEAR;
  if (distanceM < MODERATE_M) return Proximity.MODERATE;
  return Proximity.FAR;
}

export function proximityFromFrameFraction(bbox, frameHeight) {
  if (frameHeight <= 0) return Proximity.FAR;
  const fraction = bbox.h / frameHeight;
  if (fraction >= 0.60) return Proximity.IMMEDIATE;
  if (fraction >= 0.35) return Proximity.NEAR;
  if (fraction >= 0.15) return Proximity.MODERATE;
  return Proximity.FAR;
}

export function zoneFor(bbox, frameWidth, centerFraction = 0.34) {
  if (frameWidth <= 0) return Zone.CENTER;
  if (!(centerFraction > 0 && centerFraction < 1)) {
    throw new RangeError('centerFraction must be in (0, 1)');
  }
  const halfBand = (centerFraction * frameWidth) / 2;
  const midpoint = frameWidth / 2;
  const cx = bbox.cx;
  if (cx < midpoint - halfBand) return Zone.LEFT;
  if (cx > midpoint + halfBand) return Zone.RIGHT;
  return Zone.CENTER;
}

/** Combine a raw box with spatial context. Mirrors `spatial.enrich`. */
export function enrich(bbox, label, confidence, frameWidth, frameHeight, focalPx, centerFraction = 0.34) {
  const clipped = bbox.clipped(frameWidth, frameHeight);
  const estimate = estimateDistance(clipped, label, focalPx, frameWidth, frameHeight);
  const fractionBand = proximityFromFrameFraction(clipped, frameHeight);

  let proximity;
  if (estimate.metres === null) {
    proximity = fractionBand;
  } else if (estimate.isUpperBound || isVerticallyTruncated(clipped, frameHeight)) {
    proximity = Math.max(proximityFromDistance(estimate.metres), fractionBand);
  } else {
    proximity = proximityFromDistance(estimate.metres);
  }

  return {
    bbox: clipped,
    label,
    confidence,
    zone: zoneFor(clipped, frameWidth, centerFraction),
    proximity,
    distanceM: estimate.metres,
    truncated: estimate.isUpperBound,
    distanceMethod: estimate.method,
    trackId: null,
  };
}
