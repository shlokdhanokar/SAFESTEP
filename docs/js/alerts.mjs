/**
 * Deciding what to say, and when to stay quiet.
 *
 * Port of `src/safestep/alerts.py`. Four rules in order: relevance floor,
 * confirmation, priority (one obstacle, most urgent, centre-preferring), then
 * rate limiting. The clock is injectable so the harness can test timing
 * deterministically rather than with sleeps.
 */

import { PROXIMITY_NAME, PROXIMITY_SPOKEN, Proximity, ZONE_SPOKEN, Zone } from './spatial.mjs';

/** Render a detection as a short spoken phrase: object, urgency, direction. */
export function describe(detection) {
  const parts = [detection.label, PROXIMITY_SPOKEN[PROXIMITY_NAME[detection.proximity]]];
  if (detection.zone !== Zone.CENTER || detection.proximity === Proximity.IMMEDIATE) {
    parts.push(ZONE_SPOKEN[detection.zone]);
  }
  return parts.join(', ');
}

export class AlertPolicy {
  constructor({
    cooldownS = 4.0,
    minGapS = 1.2,
    minHits = 2,
    minProximity = Proximity.MODERATE,
    clock = () => performance.now() / 1000,
  } = {}) {
    this.cooldownS = cooldownS;
    this.minGapS = minGapS;
    this.minHits = minHits;
    this.minProximity = minProximity;
    this._clock = clock;
    this._lastAnnouncedAt = new Map();
    this._lastAnyAt = null;
  }

  /** Choose the one obstacle worth announcing now, or null. Records the choice. */
  select(detections, hitsFor = null) {
    const now = this._clock();

    const candidates = [];
    for (const detection of detections) {
      if (detection.proximity < this.minProximity) continue;
      if (hitsFor && detection.trackId != null && hitsFor(detection.trackId) < this.minHits) continue;
      candidates.push(detection);
    }
    if (!candidates.length) return null;

    candidates.sort((a, b) => priorityKey(b) - priorityKey(a) || b.bbox.area - a.bbox.area);

    // Applied after ranking so an urgent obstacle is not starved by lesser ones.
    if (this._lastAnyAt !== null && (now - this._lastAnyAt) < this.minGapS) return null;

    for (const detection of candidates) {
      const key = cooldownKey(detection);
      const last = this._lastAnnouncedAt.get(key);
      if (last !== undefined && (now - last) < this.cooldownS) continue;

      this._lastAnnouncedAt.set(key, now);
      this._lastAnyAt = now;
      this._prune(now);
      return {
        text: describe(detection),
        proximity: detection.proximity,
        zone: detection.zone,
        label: detection.label,
        distanceM: detection.distanceM,
      };
    }
    return null;
  }

  _prune(now) {
    const expiry = this.cooldownS * 4;
    for (const [key, seen] of this._lastAnnouncedAt) {
      if (now - seen > expiry) this._lastAnnouncedAt.delete(key);
    }
  }

  reset() {
    this._lastAnnouncedAt.clear();
    this._lastAnyAt = null;
  }
}

/** Urgency first, then straight-ahead. Area breaks the remaining tie. */
function priorityKey(d) {
  return d.proximity * 2 + (d.zone === Zone.CENTER ? 1 : 0);
}

function cooldownKey(d) {
  return d.trackId != null ? `track:${d.trackId}` : `label_zone:${d.label}:${d.zone}`;
}
