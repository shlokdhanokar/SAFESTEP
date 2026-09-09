/**
 * Lightweight IoU tracker giving obstacles stable identities across frames.
 *
 * Port of `src/safestep/tracking.py`. Cooldowns depend on knowing that the
 * chair in this frame is the same chair as last frame; without identities the
 * only options are re-announcing constantly or muting by class.
 */

export class IoUTracker {
  constructor(iouThreshold = 0.3, maxMissing = 8) {
    if (!(iouThreshold > 0 && iouThreshold <= 1)) {
      throw new RangeError('iouThreshold must be in (0, 1]');
    }
    this.iouThreshold = iouThreshold;
    this.maxMissing = maxMissing;
    this._tracks = new Map();   // id -> {detection, hits, missing}
    this._nextId = 1;
  }

  get activeTracks() { return this._tracks; }

  hitsFor(trackId) {
    const track = this._tracks.get(trackId);
    return track ? track.hits : 0;
  }

  update(detections) {
    const unmatched = new Set(this._tracks.keys());

    // Greedy: strongest overlaps claim their track first, so a marginal
    // second candidate cannot steal an identity from a confident match.
    const scored = [];
    detections.forEach((detection, detIndex) => {
      for (const [trackId, track] of this._tracks) {
        const iou = detection.bbox.iou(track.detection.bbox);
        if (iou >= this.iouThreshold) scored.push([iou, detIndex, trackId]);
      }
    });
    scored.sort((a, b) => b[0] - a[0]);

    const assigned = new Map();
    const claimed = new Set();
    for (const [, detIndex, trackId] of scored) {
      if (assigned.has(detIndex) || claimed.has(trackId)) continue;
      assigned.set(detIndex, trackId);
      claimed.add(trackId);
    }

    const results = detections.map((detection, detIndex) => {
      let trackId = assigned.get(detIndex);
      if (trackId === undefined) {
        trackId = this._nextId++;
        this._tracks.set(trackId, { detection, hits: 1, missing: 0 });
      } else {
        const track = this._tracks.get(trackId);
        track.detection = detection;
        track.hits += 1;
        track.missing = 0;
        unmatched.delete(trackId);
      }
      return { ...detection, trackId };
    });

    for (const trackId of unmatched) {
      const track = this._tracks.get(trackId);
      track.missing += 1;
      if (track.missing > this.maxMissing) this._tracks.delete(trackId);
    }

    return results;
  }

  reset() {
    this._tracks.clear();
    this._nextId = 1;
  }
}
