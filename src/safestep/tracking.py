"""Lightweight IoU tracker giving obstacles stable identities across frames.

Two things depend on this:

* **Cooldowns.** Suppressing repeat announcements requires knowing that the
  chair in this frame is the same chair as last frame. Without identities the
  only options are re-announcing constantly or muting by class, which would
  hide a second, closer chair.
* **Frame skipping.** On a Pi the detector runs every Nth frame. The tracker
  carries detections through the gaps so alerting stays smooth.

Greedy IoU matching is enough here: obstacle counts are small (single digits)
and inter-frame motion is modest at walking speed. No Kalman filter, no
Hungarian assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .spatial import Detection


@dataclass
class _Track:
    track_id: int
    detection: Detection
    hits: int = 1
    missing: int = 0


class IoUTracker:
    """Assign persistent ``track_id`` values to detections frame over frame."""

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 8) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    @property
    def active_tracks(self) -> Dict[int, _Track]:
        return self._tracks

    def hits_for(self, track_id: int) -> int:
        """How many frames this track has been confirmed in. 0 if unknown."""
        track = self._tracks.get(track_id)
        return track.hits if track else 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Match ``detections`` against live tracks and stamp them with ids.

        Returns the detections in input order, each carrying a ``track_id``.
        Tracks unmatched for more than ``max_missing`` updates are dropped.
        """
        unmatched_track_ids = set(self._tracks)
        results: List[Detection] = []

        # Greedy: strongest overlaps claim their track first, so a marginal
        # second candidate cannot steal an identity from a confident match.
        scored = []
        for det_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                iou = detection.bbox.iou(track.detection.bbox)
                if iou >= self.iou_threshold:
                    scored.append((iou, det_index, track_id))
        scored.sort(key=lambda item: item[0], reverse=True)

        assigned_detections: Dict[int, int] = {}  # det_index -> track_id
        claimed_tracks: set = set()
        for _iou, det_index, track_id in scored:
            if det_index in assigned_detections or track_id in claimed_tracks:
                continue
            assigned_detections[det_index] = track_id
            claimed_tracks.add(track_id)

        for det_index, detection in enumerate(detections):
            track_id = assigned_detections.get(det_index)

            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id=track_id, detection=detection)
            else:
                track = self._tracks[track_id]
                track.detection = detection
                track.hits += 1
                track.missing = 0
                unmatched_track_ids.discard(track_id)

            results.append(detection.with_track_id(track_id))

        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.missing += 1
            if track.missing > self.max_missing:
                del self._tracks[track_id]

        return results

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
