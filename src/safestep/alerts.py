"""Deciding what to say, and when to stay quiet.

Announcing every detection every frame is what made the original SafeStep
unusable. This module is the gatekeeper, and it is pure logic over an injected
clock so all of its timing behaviour is directly testable.

Four rules, in order:

1. **Relevance.** Obstacles below ``min_proximity`` are ignored entirely.
2. **Confirmation.** An obstacle must be seen ``min_hits`` times before it is
   announced, which discards single-frame false positives.
3. **Priority.** At most one obstacle is announced per pass: the most urgent,
   preferring what is directly ahead. Reading out a list is useless to someone
   who is walking.
4. **Rate limiting.** A per-object cooldown stops repetition, and a global
   minimum gap stops a crowded scene turning into a stream of chatter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from .spatial import Detection, Proximity, Zone


@dataclass(frozen=True)
class Announcement:
    """A single thing to tell the user."""

    text: str
    proximity: Proximity
    zone: Zone
    label: str
    distance_m: Optional[float] = None


def describe(detection: Detection) -> str:
    """Render a detection as a short spoken phrase.

    Ordering is deliberate: object, then urgency, then direction -- "chair,
    very close, on your left". The most decision-relevant word comes first so
    the user can react before the sentence finishes.
    """
    parts = [detection.label, detection.proximity.spoken]
    if detection.zone is not Zone.CENTER or detection.proximity is Proximity.IMMEDIATE:
        parts.append(detection.zone.spoken)
    return ", ".join(parts)


class AlertPolicy:
    """Select at most one announcement per detection pass."""

    def __init__(
        self,
        cooldown_s: float = 4.0,
        min_gap_s: float = 1.2,
        min_hits: int = 2,
        min_proximity: Proximity = Proximity.MODERATE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cooldown_s = cooldown_s
        self.min_gap_s = min_gap_s
        self.min_hits = min_hits
        self.min_proximity = min_proximity
        self._clock = clock
        self._last_announced_at: Dict[object, float] = {}
        self._last_any_at: Optional[float] = None

    def select(
        self,
        detections: Iterable[Detection],
        hits_for: Optional[Callable[[int], int]] = None,
    ) -> Optional[Announcement]:
        """Choose the one obstacle worth announcing now, or ``None``.

        ``hits_for`` maps a ``track_id`` to its confirmation count; when it is
        omitted the confirmation rule is skipped (used where no tracker runs).
        Calling this records the announcement, so it is not side-effect free.
        """
        now = self._clock()

        candidates: List[Detection] = []
        for detection in detections:
            if detection.proximity < self.min_proximity:
                continue
            if hits_for is not None and detection.track_id is not None:
                if hits_for(detection.track_id) < self.min_hits:
                    continue
            candidates.append(detection)

        if not candidates:
            return None

        candidates.sort(key=self._priority, reverse=True)

        # Global rate limit is applied after ranking so an urgent obstacle is
        # not permanently starved by a stream of lesser ones.
        if self._last_any_at is not None and (now - self._last_any_at) < self.min_gap_s:
            return None

        for detection in candidates:
            key = self._key(detection)
            last = self._last_announced_at.get(key)
            if last is not None and (now - last) < self.cooldown_s:
                continue

            self._last_announced_at[key] = now
            self._last_any_at = now
            self._prune(now)
            return Announcement(
                text=describe(detection),
                proximity=detection.proximity,
                zone=detection.zone,
                label=detection.label,
                distance_m=detection.distance_m,
            )

        return None

    @staticmethod
    def _priority(detection: Detection) -> tuple:
        """Rank key: urgency, then straight-ahead, then apparent size."""
        return (
            int(detection.proximity),
            1 if detection.zone is Zone.CENTER else 0,
            detection.bbox.area,
        )

    @staticmethod
    def _key(detection: Detection) -> object:
        """Cooldown identity: the track when available, else class plus zone."""
        if detection.track_id is not None:
            return ("track", detection.track_id)
        return ("label_zone", detection.label, detection.zone)

    def _prune(self, now: float) -> None:
        """Drop cooldown entries that can no longer suppress anything.

        Without this the dict grows for the lifetime of the process, since every
        new track mints a fresh key.
        """
        expiry = self.cooldown_s * 4
        stale = [key for key, seen in self._last_announced_at.items() if now - seen > expiry]
        for key in stale:
            del self._last_announced_at[key]

    def reset(self) -> None:
        self._last_announced_at.clear()
        self._last_any_at = None
