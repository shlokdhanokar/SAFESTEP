"""Debug overlay for the preview window.

Purely a development and demonstration aid -- the target user cannot see it.
Kept separate from detection so the headless path never pays for drawing, and
so nothing in the safety-relevant pipeline depends on rendering.

Boxes are coloured by urgency rather than by class, because when a sighted
helper is watching the preview the question is always "how close is it".
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from .spatial import Detection, Proximity, Zone

# BGR, ordered from calm to urgent.
_PROXIMITY_COLOR = {
    Proximity.FAR: (150, 150, 150),
    Proximity.MODERATE: (0, 200, 0),
    Proximity.NEAR: (0, 190, 255),
    Proximity.IMMEDIATE: (0, 0, 255),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Top strip reserved for the HUD. Box captions are kept out of it: a close
# obstacle fills the frame, so its box touches y=0 and its caption would
# otherwise be drawn straight over the HUD text, rendering both unreadable.
HUD_RESERVED_PX = 50


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Return a copy of ``frame`` with boxes and labels drawn.

    Copies rather than mutating: the original SafeStep drew onto the caller's
    array while claiming to return a new one, which quietly corrupted any
    downstream use of the raw frame.
    """
    canvas = frame.copy()

    for detection in detections:
        color = _PROXIMITY_COLOR.get(detection.proximity, (200, 200, 200))
        box = detection.bbox
        thickness = 3 if detection.proximity is Proximity.IMMEDIATE else 2
        cv2.rectangle(canvas, (box.x, box.y), (box.x2, box.y2), color, thickness)

        caption = f"{detection.label} {detection.confidence:.2f}"
        if detection.distance_m is not None:
            # "<" not "~" when the object is cut off by a frame edge: the
            # pinhole figure is an upper bound there, not an estimate.
            marker = "<" if detection.truncated else "~"
            caption += f" {marker}{detection.distance_m:.1f}m"
        if detection.track_id is not None:
            caption += f" #{detection.track_id}"

        _draw_caption(canvas, caption, box.x, box.y, color)

    return canvas


def _draw_caption(canvas: np.ndarray, text: str, x: int, y: int, color) -> None:
    """Draw label text on a filled background so it stays legible over any scene.

    The caption normally sits above the box. When there is no room -- a close
    obstacle fills the frame, so its box touches the top edge -- it drops just
    inside the box instead, and in either case it stays clear of the HUD band.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, 0.5, 1)
    label_h = text_h + baseline + 4

    above = y - label_h
    top = above if above >= HUD_RESERVED_PX else max(y, HUD_RESERVED_PX)
    left = max(0, min(x, canvas.shape[1] - text_w - 6))

    cv2.rectangle(canvas, (left, top), (left + text_w + 6, top + label_h), color, -1)
    cv2.putText(
        canvas, text, (left + 3, top + text_h + 2), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )


def draw_hud(
    frame: np.ndarray,
    detector_name: str,
    fps: float,
    detection_count: int,
    last_announcement: Optional[str] = None,
) -> np.ndarray:
    """Overlay run statistics in the top-left corner. Mutates and returns ``frame``.

    Sits on a dimmed band so it stays readable over a bright scene. Draw this
    last: it owns the top ``HUD_RESERVED_PX`` of the frame.
    """
    lines = [
        f"backend: {detector_name}   fps: {fps:5.1f}   objects: {detection_count}",
    ]
    if last_announcement:
        lines.append(f'last alert: "{last_announcement}"')

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], HUD_RESERVED_PX), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for index, line in enumerate(lines):
        y = 22 + index * 22
        cv2.putText(frame, line, (10, y), _FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_zone_guides(frame: np.ndarray, center_fraction: float = 0.34) -> np.ndarray:
    """Draw the left/centre/right zone boundaries. Mutates and returns ``frame``."""
    height, width = frame.shape[:2]
    half_band = int((center_fraction * width) / 2)
    midpoint = width // 2

    for x in (midpoint - half_band, midpoint + half_band):
        cv2.line(frame, (x, 0), (x, height), (90, 90, 90), 1, cv2.LINE_AA)

    return frame


def zone_label(zone: Zone) -> str:
    return zone.value
