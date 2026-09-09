"""HSV colour-threshold detector -- the zero-download fallback.

This is the original SafeStep algorithm, kept so the application runs before
any model weights are fetched, with three corrections:

* the saturation floor is lowered (the old value of 150 excluded walls,
  concrete and most real hazards);
* the hue ceiling is 179, OpenCV's actual maximum, not 255;
* morphological opening/closing plus a minimum-area filter replace the old
  ``if contours:`` test, which fired on a single pixel of sensor noise.

It still cannot tell a hazard from a coloured poster. Treat it as a
development and demo aid, not as a safety mechanism.
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from ..config import ColorDetectorSettings
from ..geometry import BBox
from ..spatial import UNKNOWN_LABEL
from .base import RawDetection

logger = logging.getLogger(__name__)


class ColorThresholdDetector:
    """Detect saturated colour blobs via HSV thresholding."""

    name = "color"

    def __init__(self, settings: ColorDetectorSettings | None = None) -> None:
        self.settings = settings or ColorDetectorSettings()
        self._lower = np.array(self.settings.lower_hsv, dtype=np.uint8)
        self._upper = np.array(self.settings.upper_hsv, dtype=np.uint8)
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.settings.open_kernel, self.settings.open_kernel)
        )
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.settings.close_kernel, self.settings.close_kernel)
        )
        logger.debug(
            "Colour detector active: hsv %s..%s, min_area=%d",
            self.settings.lower_hsv,
            self.settings.upper_hsv,
            self.settings.min_area_px,
        )

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)

        # Opening first: drop isolated speckles before closing can grow them.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)

        # RETR_EXTERNAL, not RETR_TREE: nested contours are holes in one
        # object, not separate obstacles, and the old code drew a box for each.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[RawDetection] = []
        frame_area = float(frame.shape[0] * frame.shape[1]) or 1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.settings.min_area_px:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                RawDetection(
                    bbox=BBox(int(x), int(y), int(w), int(h)),
                    label=UNKNOWN_LABEL,
                    # No real confidence exists for a threshold, so report the
                    # share of frame filled, capped to a plausible range. This
                    # keeps the field meaningful for sorting without implying
                    # the certainty a neural score would.
                    confidence=float(min(0.99, 0.30 + area / frame_area)),
                )
            )

        return detections
