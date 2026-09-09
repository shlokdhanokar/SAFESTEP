"""Turning a pixel box into something a walking person can act on.

A bounding box alone is not actionable. What matters to a blind user is *where*
the obstacle is (left / ahead / right) and *how close* it is. This module
derives both, and is deliberately free of OpenCV so it can be tested directly.

Distance uses the pinhole camera model::

    distance = (real_object_height * focal_length_px) / object_height_px

That requires knowing the object's real-world height, which is why it is only
available for recognised classes (see :data:`KNOWN_HEIGHTS_M`). For unknown
classes -- notably everything produced by the colour detector -- we fall back to
a coarse "how much of the frame does it fill" heuristic.

The estimate is monocular and therefore approximate. It assumes the object is
upright, fully visible vertically, and roughly typical in size. Genuine depth
needs stereo or a time-of-flight sensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from .geometry import BBox

# Label used by detectors that recognise "something is there" but not what it is.
UNKNOWN_LABEL = "obstacle"


class Zone(str, Enum):
    """Horizontal position of an obstacle relative to the user's heading."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

    @property
    def spoken(self) -> str:
        return {"left": "on your left", "center": "ahead", "right": "on your right"}[self.value]


class Proximity(int, Enum):
    """Coarse distance band. Ordered so a larger value is more urgent."""

    FAR = 0
    MODERATE = 1
    NEAR = 2
    IMMEDIATE = 3

    @property
    def spoken(self) -> str:
        return {
            "FAR": "far",
            "MODERATE": "ahead of you",
            "NEAR": "close",
            "IMMEDIATE": "very close",
        }[self.name]


# Distance band edges in metres. An obstacle closer than IMMEDIATE_M is within
# roughly one walking pace and warrants the most urgent cue.
IMMEDIATE_M = 1.0
NEAR_M = 2.5
MODERATE_M = 5.0

# Typical real-world heights (metres) for the COCO / VOC classes both neural
# backends can emit. Values are deliberately mid-range averages; the pinhole
# estimate degrades gracefully with modest error here.
KNOWN_HEIGHTS_M = {
    "person": 1.70,
    "bicycle": 1.05,
    "car": 1.45,
    "motorbike": 1.10,
    "motorcycle": 1.10,
    "bus": 3.00,
    "truck": 3.20,
    "train": 3.50,
    "aeroplane": 4.00,
    "airplane": 4.00,
    "boat": 1.50,
    "bench": 0.85,
    "chair": 0.90,
    "sofa": 0.80,
    "couch": 0.80,
    "bed": 0.60,
    "diningtable": 0.75,
    "dining table": 0.75,
    "table": 0.75,
    "toilet": 0.75,
    "tvmonitor": 0.45,
    "tv": 0.45,
    "pottedplant": 0.60,
    "potted plant": 0.60,
    "refrigerator": 1.70,
    "door": 2.00,
    "traffic light": 3.00,
    "fire hydrant": 0.75,
    "stop sign": 2.10,
    "parking meter": 1.20,
    "bird": 0.20,
    "cat": 0.30,
    "dog": 0.55,
    "horse": 1.60,
    "sheep": 0.90,
    "cow": 1.40,
    "backpack": 0.45,
    "suitcase": 0.55,
    "bottle": 0.25,
    "cell phone": 0.15,
    "laptop": 0.25,
    "cup": 0.11,
    "book": 0.24,
    "bowl": 0.08,
    "keyboard": 0.02,
    "mouse": 0.04,
    "remote": 0.16,
    "clock": 0.30,
    "vase": 0.25,
    "scissors": 0.20,
    "umbrella": 0.90,
    "handbag": 0.35,
    "microwave": 0.30,
    "oven": 0.60,
    "sink": 0.20,
    "teddy bear": 0.35,
    "wine glass": 0.18,
}

# Typical real-world *widths* (metres), used when an object's height is cut off
# by the frame edge but its width is not.
#
# Width is the weaker cue in general -- it changes with orientation, where
# height does not -- so it is only consulted when height is unusable. In that
# situation it is dramatically better: at half a metre a camera sees a person's
# head and shoulders, perhaps 0.4 m of the 1.7 m the height model assumes, so
# the height estimate is out by a factor of four while the shoulders remain
# fully visible and measure correctly.
KNOWN_WIDTHS_M = {
    "person": 0.50,
    "bicycle": 0.60,
    "car": 1.80,
    "motorbike": 0.80,
    "motorcycle": 0.80,
    "bus": 2.55,
    "truck": 2.50,
    "bench": 1.50,
    "chair": 0.50,
    "sofa": 1.80,
    "couch": 1.80,
    "bed": 1.50,
    "diningtable": 1.20,
    "dining table": 1.20,
    "table": 1.20,
    "toilet": 0.40,
    "tvmonitor": 0.90,
    "tv": 0.90,
    "laptop": 0.35,
    "pottedplant": 0.40,
    "potted plant": 0.40,
    "refrigerator": 0.70,
    "door": 0.85,
    "stop sign": 0.75,
    "fire hydrant": 0.40,
    "parking meter": 0.20,
    "dog": 0.30,
    "cat": 0.20,
    "horse": 0.60,
    "sheep": 0.45,
    "cow": 0.65,
    "backpack": 0.32,
    "suitcase": 0.45,
    "bottle": 0.08,
    "cell phone": 0.07,
    "cup": 0.09,
    "book": 0.16,
    "bowl": 0.15,
    "keyboard": 0.35,
    "mouse": 0.06,
    "remote": 0.05,
    "clock": 0.30,
    "vase": 0.15,
    "scissors": 0.08,
    "umbrella": 1.00,
    "handbag": 0.30,
    "microwave": 0.50,
    "oven": 0.60,
    "sink": 0.50,
    "teddy bear": 0.25,
    "wine glass": 0.08,
}


@dataclass(frozen=True)
class Detection:
    """A detected obstacle, enriched with everything the alert layer needs."""

    bbox: BBox
    label: str
    confidence: float
    zone: Zone
    proximity: Proximity
    distance_m: Optional[float] = None
    track_id: Optional[int] = None
    #: ``distance_m`` is an upper bound rather than an estimate: the object is
    #: cut off in the dimension it was measured from, so it may be much nearer.
    truncated: bool = False
    #: Which box dimension the distance came from: "height", "width" or "".
    distance_method: str = ""

    def with_track_id(self, track_id: int) -> "Detection":
        return replace(self, track_id=track_id)


def focal_length_px(frame_width: int, horizontal_fov_deg: float) -> float:
    """Focal length in pixels implied by a frame width and horizontal field of view.

    Derived from the pinhole relation ``tan(fov/2) = (width/2) / f``. Typical
    webcams are 60-70 degrees; the Raspberry Pi Camera v2 is about 62.
    """
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be in (0, 180)")

    half_fov = math.radians(horizontal_fov_deg) / 2.0
    return (frame_width / 2.0) / math.tan(half_fov)


def estimate_distance_m(bbox: BBox, label: str, focal_px: float) -> Optional[float]:
    """Estimate distance to ``bbox`` via the pinhole model.

    Returns ``None`` when the class has no known real-world height or the box
    has no vertical extent, in which case callers should fall back to
    :func:`proximity_from_frame_fraction`.
    """
    real_height = KNOWN_HEIGHTS_M.get(label.lower())
    if real_height is None or bbox.h <= 0 or focal_px <= 0:
        return None
    return (real_height * focal_px) / bbox.h


def proximity_from_distance(distance_m: float) -> Proximity:
    """Bucket a metric distance into a :class:`Proximity` band."""
    if distance_m < IMMEDIATE_M:
        return Proximity.IMMEDIATE
    if distance_m < NEAR_M:
        return Proximity.NEAR
    if distance_m < MODERATE_M:
        return Proximity.MODERATE
    return Proximity.FAR


def proximity_from_frame_fraction(bbox: BBox, frame_height: int) -> Proximity:
    """Fallback proximity for objects of unknown real-world size.

    Uses the fraction of frame height the box occupies. Much weaker than the
    pinhole estimate -- a nearby small object and a distant large one are
    indistinguishable -- but it is monotonic in distance for any *fixed* object,
    which is enough to drive an urgency cue.
    """
    if frame_height <= 0:
        return Proximity.FAR

    fraction = bbox.h / frame_height
    if fraction >= 0.60:
        return Proximity.IMMEDIATE
    if fraction >= 0.35:
        return Proximity.NEAR
    if fraction >= 0.15:
        return Proximity.MODERATE
    return Proximity.FAR


@dataclass(frozen=True)
class DistanceEstimate:
    """A distance reading plus how much to trust it."""

    metres: Optional[float]
    #: Which dimension it came from: "height", "width" or "" when unknown.
    method: str = ""
    #: True when the object is cut off in the measured dimension, making the
    #: figure an upper bound -- the object is at most this far away.
    is_upper_bound: bool = False


def _aspect_says_vertically_incomplete(bbox: BBox, real_h: float, real_w: float) -> bool:
    """Whether the box is far shorter, relative to its width, than the class should be.

    Catches partial detections that never touch a frame edge, which the
    truncation test cannot see. A hand or forearm picked up as "person" gives a
    roughly square box; a whole standing person is three to four times taller
    than wide. Measuring that square box against a 1.7 m height model puts the
    obstacle metres away when it is centimetres away.
    """
    if bbox.w <= 0 or bbox.h <= 0 or real_w <= 0:
        return False
    canonical = real_h / real_w
    observed = bbox.h / bbox.w
    return observed < canonical * 0.6


def _aspect_says_horizontally_incomplete(bbox: BBox, real_h: float, real_w: float) -> bool:
    """The mirror case: box far narrower, relative to height, than the class should be."""
    if bbox.w <= 0 or bbox.h <= 0 or real_w <= 0:
        return False
    canonical = real_h / real_w
    observed = bbox.h / bbox.w
    return observed > canonical * 1.7


def estimate_distance(
    bbox: BBox,
    label: str,
    focal_px: float,
    frame_width: int,
    frame_height: int,
) -> DistanceEstimate:
    """Estimate distance, choosing whichever box dimension is trustworthy.

    Height is the better cue in general: a person's height does not change as
    they turn, while their apparent width does. But height is unusable when the
    object is vertically cut off -- either by the frame edge, or because the
    detector only found part of it -- and in that case width is far better.

    The selection order is:

    1. one dimension known -> use it;
    2. cut off in one axis only -> use the other;
    3. aspect ratio far from the class's canonical shape -> use the axis that
       is not compressed;
    4. otherwise -> height.
    """
    real_h = KNOWN_HEIGHTS_M.get(label.lower())
    real_w = KNOWN_WIDTHS_M.get(label.lower())
    if focal_px <= 0:
        return DistanceEstimate(None)

    by_height = (real_h * focal_px / bbox.h) if (real_h and bbox.h > 0) else None
    by_width = (real_w * focal_px / bbox.w) if (real_w and bbox.w > 0) else None

    if by_height is None and by_width is None:
        return DistanceEstimate(None)
    if by_width is None:
        return DistanceEstimate(
            by_height, "height", is_vertically_truncated(bbox, frame_height)
        )
    if by_height is None:
        return DistanceEstimate(
            by_width, "width", is_horizontally_truncated(bbox, frame_width)
        )

    v_cut = is_vertically_truncated(bbox, frame_height)
    h_cut = is_horizontally_truncated(bbox, frame_width)

    if v_cut and not h_cut:
        return DistanceEstimate(by_width, "width", False)
    if h_cut and not v_cut:
        return DistanceEstimate(by_height, "height", False)

    if not v_cut and not h_cut:
        if _aspect_says_vertically_incomplete(bbox, real_h, real_w):
            return DistanceEstimate(by_width, "width", False)
        if _aspect_says_horizontally_incomplete(bbox, real_h, real_w):
            return DistanceEstimate(by_height, "height", False)
        return DistanceEstimate(by_height, "height", False)

    # Cut off in both axes: nothing is fully visible, so report the nearer of
    # the two readings as an upper bound. Under-stating distance is the safe
    # error for an obstacle warning.
    nearer = min(by_height, by_width)
    return DistanceEstimate(nearer, "height" if nearer == by_height else "width", True)


def is_horizontally_truncated(bbox: BBox, frame_width: int, margin: int = 2) -> bool:
    """Whether the box is cut off by the left or right edge of the frame."""
    if frame_width <= 0:
        return False
    return bbox.x <= margin or bbox.x2 >= frame_width - margin


def is_vertically_truncated(bbox: BBox, frame_height: int, margin: int = 2) -> bool:
    """Whether the box is cut off by the top or bottom edge of the frame.

    Only vertical truncation matters: the pinhole estimate divides by the box's
    pixel *height*, so a box clipped left or right is still measured correctly.

    This is the single largest source of error in practice. A wearable camera
    truncates anything close -- you see a person's torso, not their feet -- and
    the shortened box reads as a smaller, therefore more distant, object. The
    error is always in the same direction: it under-warns about the nearest
    obstacles, which is the worst possible bias for this device.
    """
    if frame_height <= 0:
        return False
    return bbox.y <= margin or bbox.y2 >= frame_height - margin


def zone_for(bbox: BBox, frame_width: int, center_fraction: float = 0.34) -> Zone:
    """Classify horizontal position from the box centroid.

    ``center_fraction`` is the share of frame width treated as straight ahead.
    The default 0.34 makes the centre band slightly wider than a pure third,
    biasing ambiguous obstacles toward "ahead" -- the safer reading, since an
    obstacle called "ahead" prompts the user to slow down.
    """
    if frame_width <= 0:
        return Zone.CENTER
    if not 0.0 < center_fraction < 1.0:
        raise ValueError("center_fraction must be in (0, 1)")

    half_band = (center_fraction * frame_width) / 2.0
    midpoint = frame_width / 2.0
    cx = bbox.cx

    if cx < midpoint - half_band:
        return Zone.LEFT
    if cx > midpoint + half_band:
        return Zone.RIGHT
    return Zone.CENTER


def enrich(
    bbox: BBox,
    label: str,
    confidence: float,
    frame_shape: tuple,
    focal_px: float,
    center_fraction: float = 0.34,
) -> Detection:
    """Combine a raw box with spatial context to produce a :class:`Detection`.

    ``frame_shape`` is the numpy ``(height, width, ...)`` tuple.
    """
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    clipped = bbox.clipped(frame_width, frame_height)

    estimate = estimate_distance(clipped, label, focal_px, frame_width, frame_height)
    distance = estimate.metres
    fraction_band = proximity_from_frame_fraction(clipped, frame_height)

    if distance is None:
        proximity = fraction_band
    elif estimate.is_upper_bound or is_vertically_truncated(clipped, frame_height):
        # Whenever the object runs off the top or bottom of the frame, take
        # whichever estimator is more urgent. Width may give an accurate
        # *number*, but a box tall enough to hit the frame edge is close by
        # definition, and the band is the part the user acts on.
        proximity = max(proximity_from_distance(distance), fraction_band)
    else:
        proximity = proximity_from_distance(distance)

    return Detection(
        bbox=clipped,
        label=label,
        confidence=confidence,
        zone=zone_for(clipped, frame_width, center_fraction),
        proximity=proximity,
        distance_m=distance,
        truncated=estimate.is_upper_bound,
        distance_method=estimate.method,
    )
