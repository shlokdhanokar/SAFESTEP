"""Axis-aligned bounding box primitives.

Pure geometry with no OpenCV dependency, so it is cheap to import and trivial
to test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """An axis-aligned box in pixel coordinates.

    ``x``/``y`` are the top-left corner, ``w``/``h`` the extent. Following the
    OpenCV convention, the box covers columns ``[x, x + w)`` and rows
    ``[y, y + h)``.
    """

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w < 0 or self.h < 0:
            raise ValueError(f"BBox extent must be non-negative, got w={self.w} h={self.h}")

    @property
    def x2(self) -> int:
        """Exclusive right edge."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Exclusive bottom edge."""
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def clipped(self, width: int, height: int) -> "BBox":
        """Return this box clamped to a ``width`` x ``height`` frame.

        Detector outputs regularly extend past the frame edge (the network
        regresses a box for a partially visible object); clipping keeps
        downstream area and distance maths honest.
        """
        x1 = max(0, min(self.x, width))
        y1 = max(0, min(self.y, height))
        x2 = max(x1, min(self.x2, width))
        y2 = max(y1, min(self.y2, height))
        return BBox(x1, y1, x2 - x1, y2 - y1)

    def iou(self, other: "BBox") -> float:
        """Intersection-over-union with ``other``. Returns 0.0 for degenerate boxes."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw <= 0 or ih <= 0:
            return 0.0

        intersection = iw * ih
        union = self.area + other.area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BBox":
        """Build from corner coordinates, rounding to whole pixels.

        Corners are ordered defensively: several detector heads emit boxes with
        inverted edges near the frame border.
        """
        left, right = sorted((int(round(x1)), int(round(x2))))
        top, bottom = sorted((int(round(y1)), int(round(y2))))
        return cls(left, top, right - left, bottom - top)
