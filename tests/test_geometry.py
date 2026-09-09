"""Bounding box geometry."""

from __future__ import annotations

import pytest

from safestep.geometry import BBox


class TestBasicProperties:
    def test_edges_and_area(self):
        box = BBox(10, 20, 30, 40)
        assert box.x2 == 40
        assert box.y2 == 60
        assert box.area == 1200

    def test_centroid(self):
        box = BBox(10, 20, 30, 40)
        assert box.centroid == (25.0, 40.0)
        assert box.cx == 25.0
        assert box.cy == 40.0

    def test_negative_extent_rejected(self):
        with pytest.raises(ValueError):
            BBox(0, 0, -5, 10)


class TestClipping:
    def test_box_inside_frame_is_unchanged(self):
        box = BBox(10, 10, 50, 50)
        assert box.clipped(640, 480) == box

    def test_overhanging_box_is_trimmed(self):
        box = BBox(600, 450, 100, 100)
        clipped = box.clipped(640, 480)
        assert clipped == BBox(600, 450, 40, 30)

    def test_negative_origin_is_clamped(self):
        clipped = BBox(0, 0, 50, 50).clipped(640, 480)
        assert clipped.x == 0 and clipped.y == 0

    def test_fully_outside_box_becomes_empty(self):
        clipped = BBox(700, 500, 50, 50).clipped(640, 480)
        assert clipped.area == 0


class TestIoU:
    def test_identical_boxes(self):
        box = BBox(0, 0, 10, 10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert BBox(0, 0, 10, 10).iou(BBox(100, 100, 10, 10)) == 0.0

    def test_touching_edges_do_not_overlap(self):
        assert BBox(0, 0, 10, 10).iou(BBox(10, 0, 10, 10)) == 0.0

    def test_half_overlap(self):
        # Two 10x10 boxes offset by 5 on x: intersection 50, union 150.
        assert BBox(0, 0, 10, 10).iou(BBox(5, 0, 10, 10)) == pytest.approx(50 / 150)

    def test_degenerate_box_is_safe(self):
        assert BBox(0, 0, 0, 0).iou(BBox(0, 0, 10, 10)) == 0.0

    def test_is_symmetric(self):
        a, b = BBox(0, 0, 20, 20), BBox(10, 10, 20, 20)
        assert a.iou(b) == pytest.approx(b.iou(a))


class TestFromXYXY:
    def test_corners_to_box(self):
        assert BBox.from_xyxy(10, 20, 40, 60) == BBox(10, 20, 30, 40)

    def test_inverted_corners_are_normalised(self):
        # Some detector heads emit reversed edges near the frame border.
        assert BBox.from_xyxy(40, 60, 10, 20) == BBox(10, 20, 30, 40)

    def test_floats_are_rounded(self):
        assert BBox.from_xyxy(10.4, 20.6, 40.5, 60.4) == BBox(10, 21, 30, 39)
