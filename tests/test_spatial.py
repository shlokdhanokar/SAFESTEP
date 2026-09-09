"""Zone classification, pinhole distance estimation and enrichment."""

from __future__ import annotations

import math

import pytest

from safestep.geometry import BBox
from safestep.spatial import (
    KNOWN_HEIGHTS_M,
    Proximity,
    Zone,
    enrich,
    estimate_distance,
    estimate_distance_m,
    focal_length_px,
    is_horizontally_truncated,
    is_vertically_truncated,
    proximity_from_distance,
    proximity_from_frame_fraction,
    zone_for,
)


class TestFocalLength:
    def test_ninety_degree_fov_gives_focal_equal_to_half_width(self):
        # At 90 degrees, tan(45) == 1, so f == width/2 exactly.
        assert focal_length_px(640, 90.0) == pytest.approx(320.0)

    def test_narrower_fov_gives_longer_focal_length(self):
        assert focal_length_px(640, 40.0) > focal_length_px(640, 80.0)

    def test_typical_webcam(self):
        # 640 px at 65 degrees -> (320) / tan(32.5 deg)
        expected = 320.0 / math.tan(math.radians(32.5))
        assert focal_length_px(640, 65.0) == pytest.approx(expected)

    @pytest.mark.parametrize("width,fov", [(0, 65.0), (-1, 65.0)])
    def test_invalid_width_rejected(self, width, fov):
        with pytest.raises(ValueError):
            focal_length_px(width, fov)

    @pytest.mark.parametrize("fov", [0.0, 180.0, -10.0, 200.0])
    def test_invalid_fov_rejected(self, fov):
        with pytest.raises(ValueError):
            focal_length_px(640, fov)


class TestDistanceEstimation:
    def test_pinhole_matches_hand_computed_value(self):
        # A 1.70 m person occupying 200 px, focal length 500 px:
        #   d = 1.70 * 500 / 200 = 4.25 m
        distance = estimate_distance_m(BBox(0, 0, 60, 200), "person", 500.0)
        assert distance == pytest.approx(4.25)

    def test_distance_is_inversely_proportional_to_pixel_height(self):
        near = estimate_distance_m(BBox(0, 0, 60, 400), "person", 500.0)
        far = estimate_distance_m(BBox(0, 0, 60, 100), "person", 500.0)
        assert far == pytest.approx(near * 4)

    def test_label_lookup_is_case_insensitive(self):
        assert estimate_distance_m(BBox(0, 0, 10, 100), "PERSON", 500.0) == pytest.approx(
            estimate_distance_m(BBox(0, 0, 10, 100), "person", 500.0)
        )

    def test_unknown_class_returns_none(self):
        assert estimate_distance_m(BBox(0, 0, 60, 200), "obstacle", 500.0) is None

    def test_zero_height_box_returns_none(self):
        assert estimate_distance_m(BBox(0, 0, 60, 0), "person", 500.0) is None

    def test_known_heights_are_physically_plausible(self):
        for label, height in KNOWN_HEIGHTS_M.items():
            assert 0.01 < height < 5.0, f"{label} has implausible height {height}"


class TestProximityBands:
    @pytest.mark.parametrize(
        "distance,expected",
        [
            (0.4, Proximity.IMMEDIATE),
            (0.99, Proximity.IMMEDIATE),
            (1.0, Proximity.NEAR),
            (2.4, Proximity.NEAR),
            (2.5, Proximity.MODERATE),
            (4.9, Proximity.MODERATE),
            (5.0, Proximity.FAR),
            (50.0, Proximity.FAR),
        ],
    )
    def test_distance_buckets(self, distance, expected):
        assert proximity_from_distance(distance) is expected

    def test_ordering_is_by_urgency(self):
        assert Proximity.IMMEDIATE > Proximity.NEAR > Proximity.MODERATE > Proximity.FAR

    @pytest.mark.parametrize(
        "box_height,expected",
        [
            (300, Proximity.IMMEDIATE),  # 62% of frame
            (200, Proximity.NEAR),       # 42%
            (100, Proximity.MODERATE),   # 21%
            (30, Proximity.FAR),         # 6%
        ],
    )
    def test_frame_fraction_fallback(self, box_height, expected):
        assert proximity_from_frame_fraction(BBox(0, 0, 50, box_height), 480) is expected

    def test_fallback_handles_zero_height_frame(self):
        assert proximity_from_frame_fraction(BBox(0, 0, 50, 50), 0) is Proximity.FAR


class TestZones:
    @pytest.mark.parametrize(
        "cx,expected",
        [
            (10, Zone.LEFT),
            (200, Zone.LEFT),
            (320, Zone.CENTER),
            (250, Zone.CENTER),   # inside the 34% centre band
            (400, Zone.CENTER),
            (500, Zone.RIGHT),
            (630, Zone.RIGHT),
        ],
    )
    def test_zone_by_centroid(self, cx, expected):
        box = BBox(int(cx) - 5, 100, 10, 100)
        assert zone_for(box, 640) is expected

    def test_centre_band_width_is_configurable(self):
        box = BBox(245, 100, 10, 100)  # centroid at 250
        assert zone_for(box, 640, center_fraction=0.34) is Zone.CENTER
        assert zone_for(box, 640, center_fraction=0.10) is Zone.LEFT

    def test_zero_width_frame_defaults_to_centre(self):
        assert zone_for(BBox(0, 0, 10, 10), 0) is Zone.CENTER

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_centre_fraction_rejected(self, fraction):
        with pytest.raises(ValueError):
            zone_for(BBox(0, 0, 10, 10), 640, center_fraction=fraction)

    def test_spoken_forms(self):
        assert Zone.LEFT.spoken == "on your left"
        assert Zone.CENTER.spoken == "ahead"
        assert Zone.RIGHT.spoken == "on your right"


class TestEnrich:
    FRAME_SHAPE = (480, 640, 3)

    def test_known_class_gets_metric_distance(self):
        detection = enrich(
            BBox(300, 200, 60, 200), "person", 0.9, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.distance_m == pytest.approx(4.25)
        assert detection.proximity is Proximity.MODERATE
        assert detection.zone is Zone.CENTER

    def test_unknown_class_falls_back_to_frame_fraction(self):
        detection = enrich(
            BBox(50, 100, 60, 300), "obstacle", 0.5, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.distance_m is None
        assert detection.proximity is Proximity.IMMEDIATE  # 300/480 = 62%
        assert detection.zone is Zone.LEFT

    def test_box_is_clipped_to_frame(self):
        detection = enrich(
            BBox(600, 400, 200, 200), "person", 0.8, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.bbox.x2 <= 640
        assert detection.bbox.y2 <= 480

    def test_confidence_and_label_pass_through(self):
        detection = enrich(BBox(0, 0, 10, 10), "chair", 0.77, self.FRAME_SHAPE, 500.0)
        assert detection.label == "chair"
        assert detection.confidence == pytest.approx(0.77)
        assert detection.track_id is None

    def test_closer_object_reports_higher_urgency(self):
        far = enrich(BBox(300, 200, 20, 60), "person", 0.9, self.FRAME_SHAPE, 500.0)
        near = enrich(BBox(300, 100, 90, 300), "person", 0.9, self.FRAME_SHAPE, 500.0)
        assert near.proximity > far.proximity


class TestDimensionSelection:
    """Choosing height vs width, the fix for close-range distance error.

    Measured on real hardware: a webcam at 0.5 m sees head and shoulders,
    roughly 0.4 m of the 1.7 m the height model assumes, so a height-only
    estimate reported ~2 m. The shoulders stay fully visible, so width is
    accurate exactly where height fails.
    """

    SHAPE = (480, 640, 3)
    FOCAL = 502.0

    def test_close_truncated_person_uses_width(self):
        detection = enrich(BBox(150, 0, 340, 480), "person", 0.9, self.SHAPE, self.FOCAL)
        assert detection.distance_method == "width"
        assert detection.distance_m < 1.0

    def test_close_truncated_person_is_no_longer_reported_as_metres_away(self):
        # The regression: this exact geometry used to read about 2 m.
        detection = enrich(BBox(150, 0, 340, 480), "person", 0.9, self.SHAPE, self.FOCAL)
        assert detection.distance_m < 1.2
        assert detection.proximity is Proximity.IMMEDIATE

    def test_fully_visible_person_still_uses_height(self):
        # Height is the better cue when available: it does not change as a
        # person turns, while apparent width does.
        detection = enrich(BBox(280, 140, 60, 205), "person", 0.9, self.SHAPE, self.FOCAL)
        assert detection.distance_method == "height"
        assert detection.distance_m == pytest.approx(4.2, abs=0.3)

    def test_squat_box_falls_back_to_width_even_without_truncation(self):
        # A hand or forearm detected as "person" gives a roughly square box
        # that touches no frame edge. Measured by height it reads metres away.
        detection = enrich(BBox(300, 200, 130, 150), "person", 0.66, self.SHAPE, self.FOCAL)
        assert detection.distance_method == "width"
        assert detection.distance_m < 3.0

    def test_small_objects_now_get_a_distance(self):
        # "cell phone" had no dimensions at all, so range showed as blank.
        detection = enrich(BBox(300, 190, 60, 120), "cell phone", 0.8, self.SHAPE, self.FOCAL)
        assert detection.distance_m is not None
        assert detection.distance_m < 1.5

    def test_unknown_class_still_has_no_distance(self):
        detection = enrich(BBox(10, 10, 50, 50), "obstacle", 0.5, self.SHAPE, self.FOCAL)
        assert detection.distance_m is None
        assert detection.distance_method == ""

    def test_estimate_reports_which_dimension_it_used(self):
        estimate = estimate_distance(BBox(280, 140, 60, 205), "person", self.FOCAL, 640, 480)
        assert estimate.method in ("height", "width")
        assert estimate.metres is not None

    def test_both_axes_cut_off_reports_an_upper_bound(self):
        estimate = estimate_distance(BBox(0, 0, 640, 480), "person", self.FOCAL, 640, 480)
        assert estimate.is_upper_bound is True

    def test_both_axes_cut_off_takes_the_nearer_reading(self):
        # Under-stating distance is the safe error for an obstacle warning.
        estimate = estimate_distance(BBox(0, 0, 640, 480), "person", self.FOCAL, 640, 480)
        by_height = 1.70 * self.FOCAL / 480
        by_width = 0.50 * self.FOCAL / 640
        assert estimate.metres == pytest.approx(min(by_height, by_width))

    def test_zero_focal_length_yields_nothing(self):
        assert estimate_distance(BBox(0, 0, 10, 10), "person", 0.0, 640, 480).metres is None


class TestHorizontalTruncation:
    def test_box_on_the_left_edge_is_flagged(self):
        assert is_horizontally_truncated(BBox(0, 100, 80, 200), 640) is True

    def test_box_on_the_right_edge_is_flagged(self):
        assert is_horizontally_truncated(BBox(560, 100, 80, 200), 640) is True

    def test_interior_box_is_not_flagged(self):
        assert is_horizontally_truncated(BBox(100, 100, 80, 200), 640) is False

    def test_zero_width_frame_is_handled(self):
        assert is_horizontally_truncated(BBox(0, 0, 10, 10), 0) is False


class TestVerticalTruncation:
    """A box cut off by a frame edge shows only part of the object's height.

    The pinhole estimate then reads it as smaller, therefore further away --
    always under-warning, and always about the nearest obstacles.
    """

    FRAME_SHAPE = (480, 640, 3)

    def test_box_touching_the_top_edge_is_truncated(self):
        assert is_vertically_truncated(BBox(100, 0, 80, 200), 480) is True

    def test_box_touching_the_bottom_edge_is_truncated(self):
        assert is_vertically_truncated(BBox(100, 280, 80, 200), 480) is True

    def test_box_spanning_the_whole_frame_is_truncated(self):
        assert is_vertically_truncated(BBox(0, 0, 640, 480), 480) is True

    def test_box_clear_of_both_edges_is_not_truncated(self):
        assert is_vertically_truncated(BBox(100, 50, 80, 200), 480) is False

    def test_horizontal_clipping_alone_does_not_count(self):
        # Height is what the pinhole estimate divides by, so a box running off
        # the left or right edge is still measured correctly.
        assert is_vertically_truncated(BBox(0, 100, 80, 200), 480) is False

    def test_zero_height_frame_is_handled(self):
        assert is_vertically_truncated(BBox(0, 0, 10, 10), 0) is False

    def test_vertically_cut_box_switches_to_width_and_is_not_a_bound(self):
        # Height is unusable, but the shoulders are fully visible, so the
        # reading is a real estimate rather than an upper bound.
        detection = enrich(
            BBox(280, 0, 100, 480), "person", 0.9, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.distance_method == "width"
        assert detection.truncated is False

    def test_enrich_does_not_flag_a_fully_visible_detection(self):
        detection = enrich(
            BBox(300, 100, 60, 200), "person", 0.9, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.truncated is False

    def test_truncated_close_subject_is_escalated(self):
        # The real-world case: a webcam sees head and shoulders at ~0.5 m. The
        # short box reads as 1.7 m by pinhole (NEAR), but it fills the frame,
        # so it must be reported as IMMEDIATE.
        detection = enrich(
            BBox(220, 0, 200, 480), "person", 0.88, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.proximity is Proximity.IMMEDIATE

    def test_escalation_never_lowers_urgency(self):
        # A truncated object whose pinhole distance is already alarming must
        # keep that reading rather than be softened by the fraction heuristic.
        detection = enrich(
            BBox(300, 0, 40, 100), "person", 0.9, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.proximity >= proximity_from_frame_fraction(
            BBox(300, 0, 40, 100), 480
        )

    def test_distant_truncated_object_is_not_over_escalated(self):
        # A chair whose base sits on the bottom frame edge but occupies little
        # height is genuinely far; escalation must not cry wolf.
        detection = enrich(
            BBox(300, 420, 40, 60), "chair", 0.8, self.FRAME_SHAPE, focal_px=500.0
        )
        assert is_vertically_truncated(detection.bbox, 480) is True
        assert detection.proximity is Proximity.FAR

    def test_unknown_class_is_unaffected_by_the_escalation_path(self):
        # No pinhole distance exists, so the fraction heuristic already governs.
        detection = enrich(
            BBox(0, 0, 640, 480), "obstacle", 0.5, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.distance_m is None
        assert detection.proximity is Proximity.IMMEDIATE

    def test_vertically_cut_distance_comes_from_shoulder_width(self):
        detection = enrich(
            BBox(220, 0, 200, 480), "person", 0.9, self.FRAME_SHAPE, focal_px=500.0
        )
        assert detection.distance_m == pytest.approx(0.50 * 500.0 / 200)
