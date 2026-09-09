"""Colour-threshold fallback detector.

These tests pin the three corrections made to the original algorithm: a
saturation floor that no longer excludes real surfaces, a minimum area that
rejects noise, and one box per object rather than one per nested contour.
"""

from __future__ import annotations

import numpy as np

from safestep.config import ColorDetectorSettings
from safestep.detection.color import ColorThresholdDetector
from safestep.spatial import UNKNOWN_LABEL


class TestBlobDetection:
    def test_finds_a_saturated_blob(self, frame_with_blob):
        detections = ColorThresholdDetector().detect(frame_with_blob)
        assert len(detections) == 1

    def test_box_matches_the_blob_location(self, frame_with_blob):
        # The fixture paints rows 150:350, columns 250:450.
        detection = ColorThresholdDetector().detect(frame_with_blob)[0]
        assert abs(detection.bbox.x - 250) <= 3
        assert abs(detection.bbox.y - 150) <= 3
        assert abs(detection.bbox.w - 200) <= 6
        assert abs(detection.bbox.h - 200) <= 6

    def test_blobs_are_labelled_unknown(self, frame_with_blob):
        # The detector knows something is there, not what it is.
        assert ColorThresholdDetector().detect(frame_with_blob)[0].label == UNKNOWN_LABEL

    def test_confidence_is_within_range(self, frame_with_blob):
        confidence = ColorThresholdDetector().detect(frame_with_blob)[0].confidence
        assert 0.0 < confidence <= 0.99

    def test_two_separated_blobs_yield_two_detections(self, blank_frame):
        frame = blank_frame.copy()
        frame[100:200, 50:150] = (0, 255, 0)
        frame[100:200, 400:500] = (0, 255, 0)
        assert len(ColorThresholdDetector().detect(frame)) == 2


class TestNoiseRejection:
    def test_empty_frame_yields_nothing(self, blank_frame):
        assert ColorThresholdDetector().detect(blank_frame) == []

    def test_single_pixel_speckle_is_filtered(self, blank_frame):
        # The original `if contours:` fired on exactly this.
        frame = blank_frame.copy()
        rng = np.random.default_rng(seed=42)
        for _ in range(200):
            y, x = rng.integers(0, 480), rng.integers(0, 640)
            frame[y, x] = (0, 255, 0)
        assert ColorThresholdDetector().detect(frame) == []

    def test_blob_below_min_area_is_ignored(self, blank_frame):
        frame = blank_frame.copy()
        frame[100:110, 100:110] = (0, 255, 0)  # 100 px, below the 900 px floor
        assert ColorThresholdDetector().detect(frame) == []

    def test_min_area_is_configurable(self, blank_frame):
        frame = blank_frame.copy()
        frame[100:130, 100:130] = (0, 255, 0)  # 900 px
        permissive = ColorThresholdDetector(ColorDetectorSettings(min_area_px=100))
        strict = ColorThresholdDetector(ColorDetectorSettings(min_area_px=5000))
        assert len(permissive.detect(frame)) == 1
        assert strict.detect(frame) == []


class TestNestedContours:
    def test_a_ring_produces_one_box_not_two(self, blank_frame):
        # RETR_TREE returned the outer edge and the hole as separate contours,
        # so a single doughnut-shaped object was announced twice.
        frame = blank_frame.copy()
        frame[100:300, 100:300] = (0, 255, 0)
        frame[160:240, 160:240] = (0, 0, 0)  # punch a hole
        assert len(ColorThresholdDetector().detect(frame)) == 1


class TestThresholds:
    def test_low_saturation_surface_is_now_detectable(self, blank_frame):
        # A muted grey-green wall: saturation ~100, which the original floor of
        # 150 discarded. Exactly the class of surface that matters most.
        frame = blank_frame.copy()
        frame[100:300, 100:300] = (90, 140, 90)
        assert len(ColorThresholdDetector().detect(frame)) == 1

    def test_bright_object_is_not_excluded_by_a_value_ceiling(self, blank_frame):
        # The original upper bound capped value at 180, blanking bright scenes.
        frame = blank_frame.copy()
        frame[100:300, 100:300] = (40, 250, 40)
        assert len(ColorThresholdDetector().detect(frame)) == 1

    def test_hue_bound_respects_opencvs_0_to_179_range(self):
        settings = ColorDetectorSettings()
        assert settings.upper_hsv[0] <= 179

    def test_detector_reports_its_name(self):
        assert ColorThresholdDetector().name == "color"
