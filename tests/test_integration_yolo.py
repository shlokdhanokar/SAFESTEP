"""Live YOLO integration test -- skipped unless the ONNX weights are present.

Everything else in this suite runs without model files. This module is the one
exception: it loads the real network and runs a real forward pass, covering the
blob construction and ``cv2.dnn`` plumbing that the synthetic-tensor tests in
test_detection_neural.py deliberately cannot reach.

Get the weights with::

    pip install ultralytics
    python scripts/fetch_models.py --model yolo
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from safestep.detection.yolo_onnx import MODEL_NAME, YoloOnnxDetector
from safestep.spatial import Proximity, enrich, focal_length_px

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / MODEL_NAME

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(),
    reason=f"{MODEL_NAME} not present; run scripts/fetch_models.py --model yolo",
)


@pytest.fixture(scope="module")
def detector() -> YoloOnnxDetector:
    return YoloOnnxDetector(MODEL_DIR, confidence=0.45)


@pytest.fixture(scope="module")
def person_frame() -> np.ndarray:
    """A crude synthetic scene: a tall pale figure on a plain background.

    Not expected to be recognised -- the assertion for this frame is only that
    inference completes and returns well-formed output.
    """
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (280, 140), (360, 400), (170, 160, 150), -1)
    cv2.circle(frame, (320, 120), 34, (175, 165, 155), -1)
    return frame


class TestRealInference:
    def test_forward_pass_completes(self, detector, person_frame):
        assert isinstance(detector.detect(person_frame), list)

    def test_blank_frame_produces_no_spurious_detections(self, detector):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert detector.detect(blank) == []

    def test_output_is_well_formed(self, detector, person_frame):
        for detection in detector.detect(person_frame):
            assert 0.0 <= detection.confidence <= 1.0
            assert detection.bbox.w > 0 and detection.bbox.h > 0
            assert detection.label

    def test_boxes_stay_inside_the_frame(self, detector, person_frame):
        for detection in detector.detect(person_frame):
            assert 0 <= detection.bbox.x <= 640
            assert 0 <= detection.bbox.y <= 480
            assert detection.bbox.x2 <= 640 and detection.bbox.y2 <= 480

    def test_handles_a_non_square_frame(self, detector):
        # Exercises the independent per-axis rescale from 640x640 model space.
        wide = np.full((360, 1280, 3), 128, dtype=np.uint8)
        for detection in detector.detect(wide):
            assert detection.bbox.x2 <= 1280 and detection.bbox.y2 <= 360

    def test_confidence_threshold_is_respected(self, detector, person_frame):
        for detection in detector.detect(person_frame):
            assert detection.confidence >= 0.45

    def test_backend_reports_its_name(self, detector):
        assert detector.name == "yolo"


class TestClassFilter:
    def test_filter_admits_only_requested_classes(self):
        constrained = YoloOnnxDetector(
            MODEL_DIR, confidence=0.25, classes_of_interest=("person",)
        )
        noisy = np.random.default_rng(seed=7).integers(
            0, 255, (480, 640, 3), dtype=np.uint8
        )
        for detection in constrained.detect(noisy):
            assert detection.label == "person"


class TestEndToEndEnrichment:
    def test_detections_flow_through_spatial_enrichment(self, detector, person_frame):
        focal = focal_length_px(640, 65.0)
        for raw in detector.detect(person_frame):
            enriched = enrich(
                raw.bbox, raw.label, raw.confidence, person_frame.shape, focal
            )
            assert isinstance(enriched.proximity, Proximity)
            if enriched.distance_m is not None:
                # A monocular estimate should still land in a sane range for
                # anything visible in a 640x480 frame.
                assert 0.05 < enriched.distance_m < 500.0
