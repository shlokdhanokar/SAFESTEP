"""Neural backend tensor decoding, verified against synthetic model output.

The network forward pass is not exercised here -- that needs weights on disk.
What *is* exercised is the part most likely to be wrong: the transpose, the
centre-form box conversion, the rescale from model input space back to frame
pixels, class mapping and NMS. Feeding hand-built tensors pins that arithmetic
exactly, with no download and no PyTorch.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from safestep.detection.mobilenet_ssd import (
    VOC_CLASSES,
    caffe_support_available,
    decode_ssd_output,
)
from safestep.detection.yolo_onnx import COCO_CLASSES, decode_yolov8_output

# ---------------------------------------------------------------- YOLOv8 ----

NUM_CLASSES = 80
NUM_ANCHORS = 40  # Small stand-in for the real 8400; the maths is identical.


def yolo_output(rows) -> np.ndarray:
    """Build a (1, 84, N) YOLOv8-shaped tensor from (cx, cy, w, h, class, score)."""
    tensor = np.zeros((1, 4 + NUM_CLASSES, NUM_ANCHORS), dtype=np.float32)
    for anchor, (cx, cy, w, h, class_id, score) in enumerate(rows):
        tensor[0, 0, anchor] = cx
        tensor[0, 1, anchor] = cy
        tensor[0, 2, anchor] = w
        tensor[0, 3, anchor] = h
        tensor[0, 4 + class_id, anchor] = score
    return tensor


class TestYoloDecoding:
    def test_empty_output_yields_nothing(self):
        assert decode_yolov8_output(yolo_output([]), 640, 640) == []

    def test_single_confident_box_is_decoded(self):
        # Class 0 is "person"; centre (320, 320), 100x200, at a 640 frame so
        # the scale factor is exactly 1.
        output = yolo_output([(320, 320, 100, 200, 0, 0.9)])
        detections = decode_yolov8_output(output, 640, 640)

        assert len(detections) == 1
        detection = detections[0]
        assert detection.label == "person"
        assert detection.confidence == pytest.approx(0.9)
        assert detection.bbox.x == 270  # 320 - 100/2
        assert detection.bbox.y == 220  # 320 - 200/2
        assert detection.bbox.w == 100
        assert detection.bbox.h == 200

    def test_low_confidence_boxes_are_dropped(self):
        output = yolo_output([(320, 320, 100, 200, 0, 0.2)])
        assert decode_yolov8_output(output, 640, 640, confidence=0.45) == []

    def test_confidence_threshold_is_inclusive_at_the_boundary(self):
        output = yolo_output([(320, 320, 100, 200, 0, 0.5)])
        assert len(decode_yolov8_output(output, 640, 640, confidence=0.5)) == 1

    def test_coordinates_are_rescaled_to_the_frame(self):
        # A 1280x720 frame: x scales by 2.0, y by 1.125.
        output = yolo_output([(320, 320, 100, 200, 0, 0.9)])
        detection = decode_yolov8_output(output, 1280, 720)[0]
        assert detection.bbox.w == pytest.approx(200, abs=1)
        assert detection.bbox.h == pytest.approx(225, abs=1)
        assert detection.bbox.cx == pytest.approx(640, abs=2)

    def test_highest_scoring_class_wins(self):
        tensor = yolo_output([(320, 320, 100, 200, 0, 0.6)])
        tensor[0, 4 + 56, 0] = 0.95  # class 56 is "chair"
        assert decode_yolov8_output(tensor, 640, 640)[0].label == "chair"

    def test_overlapping_duplicates_are_suppressed(self):
        # Three near-identical boxes: NMS must collapse them to one.
        output = yolo_output([
            (320, 320, 100, 200, 0, 0.90),
            (322, 321, 102, 198, 0, 0.85),
            (318, 319, 98, 202, 0, 0.80),
        ])
        assert len(decode_yolov8_output(output, 640, 640)) == 1

    def test_distinct_objects_are_both_kept(self):
        output = yolo_output([
            (150, 320, 80, 160, 0, 0.9),
            (500, 320, 80, 160, 0, 0.9),
        ])
        assert len(decode_yolov8_output(output, 640, 640)) == 2

    def test_class_filter_excludes_other_labels(self):
        output = yolo_output([
            (150, 320, 80, 160, 0, 0.9),   # person
            (500, 320, 80, 160, 56, 0.9),  # chair
        ])
        detections = decode_yolov8_output(
            output, 640, 640, classes_of_interest={"person"}
        )
        assert [d.label for d in detections] == ["person"]

    def test_boxes_are_clipped_to_the_frame(self):
        # Centred on the corner, so half the box falls outside.
        output = yolo_output([(0, 0, 200, 200, 0, 0.9)])
        detections = decode_yolov8_output(output, 640, 640)
        if detections:  # NMS keeps it; the clip is what matters
            assert detections[0].bbox.x >= 0
            assert detections[0].bbox.y >= 0

    def test_class_list_length_matches_the_model_head(self):
        assert len(COCO_CLASSES) == NUM_CLASSES

    def test_navigation_relevant_classes_are_present(self):
        for label in ("person", "chair", "bicycle", "car", "dining table", "couch"):
            assert label in COCO_CLASSES


# ------------------------------------------------------------ MobileNet ----


def ssd_output(rows) -> np.ndarray:
    """Build a (1, 1, N, 7) SSD-shaped tensor from (class_id, score, x1, y1, x2, y2)."""
    tensor = np.zeros((1, 1, len(rows), 7), dtype=np.float32)
    for i, (class_id, score, x1, y1, x2, y2) in enumerate(rows):
        tensor[0, 0, i] = [0, class_id, score, x1, y1, x2, y2]
    return tensor


class TestSsdDecoding:
    def test_empty_output_yields_nothing(self):
        assert decode_ssd_output(ssd_output([]), 640, 480, 0.45) == []

    def test_normalised_corners_become_frame_pixels(self):
        # Class 15 is "person"; corners at 25%..75% of a 640x480 frame.
        output = ssd_output([(15, 0.9, 0.25, 0.25, 0.75, 0.75)])
        detection = decode_ssd_output(output, 640, 480, 0.45)[0]

        assert detection.label == "person"
        assert detection.bbox.x == 160   # 0.25 * 640
        assert detection.bbox.y == 120   # 0.25 * 480
        assert detection.bbox.w == 320   # 0.50 * 640
        assert detection.bbox.h == 240   # 0.50 * 480

    def test_low_confidence_rows_are_dropped(self):
        output = ssd_output([(15, 0.10, 0.25, 0.25, 0.75, 0.75)])
        assert decode_ssd_output(output, 640, 480, 0.45) == []

    def test_background_class_is_never_reported(self):
        # Class 0 is "background" and must never reach the user.
        output = ssd_output([(0, 0.99, 0.1, 0.1, 0.9, 0.9)])
        assert decode_ssd_output(output, 640, 480, 0.45) == []

    def test_out_of_range_class_id_is_ignored(self):
        output = ssd_output([(999, 0.99, 0.1, 0.1, 0.9, 0.9)])
        assert decode_ssd_output(output, 640, 480, 0.45) == []

    def test_class_filter_is_applied(self):
        output = ssd_output([
            (15, 0.9, 0.1, 0.1, 0.3, 0.3),  # person
            (9, 0.9, 0.5, 0.5, 0.7, 0.7),   # chair
        ])
        detections = decode_ssd_output(output, 640, 480, 0.45, {"person"})
        assert [d.label for d in detections] == ["person"]

    def test_overhanging_boxes_are_clipped(self):
        output = ssd_output([(15, 0.9, -0.2, -0.2, 1.5, 1.5)])
        detection = decode_ssd_output(output, 640, 480, 0.45)[0]
        assert detection.bbox.x == 0 and detection.bbox.y == 0
        assert detection.bbox.x2 <= 640 and detection.bbox.y2 <= 480

    def test_degenerate_boxes_are_discarded(self):
        output = ssd_output([(15, 0.9, 0.5, 0.5, 0.5, 0.5)])
        assert decode_ssd_output(output, 640, 480, 0.45) == []

    def test_multiple_detections_are_all_returned(self):
        output = ssd_output([
            (15, 0.9, 0.0, 0.0, 0.2, 0.2),
            (9, 0.8, 0.4, 0.4, 0.6, 0.6),
            (7, 0.7, 0.7, 0.7, 0.9, 0.9),
        ])
        assert len(decode_ssd_output(output, 640, 480, 0.45)) == 3

    def test_class_list_matches_the_voc_head(self):
        assert len(VOC_CLASSES) == 21
        assert VOC_CLASSES[0] == "background"
        assert VOC_CLASSES[15] == "person"


class TestOpenCvCompatibility:
    def test_caffe_support_probe_matches_the_installed_build(self):
        assert caffe_support_available() == hasattr(cv2.dnn, "readNetFromCaffe")

    def test_onnx_reader_is_available_on_every_supported_opencv(self):
        # ONNX is why the YOLO backend is the recommended one: OpenCV 5.0
        # removed the Caffe importer but kept this.
        assert hasattr(cv2.dnn, "readNetFromONNX")

    @pytest.mark.skipif(caffe_support_available(), reason="OpenCV has Caffe support")
    def test_ssd_reports_backend_unavailable_without_caffe(self, tmp_path):
        from safestep.detection.base import BackendUnavailable
        from safestep.detection.mobilenet_ssd import MobileNetSSDDetector

        with pytest.raises(BackendUnavailable) as excinfo:
            MobileNetSSDDetector(model_dir=tmp_path)
        assert "opencv-python<5" in str(excinfo.value)
