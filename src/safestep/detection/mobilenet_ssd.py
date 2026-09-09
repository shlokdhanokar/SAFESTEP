"""MobileNet-SSD (Caffe) obstacle detector via ``cv2.dnn``.

Twenty PASCAL VOC classes and roughly 23 MB of weights, historically the
workhorse for Raspberry Pi vision.

.. important::
   **This backend requires OpenCV 4.x.** OpenCV 5.0 removed the Caffe importer
   (``cv2.dnn.readNetFromCaffe`` no longer exists), so on a 5.x build this
   detector reports :class:`BackendUnavailable` and you should use the ONNX
   YOLO backend instead. The check is made at construction rather than at
   import so the rest of SafeStep stays usable either way.

Tensor decoding lives in :func:`decode_ssd_output`, a pure function kept
separate from network I/O so it can be verified against synthetic tensors
without any weights on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Container, List, Optional, Sequence

import cv2
import numpy as np

from ..geometry import BBox
from .base import BackendUnavailable, ModelFilesMissing, RawDetection

logger = logging.getLogger(__name__)

PROTOTXT_NAME = "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_NAME = "MobileNetSSD_deploy.caffemodel"

# Index order is fixed by the trained network; do not reorder.
VOC_CLASSES: Sequence[str] = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# Preprocessing constants baked into the published Caffe model.
INPUT_SIZE = (300, 300)
SCALE = 0.007843  # 1 / 127.5
MEAN = 127.5


def caffe_support_available() -> bool:
    """Whether the installed OpenCV build can load Caffe models."""
    return hasattr(cv2.dnn, "readNetFromCaffe")


def decode_ssd_output(
    raw: np.ndarray,
    frame_width: int,
    frame_height: int,
    confidence: float,
    classes_of_interest: Optional[Container[str]] = None,
) -> List[RawDetection]:
    """Convert a raw SSD forward pass into detections.

    ``raw`` has shape ``(1, 1, N, 7)`` where each row is
    ``[batch, class_id, score, x1, y1, x2, y2]`` with corners normalised
    to ``0..1``.
    """
    results: List[RawDetection] = []

    for i in range(raw.shape[2]):
        score = float(raw[0, 0, i, 2])
        if score < confidence:
            continue

        class_id = int(raw[0, 0, i, 1])
        if not 0 <= class_id < len(VOC_CLASSES):
            continue
        label = VOC_CLASSES[class_id]
        if label == "background":
            continue
        if classes_of_interest and label.lower() not in classes_of_interest:
            continue

        x1, y1, x2, y2 = raw[0, 0, i, 3:7] * np.array(
            [frame_width, frame_height, frame_width, frame_height]
        )
        bbox = BBox.from_xyxy(x1, y1, x2, y2).clipped(frame_width, frame_height)
        if bbox.area <= 0:
            continue

        results.append(RawDetection(bbox=bbox, label=label, confidence=score))

    return results


class MobileNetSSDDetector:
    """Single-shot detector wrapping the Caffe MobileNet-SSD model."""

    name = "ssd"

    def __init__(
        self,
        model_dir: Path,
        confidence: float = 0.45,
        classes_of_interest: Sequence[str] = (),
    ) -> None:
        if not caffe_support_available():
            raise BackendUnavailable(
                "ssd",
                f"OpenCV {cv2.__version__} has no Caffe importer "
                f"(cv2.dnn.readNetFromCaffe was removed in OpenCV 5.0). "
                f"MobileNet-SSD needs an OpenCV 4.x build: pip install 'opencv-python<5'",
            )

        prototxt = Path(model_dir) / PROTOTXT_NAME
        caffemodel = Path(model_dir) / CAFFEMODEL_NAME

        missing = [p for p in (prototxt, caffemodel) if not p.is_file()]
        if missing:
            raise ModelFilesMissing("ssd", [str(p) for p in missing])

        self.confidence = confidence
        self.classes_of_interest = {c.lower() for c in classes_of_interest}
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        logger.info("Loaded MobileNet-SSD from %s", model_dir)

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, INPUT_SIZE), SCALE, INPUT_SIZE, MEAN)
        self._net.setInput(blob)
        raw = self._net.forward()

        return decode_ssd_output(
            raw, width, height, self.confidence, self.classes_of_interest
        )
