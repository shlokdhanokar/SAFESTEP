"""YOLOv8 obstacle detector, PyTorch-free via ``cv2.dnn``.

Deliberately loads an exported ``.onnx`` rather than using ``ultralytics``:
that keeps PyTorch (~2 GB of dependencies) off the device while still giving
markedly better accuracy than MobileNet-SSD on small and overlapping objects.
The ``yolov8n`` export is around 12 MB.

ONNX is also the only format supported across both OpenCV 4.x and 5.x, which
makes this the recommended backend -- OpenCV 5.0 dropped the Caffe importer the
SSD backend relies on.

Tensor decoding lives in :func:`decode_yolov8_output`, a pure function kept
separate from network I/O so the fiddly parts -- the transpose, the centre-form
box conversion, the letterbox-free rescale and NMS -- can be verified against
synthetic tensors with no weights on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Container, List, Optional, Sequence

import cv2
import numpy as np

from ..geometry import BBox
from .base import ModelFilesMissing, RawDetection

logger = logging.getLogger(__name__)

MODEL_NAME = "yolov8n.onnx"
INPUT_SIZE = 640

COCO_CLASSES: Sequence[str] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def decode_yolov8_output(
    output: np.ndarray,
    frame_width: int,
    frame_height: int,
    confidence: float = 0.45,
    nms_threshold: float = 0.45,
    classes_of_interest: Optional[Container[str]] = None,
    input_size: int = INPUT_SIZE,
) -> List[RawDetection]:
    """Convert a raw YOLOv8 forward pass into detections.

    ``output`` has shape ``(1, 4 + num_classes, num_anchors)``: rows 0-3 are the
    box in centre form (``cx, cy, w, h``) in input-space pixels, and the rest
    are per-class scores. YOLOv8 has no separate objectness score, so the class
    score *is* the confidence.
    """
    # (1, 84, 8400) -> (8400, 84): one candidate per row.
    predictions = np.squeeze(output, axis=0).T

    x_factor = frame_width / input_size
    y_factor = frame_height / input_size

    # Vectorised prefilter. Scoring 8400 candidates in a Python loop is the
    # difference between usable and unusable on a Pi.
    class_scores = predictions[:, 4:]
    best_class = np.argmax(class_scores, axis=1)
    best_score = class_scores[np.arange(class_scores.shape[0]), best_class]

    keep = best_score >= confidence
    if not np.any(keep):
        return []

    boxes_cxcywh = predictions[keep, :4]
    scores = best_score[keep]
    class_ids = best_class[keep]

    # cx, cy, w, h (input-space pixels) -> x, y, w, h in frame pixels. The
    # blob is built with a plain resize rather than a letterbox, so recovering
    # frame coordinates is one independent scale factor per axis.
    cx, cy, bw, bh = boxes_cxcywh.T
    xs = (cx - bw / 2.0) * x_factor
    ys = (cy - bh / 2.0) * y_factor
    ws = bw * x_factor
    hs = bh * y_factor

    nms_boxes = [
        [float(x), float(y), float(w), float(h)] for x, y, w, h in zip(xs, ys, ws, hs)
    ]
    # Score threshold 0.0, not `confidence`: the vectorised prefilter above is
    # the single source of truth for confidence. NMSBoxes compares strictly
    # greater-than, so passing `confidence` here would silently re-drop a
    # candidate scoring exactly at the threshold.
    indices = cv2.dnn.NMSBoxes(
        nms_boxes, scores.astype(float).tolist(), 0.0, nms_threshold
    )
    if len(indices) == 0:
        return []

    results: List[RawDetection] = []
    for idx in np.array(indices).flatten():
        idx = int(idx)
        class_id = int(class_ids[idx])
        if not 0 <= class_id < len(COCO_CLASSES):
            continue
        label = COCO_CLASSES[class_id]
        if classes_of_interest and label.lower() not in classes_of_interest:
            continue

        x, y, w, h = nms_boxes[idx]
        bbox = BBox.from_xyxy(x, y, x + w, y + h).clipped(frame_width, frame_height)
        if bbox.area <= 0:
            continue

        results.append(RawDetection(bbox=bbox, label=label, confidence=float(scores[idx])))

    return results


class YoloOnnxDetector:
    """YOLOv8 inference through OpenCV's DNN module."""

    name = "yolo"

    def __init__(
        self,
        model_dir: Path,
        confidence: float = 0.45,
        nms_threshold: float = 0.45,
        classes_of_interest: Sequence[str] = (),
    ) -> None:
        model_path = Path(model_dir) / MODEL_NAME
        if not model_path.is_file():
            raise ModelFilesMissing("yolo", [str(model_path)])

        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.classes_of_interest = {c.lower() for c in classes_of_interest}
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        logger.info("Loaded YOLOv8 ONNX from %s", model_path)

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()

        return decode_yolov8_output(
            output,
            frame_width=width,
            frame_height=height,
            confidence=self.confidence,
            nms_threshold=self.nms_threshold,
            classes_of_interest=self.classes_of_interest,
        )
