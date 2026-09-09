"""Detector backends and the factory that selects one.

Backends are imported lazily inside :func:`build_detector` so that importing
this package never pulls in a model, and a missing or corrupt weight file
cannot break unrelated code paths.
"""

from __future__ import annotations

import logging

from ..config import DetectorSettings
from .base import BackendUnavailable, Detector, ModelFilesMissing, RawDetection

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "yolo", "ssd", "color")

# Preference order used by "auto": best accuracy first, guaranteed-available last.
_AUTO_ORDER = ("yolo", "ssd", "color")


def build_detector(settings: DetectorSettings) -> Detector:
    """Construct the detector named by ``settings.backend``.

    An explicit backend that cannot load raises :class:`ModelFilesMissing` with
    remediation instructions -- silently downgrading a user who asked for a
    neural model would be the wrong call for an assistive device. ``auto``
    instead walks the preference order and falls back, logging each step.
    """
    backend = settings.backend.lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown detector backend {backend!r}; expected one of {BACKENDS}")

    if backend != "auto":
        return _construct(backend, settings)

    for candidate in _AUTO_ORDER:
        try:
            detector = _construct(candidate, settings)
        except ModelFilesMissing:
            logger.info("Detector 'auto': %s weights not present, trying next", candidate)
            continue
        except BackendUnavailable as exc:
            logger.info("Detector 'auto': %s unusable here (%s), trying next", candidate, exc.reason)
            continue
        except Exception:  # noqa: BLE001 - a broken backend must not be fatal in auto mode
            logger.warning("Detector 'auto': %s failed to load, trying next", candidate, exc_info=True)
            continue

        if candidate == "color":
            logger.warning(
                "Falling back to the COLOUR detector - no neural model weights were found. "
                "This backend detects coloured blobs, not obstacles, and is not a safety "
                "mechanism. Run 'python scripts/fetch_models.py' to enable real detection."
            )
        else:
            logger.info("Detector 'auto' selected: %s", candidate)
        return detector

    raise RuntimeError("No detector backend could be constructed")


def _construct(backend: str, settings: DetectorSettings) -> Detector:
    if backend == "yolo":
        from .yolo_onnx import YoloOnnxDetector

        return YoloOnnxDetector(
            model_dir=settings.model_dir,
            confidence=settings.confidence,
            nms_threshold=settings.nms_threshold,
            classes_of_interest=settings.classes_of_interest,
        )

    if backend == "ssd":
        from .mobilenet_ssd import MobileNetSSDDetector

        return MobileNetSSDDetector(
            model_dir=settings.model_dir,
            confidence=settings.confidence,
            classes_of_interest=settings.classes_of_interest,
        )

    if backend == "color":
        from .color import ColorThresholdDetector

        return ColorThresholdDetector(settings.color)

    raise ValueError(f"Unhandled backend {backend!r}")


__all__ = [
    "Detector",
    "RawDetection",
    "ModelFilesMissing",
    "BackendUnavailable",
    "build_detector",
    "BACKENDS",
]
