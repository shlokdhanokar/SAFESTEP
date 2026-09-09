"""The detector contract.

A detector answers exactly one question -- "what boxes are in this frame?" --
and does nothing else. It does not speak, draw, or track. That separation is
what lets the colour, SSD and YOLO backends be swapped at runtime, and what
makes the pipeline testable with a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import numpy as np

from ..geometry import BBox


@dataclass(frozen=True)
class RawDetection:
    """A detector's output before any spatial reasoning is applied."""

    bbox: BBox
    label: str
    confidence: float


@runtime_checkable
class Detector(Protocol):
    """Anything that can find obstacles in a BGR frame."""

    #: Human-readable backend name, surfaced in logs and the on-screen HUD.
    name: str

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        """Return detections for a single BGR frame."""
        ...


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot run in this environment at all.

    Distinct from :class:`ModelFilesMissing`: the weights may be present and
    correct, but the installed OpenCV build cannot load them. Fetching files
    again will not help, so the message must say what actually needs changing.
    """

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(
            f"The '{backend}' detector cannot run in this environment:\n"
            f"  {reason}\n\n"
            f"Alternatives:\n"
            f"  safestep --detector yolo    (ONNX; supported on all OpenCV 4.x and 5.x builds)\n"
            f"  safestep --detector color   (no model required, demo-grade only)"
        )
        self.backend = backend
        self.reason = reason


class ModelFilesMissing(RuntimeError):
    """Raised when a neural backend's weights are not on disk.

    Carries actionable remediation text rather than a bare traceback, because
    the overwhelmingly likely cause is that the user has not run the fetch
    script yet.
    """

    def __init__(self, backend: str, missing: List[str]) -> None:
        listed = "\n  ".join(str(p) for p in missing)
        super().__init__(
            f"Cannot start the '{backend}' detector - required model files are missing:\n"
            f"  {listed}\n\n"
            f"Fetch them with:\n"
            f"  python scripts/fetch_models.py --model {backend}\n\n"
            f"Or run without a neural model using the colour fallback:\n"
            f"  safestep --detector color"
        )
        self.backend = backend
        self.missing = missing
