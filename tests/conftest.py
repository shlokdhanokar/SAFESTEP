"""Shared fixtures and fakes.

Everything here is hardware-free by construction: no camera is opened, no model
is loaded and no audio device is touched. That is a hard requirement -- the
suite must pass in CI and on a developer laptop with nothing plugged in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pytest

# Make the src/ layout importable without requiring an editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safestep.alerts import Announcement  # noqa: E402
from safestep.detection.base import RawDetection  # noqa: E402
from safestep.geometry import BBox  # noqa: E402
from safestep.spatial import Detection, Proximity, Zone  # noqa: E402


class FakeClock:
    """A manually advanced monotonic clock, for deterministic timing tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingAnnouncer:
    """Captures announcements instead of making sound."""

    def __init__(self) -> None:
        self.announcements: List[Announcement] = []
        self.closed = False

    def announce(self, announcement: Announcement) -> None:
        self.announcements.append(announcement)

    def close(self) -> None:
        self.closed = True

    @property
    def texts(self) -> List[str]:
        return [a.text for a in self.announcements]


class StubDetector:
    """Returns a scripted list of detections per call."""

    name = "stub"

    def __init__(self, script: Optional[List[List[RawDetection]]] = None) -> None:
        self.script = script if script is not None else []
        self.calls = 0

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        index = min(self.calls, len(self.script) - 1) if self.script else -1
        self.calls += 1
        if index < 0:
            return []
        return self.script[index]


class FakeFrameSource:
    """Yields a fixed list of frames, mimicking the FrameSource interface."""

    def __init__(self, frames: List[np.ndarray]) -> None:
        self._frames = frames
        self.released = False

    def frames(self) -> Iterator[np.ndarray]:
        for frame in self._frames:
            yield frame

    def release(self) -> None:
        self.released = True


def make_detection(
    x: int = 100,
    y: int = 100,
    w: int = 50,
    h: int = 100,
    label: str = "person",
    confidence: float = 0.9,
    zone: Zone = Zone.CENTER,
    proximity: Proximity = Proximity.NEAR,
    distance_m: Optional[float] = 2.0,
    track_id: Optional[int] = None,
) -> Detection:
    """Build a Detection directly, bypassing enrichment."""
    return Detection(
        bbox=BBox(x, y, w, h),
        label=label,
        confidence=confidence,
        zone=zone,
        proximity=proximity,
        distance_m=distance_m,
        track_id=track_id,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def announcer() -> RecordingAnnouncer:
    return RecordingAnnouncer()


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A 640x480 black BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_with_blob() -> np.ndarray:
    """A black frame containing one saturated green rectangle.

    Chosen to sit inside the colour detector's HSV window: pure green is hue 60
    in OpenCV terms, with maximum saturation.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[150:350, 250:450] = (0, 255, 0)  # BGR green
    return frame
