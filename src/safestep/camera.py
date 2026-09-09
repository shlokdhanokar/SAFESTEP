"""Frame acquisition from a camera or a video file.

Accepting a file path as well as a device index is what makes the full pipeline
runnable -- and therefore testable and demonstrable -- on a machine with no
camera attached.

Two behaviours are specific to the assistive use case:

* **The driver buffer is pinned to one frame.** Otherwise a slow detection pass
  lets frames accumulate and the user is warned about where the obstacle was a
  second ago. Stale guidance is dangerous guidance.
* **Transient read failures trigger a reconnect** rather than terminating. A USB
  camera on a wearable device gets knocked, and the correct response is to
  recover, not to stop guiding the user mid-walk.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

import cv2
import numpy as np

from .config import CameraSettings

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when a frame source cannot be opened at all."""


class FrameSource:
    """Context manager yielding BGR frames from a camera index or video file."""

    def __init__(self, settings: Optional[CameraSettings] = None) -> None:
        self.settings = settings or CameraSettings()
        self._capture: Optional[cv2.VideoCapture] = None
        self._frames_read = 0
        #: Horizontal flip, applied before anything downstream sees the frame so
        #: the picture, the zones and the haptics all agree on which side is
        #: which. Mutable so the web UI can toggle it live.
        self.mirror = self.settings.mirror

    @property
    def frames_read(self) -> int:
        return self._frames_read

    @property
    def is_file(self) -> bool:
        return self.settings.is_file

    def open(self) -> "FrameSource":
        self._capture = self._open_capture()
        return self

    def _open_capture(self) -> cv2.VideoCapture:
        source = self.settings.source
        capture = cv2.VideoCapture(source)

        if not capture.isOpened():
            capture.release()
            if isinstance(source, int):
                raise CameraError(
                    f"Could not open camera index {source}. Check that a camera is connected "
                    f"and not in use by another application, or pass --source with a different "
                    f"index or a video file path."
                )
            raise CameraError(f"Could not open video source {source!r}. Check the path exists.")

        # Resolution and buffering only apply to live devices; forcing them on a
        # file source is either ignored or actively harmful.
        if not self.settings.is_file:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
            capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, self.settings.buffer_size)
            except cv2.error:  # pragma: no cover - unsupported on some backends
                logger.debug("Backend does not support CAP_PROP_BUFFERSIZE")

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Frame source %r opened at %dx%d", source, actual_w, actual_h)
        return capture

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the source ends or is exhausted after reconnects."""
        if self._capture is None:
            self.open()

        attempts = 0
        while True:
            ok, frame = self._capture.read()

            if ok and frame is not None:
                attempts = 0
                self._frames_read += 1
                yield cv2.flip(frame, 1) if self.mirror else frame
                continue

            # A file simply ended; that is a normal, expected stop.
            if self.settings.is_file:
                logger.info("End of video source after %d frames", self._frames_read)
                return

            attempts += 1
            if attempts > self.settings.reconnect_attempts:
                logger.error(
                    "Camera read failed %d times in a row; giving up", attempts - 1
                )
                return

            logger.warning(
                "Camera read failed (attempt %d/%d), reconnecting...",
                attempts,
                self.settings.reconnect_attempts,
            )
            time.sleep(self.settings.reconnect_delay_s)
            self.release()
            try:
                self._capture = self._open_capture()
            except CameraError:
                logger.warning("Reconnect failed", exc_info=True)

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
