"""The SafeStep application loop.

Wires a frame source, a detector, a tracker, the alert policy and the feedback
channels together. Everything it depends on is injected, so the whole loop can
be exercised in a unit test with a fake source and a stub detector -- no camera,
no display, no audio device.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .alerts import AlertPolicy
from .annotate import draw_detections, draw_hud, draw_zone_guides
from .camera import CameraError, FrameSource
from .config import Settings
from .detection import Detector, build_detector
from .feedback import Announcer, build_announcer
from .spatial import Detection, Proximity, enrich, focal_length_px
from .tracking import IoUTracker

logger = logging.getLogger(__name__)

WINDOW_NAME = "SafeStep - obstacle detection"


class FpsMeter:
    """Exponentially smoothed frame rate.

    Smoothed rather than instantaneous because a single slow detection pass
    would otherwise make the HUD unreadable.
    """

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.value = 0.0
        self._last = time.monotonic()

    def reset(self) -> None:
        """Restart timing, discarding any gap before the first real frame."""
        self.value = 0.0
        self._last = time.monotonic()

    def tick(self) -> float:
        now = time.monotonic()
        delta = now - self._last
        self._last = now
        if delta <= 0:
            return self.value
        instant = 1.0 / delta
        self.value = instant if self.value == 0.0 else (
            self.smoothing * self.value + (1.0 - self.smoothing) * instant
        )
        return self.value


class RunClock:
    """Separates camera warm-up from frame-processing time.

    Opening a camera can take tens of seconds on some Windows backends. Folding
    that into the processing duration made the reported frame rate wrong by
    close to an order of magnitude, so the two are measured apart.
    """

    def __init__(self) -> None:
        self._started = time.monotonic()
        self._first_frame_at: Optional[float] = None

    def on_frame(self, fps: "FpsMeter") -> None:
        """Record the arrival of a frame; the first one ends the startup phase."""
        if self._first_frame_at is None:
            self._first_frame_at = time.monotonic()
            fps.reset()

    def finalize(self, stats: "RunStats") -> None:
        """Write the split timings into ``stats``."""
        ended = time.monotonic()
        if self._first_frame_at is None:
            # Never got a frame, so all of it was startup.
            stats.startup_s = ended - self._started
            return
        stats.startup_s = self._first_frame_at - self._started
        stats.elapsed_s = ended - self._first_frame_at


@dataclass
class RunStats:
    """Summary of a completed run, returned for logging and tests."""

    frames: int = 0
    detection_passes: int = 0
    detections_seen: int = 0
    announcements: int = 0
    #: Time spent processing frames, measured from the first frame received.
    elapsed_s: float = 0.0
    #: Time from start until the first frame arrived. Opening a camera can take
    #: tens of seconds on some Windows backends; counting that as processing
    #: time made the reported frame rate meaninglessly low.
    startup_s: float = 0.0
    announcement_log: List[str] = field(default_factory=list)

    @property
    def fps(self) -> float:
        """Processing frame rate, excluding camera startup."""
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def total_s(self) -> float:
        return self.startup_s + self.elapsed_s


class SafeStepApp:
    """Capture, detect, track, alert."""

    def __init__(
        self,
        settings: Settings,
        detector: Detector,
        announcer: Announcer,
        frame_source: Optional[FrameSource] = None,
        tracker: Optional[IoUTracker] = None,
        policy: Optional[AlertPolicy] = None,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.announcer = announcer
        self.frame_source = frame_source or FrameSource(settings.camera)
        self.tracker = tracker or IoUTracker()
        self.policy = policy or AlertPolicy(
            cooldown_s=settings.alerts.cooldown_s,
            min_gap_s=settings.alerts.min_gap_s,
            min_hits=settings.alerts.min_hits,
            min_proximity=Proximity(settings.alerts.min_proximity),
        )
        self._stop = False

    def stop(self) -> None:
        """Request a graceful shutdown after the current frame."""
        self._stop = True

    def run(self) -> RunStats:
        stats = RunStats()
        clock = RunClock()

        detections: List[Detection] = []
        focal_px: Optional[float] = None
        fps = FpsMeter()
        last_announcement: Optional[str] = None

        try:
            for frame_index, frame in enumerate(self.frame_source.frames()):
                if self._stop:
                    break

                clock.on_frame(fps)
                stats.frames += 1

                if focal_px is None:
                    focal_px = self._focal_for(frame)

                # Detection is the expensive step. On constrained hardware it
                # runs every Nth frame and the tracker carries identities
                # across the gaps; positions go slightly stale in between,
                # which is an acceptable trade for real-time throughput.
                if frame_index % self.settings.detect_every_n == 0:
                    detections = self._detect(frame, focal_px)
                    spoken = self._maybe_announce(detections, stats)
                    if spoken is not None:
                        last_announcement = spoken

                fps.tick()

                if not self._preview_step(frame, detections, fps.value, last_announcement):
                    break

                if self._reached_frame_limit(stats):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            clock.finalize(stats)
            self._shutdown()

        logger.info(
            "Run complete: %d frames in %.1fs (%.1f fps), %d detection passes, "
            "%d detections, %d announcements (startup %.1fs)",
            stats.frames, stats.elapsed_s, stats.fps,
            stats.detection_passes, stats.detections_seen, stats.announcements,
            stats.startup_s,
        )
        return stats

    def _focal_for(self, frame: np.ndarray) -> float:
        """Focal length implied by this frame's width and the configured FOV."""
        focal_px = focal_length_px(frame.shape[1], self.settings.horizontal_fov_deg)
        logger.debug(
            "Focal length %.1f px from %d px width at %.1f deg FOV",
            focal_px, frame.shape[1], self.settings.horizontal_fov_deg,
        )
        return focal_px

    def _maybe_announce(self, detections: List[Detection], stats: RunStats) -> Optional[str]:
        """Run the alert policy and emit an announcement if one is warranted."""
        stats.detection_passes += 1
        stats.detections_seen += len(detections)

        announcement = self.policy.select(detections, hits_for=self.tracker.hits_for)
        if announcement is None:
            return None

        self.announcer.announce(announcement)
        stats.announcements += 1
        stats.announcement_log.append(announcement.text)
        logger.info("ALERT: %s", announcement.text)
        return announcement.text

    def _preview_step(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float,
        last_announcement: Optional[str],
    ) -> bool:
        """Update the preview window. Returns False when the user asks to quit."""
        if self.settings.headless:
            return True
        if self._render(frame, detections, fps, last_announcement):
            return True
        logger.info("Quit requested from preview window")
        return False

    def _reached_frame_limit(self, stats: RunStats) -> bool:
        limit = self.settings.max_frames
        if limit is not None and stats.frames >= limit:
            logger.info("Reached max_frames=%d", limit)
            return True
        return False

    def _detect(self, frame: np.ndarray, focal_px: float) -> List[Detection]:
        raw_detections = self.detector.detect(frame)
        enriched = [
            enrich(
                bbox=raw.bbox,
                label=raw.label,
                confidence=raw.confidence,
                frame_shape=frame.shape,
                focal_px=focal_px,
                center_fraction=self.settings.center_fraction,
            )
            for raw in raw_detections
        ]
        return self.tracker.update(enriched)

    def _render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float,
        last_announcement: Optional[str],
    ) -> bool:
        """Draw the preview. Returns False when the user asks to quit."""
        import cv2

        canvas = draw_detections(frame, detections)
        draw_zone_guides(canvas, self.settings.center_fraction)
        draw_hud(canvas, self.detector.name, fps, len(detections), last_announcement)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)  # q or Esc

    def _shutdown(self) -> None:
        self.frame_source.release()
        self.announcer.close()

        if not self.settings.headless:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001 - never let teardown mask a real error
                logger.debug("Error destroying windows", exc_info=True)


def build_app(settings: Settings) -> SafeStepApp:
    """Construct a fully wired app from settings alone."""
    detector = build_detector(settings.detector)
    announcer = build_announcer(settings.alerts)
    return SafeStepApp(settings=settings, detector=detector, announcer=announcer)


def run(settings: Settings) -> int:
    """Build and run the app, returning a process exit code."""
    try:
        app = build_app(settings)
    except CameraError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
        logger.error("%s", exc)
        logger.debug("Startup failure detail", exc_info=True)
        return 2

    try:
        app.run()
    except CameraError as exc:
        logger.error("%s", exc)
        return 2
    return 0
