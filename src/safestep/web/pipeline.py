"""Runs the SafeStep pipeline in a worker thread and publishes telemetry.

The web UI needs the same detections, distances and alerts the audio channels
get -- not a reimplementation. This wraps the existing components (detector,
tracker, alert policy, haptics) and exposes the latest state as a snapshot the
server can broadcast.

Frame and telemetry are published *together* in one snapshot. Sending them
separately would let bounding boxes drift out of sync with the picture they
describe, which looks broken and, in a tool about spatial awareness, misleads.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2

from ..alerts import AlertPolicy
from ..camera import FrameSource
from ..config import Settings
from ..detection import build_detector
from ..haptics import IDLE, HapticCue, strongest_cue
from ..spatial import Detection, Proximity, enrich, focal_length_px
from ..tracking import IoUTracker

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT = 60


@dataclass
class TranscriptEntry:
    """One spoken alert, as shown in the UI's live transcript."""

    text: str
    label: str
    proximity: str
    zone: str
    distance_m: Optional[float]
    at: float

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "proximity": self.proximity,
            "zone": self.zone,
            "distanceM": round(self.distance_m, 2) if self.distance_m else None,
            "at": self.at,
        }


@dataclass
class Snapshot:
    """Everything the UI needs for one frame."""

    frame_jpeg: Optional[str] = None
    width: int = 0
    height: int = 0
    detections: List[dict] = field(default_factory=list)
    haptics: dict = field(default_factory=lambda: IDLE.as_dict())
    alert: Optional[dict] = None
    transcript: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    seq: int = 0

    def as_dict(self) -> dict:
        return {
            "frame": self.frame_jpeg,
            "width": self.width,
            "height": self.height,
            "detections": self.detections,
            "haptics": self.haptics,
            "alert": self.alert,
            "transcript": self.transcript,
            "stats": self.stats,
            "seq": self.seq,
        }


def detection_to_dict(detection: Detection) -> dict:
    box = detection.bbox
    return {
        "x": box.x,
        "y": box.y,
        "w": box.w,
        "h": box.h,
        "label": detection.label,
        "confidence": round(detection.confidence, 3),
        "zone": detection.zone.value,
        "proximity": detection.proximity.name,
        "proximityRank": int(detection.proximity),
        "distanceM": round(detection.distance_m, 2) if detection.distance_m else None,
        "truncated": detection.truncated,
        "distanceMethod": detection.distance_method,
        "trackId": detection.track_id,
    }


class TelemetryPipeline:
    """Drives detection in a background thread and keeps the latest snapshot."""

    def __init__(self, settings: Settings, jpeg_quality: int = 72) -> None:
        self.settings = settings
        self.jpeg_quality = jpeg_quality

        self._detector = build_detector(settings.detector)
        self._tracker = IoUTracker()
        self._policy = AlertPolicy(
            cooldown_s=settings.alerts.cooldown_s,
            min_gap_s=settings.alerts.min_gap_s,
            min_hits=settings.alerts.min_hits,
            min_proximity=Proximity(settings.alerts.min_proximity),
        )

        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._transcript: List[TranscriptEntry] = []
        self._new_frame = threading.Condition(self._lock)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._error: Optional[str] = None
        self._seq = 0
        self._mirror = settings.camera.mirror

    @property
    def mirror(self) -> bool:
        return self._mirror

    def set_mirror(self, enabled: bool) -> None:
        """Toggle horizontal flip. Takes effect on the next frame."""
        self._mirror = bool(enabled)

    # ---------------------------------------------------------------- state

    @property
    def detector_name(self) -> str:
        return self._detector.name

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0) -> Optional[Snapshot]:
        """Block until a snapshot newer than ``last_seq`` exists.

        Lets a client consume at the pipeline's pace instead of polling, and
        means a slow client simply skips frames rather than backing up a queue.
        """
        with self._new_frame:
            if self._snapshot.seq <= last_seq:
                self._new_frame.wait(timeout)
            if self._snapshot.seq <= last_seq:
                return None
            return self._snapshot

    # -------------------------------------------------------------- control

    def start(self) -> "TelemetryPipeline":
        if self.running:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="safestep-pipeline", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

        # Clear the picture so a stopped feed cannot be mistaken for a live one
        # showing a stationary scene.
        with self._new_frame:
            self._snapshot = Snapshot(seq=self._snapshot.seq + 1)
            self._new_frame.notify_all()

    def restart(self) -> None:
        """Stop and start again, releasing and re-acquiring the camera."""
        self.stop()
        self._tracker.reset()
        self._policy.reset()
        self._error = None
        self.start()

    # ----------------------------------------------------------------- loop

    def _run(self) -> None:
        source = FrameSource(self.settings.camera)
        source.mirror = self._mirror
        focal_px: Optional[float] = None
        detections: List[Detection] = []
        frame_index = 0
        last_tick = time.monotonic()
        fps = 0.0

        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break

                # Re-read each frame so the UI toggle takes effect immediately.
                source.mirror = self._mirror

                if focal_px is None:
                    focal_px = focal_length_px(
                        frame.shape[1], self.settings.horizontal_fov_deg
                    )

                alert_payload: Optional[dict] = None
                if frame_index % self.settings.detect_every_n == 0:
                    detections = self._detect(frame, focal_px)
                    alert_payload = self._maybe_alert(detections)

                now = time.monotonic()
                delta = now - last_tick
                last_tick = now
                if delta > 0:
                    instant = 1.0 / delta
                    fps = instant if fps == 0.0 else 0.85 * fps + 0.15 * instant

                self._publish(frame, detections, alert_payload, fps)
                frame_index += 1

        except Exception as exc:  # noqa: BLE001 - surface it to the UI, do not crash the server
            logger.exception("Pipeline stopped")
            self._error = str(exc)
        finally:
            source.release()
            logger.info("Pipeline thread finished after %d frames", frame_index)

    def _detect(self, frame, focal_px: float) -> List[Detection]:
        enriched = [
            enrich(
                bbox=raw.bbox,
                label=raw.label,
                confidence=raw.confidence,
                frame_shape=frame.shape,
                focal_px=focal_px,
                center_fraction=self.settings.center_fraction,
            )
            for raw in self._detector.detect(frame)
        ]
        return self._tracker.update(enriched)

    def _maybe_alert(self, detections: List[Detection]) -> Optional[dict]:
        announcement = self._policy.select(detections, hits_for=self._tracker.hits_for)
        if announcement is None:
            return None

        entry = TranscriptEntry(
            text=announcement.text,
            label=announcement.label,
            proximity=announcement.proximity.name,
            zone=announcement.zone.value,
            distance_m=announcement.distance_m,
            at=time.time(),
        )
        self._transcript.append(entry)
        del self._transcript[:-MAX_TRANSCRIPT]
        logger.info("ALERT: %s", announcement.text)
        return entry.as_dict()

    def _publish(self, frame, detections: List[Detection], alert: Optional[dict], fps: float) -> None:
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return

        cue: HapticCue = strongest_cue(detections)
        height, width = frame.shape[:2]

        self._seq += 1
        snapshot = Snapshot(
            frame_jpeg=base64.b64encode(buffer).decode("ascii"),
            width=width,
            height=height,
            detections=[detection_to_dict(d) for d in detections],
            haptics=cue.as_dict(),
            alert=alert,
            transcript=[e.as_dict() for e in self._transcript[-12:]],
            stats=self._stats(fps, len(detections)),
            seq=self._seq,
        )

        with self._new_frame:
            self._snapshot = snapshot
            self._new_frame.notify_all()

    def _stats(self, fps: float, detection_count: int) -> Dict[str, object]:
        return {
            "fps": round(fps, 1),
            "detector": self._detector.name,
            "objects": detection_count,
            "detectEveryN": self.settings.detect_every_n,
            "hfov": self.settings.horizontal_fov_deg,
            "centerFraction": self.settings.center_fraction,
            "source": str(self.settings.camera.source),
            "mirror": self._mirror,
            "running": True,
        }
