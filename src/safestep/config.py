"""Typed configuration with hardware profiles.

Every tunable lives here rather than as a module-level constant, so behaviour
can be changed from the CLI or a test without editing source. Settings are
frozen: the running app never mutates its own configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Tuple

# Repository root, resolved from this file: src/safestep/config.py -> repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class CameraSettings:
    """Frame source configuration.

    ``source`` is either a camera index (``int``) or a path to a video file.
    File sources make the whole pipeline runnable, and therefore verifiable,
    with no hardware attached.
    """

    source: object = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    # Keep the driver buffer at a single frame. Without this, a slow detection
    # pass lets frames queue up and the user is warned about where they were a
    # second ago rather than where they are.
    buffer_size: int = 1
    reconnect_attempts: int = 3
    reconnect_delay_s: float = 0.5
    # Flip horizontally before detection. A wearable camera faces away from the
    # user, so image-left is the user's left and no flip is wanted. A laptop or
    # phone webcam faces the user, so the two are reversed and "on your left"
    # comes out backwards -- turn this on when testing with a selfie camera.
    mirror: bool = False

    @property
    def is_file(self) -> bool:
        return not isinstance(self.source, int)


@dataclass(frozen=True)
class ColorDetectorSettings:
    """HSV threshold fallback.

    The historical defaults required saturation >= 150, which excludes walls,
    concrete, doorframes and most real hazards. The floor is lowered here and
    the hue ceiling corrected to 179 (OpenCV's actual hue range), but this
    backend remains a demo-grade fallback -- see the README safety notes.
    """

    lower_hsv: Tuple[int, int, int] = (30, 80, 40)
    upper_hsv: Tuple[int, int, int] = (179, 255, 255)
    min_area_px: int = 900
    # Morphological opening removes speckle noise; closing fills small holes so
    # one object yields one contour instead of several.
    open_kernel: int = 5
    close_kernel: int = 9


@dataclass(frozen=True)
class DetectorSettings:
    """Neural backend configuration."""

    backend: str = "auto"  # auto | yolo | ssd | color
    model_dir: Path = DEFAULT_MODEL_DIR
    confidence: float = 0.45
    nms_threshold: float = 0.45
    # Classes worth announcing. Empty means "announce everything the model
    # emits" -- useful for debugging, noisy in practice.
    classes_of_interest: Tuple[str, ...] = ()
    color: ColorDetectorSettings = field(default_factory=ColorDetectorSettings)


@dataclass(frozen=True)
class AlertSettings:
    """How obstacle reports become sound."""

    # Per-object silence window. The same tracked obstacle is not re-announced
    # inside this many seconds.
    cooldown_s: float = 4.0
    # Floor on the gap between any two announcements, regardless of object.
    min_gap_s: float = 1.2
    # An obstacle must be seen this many consecutive detection passes before it
    # is announced, which suppresses single-frame false positives.
    min_hits: int = 2
    speech_enabled: bool = True
    earcons_enabled: bool = True
    speech_rate: int = 165
    speech_volume: float = 1.0
    # Bands quieter than this are not announced at all.
    min_proximity: int = 1  # Proximity.MODERATE


@dataclass(frozen=True)
class Settings:
    """Top-level application configuration."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)

    headless: bool = False
    # Run the detector every Nth frame and coast on the tracker in between.
    # The dominant performance lever on constrained hardware.
    detect_every_n: int = 1
    # Camera horizontal field of view, degrees. Drives the pinhole distance
    # estimate; measure yours for accurate distances.
    horizontal_fov_deg: float = 65.0
    center_fraction: float = 0.34
    log_level: str = "INFO"
    # Stop after this many frames. None runs until the source ends or the user
    # quits; used by tests and demos.
    max_frames: Optional[int] = None

    def with_profile(self, profile: str) -> "Settings":
        """Return a copy tuned for ``profile`` ("desktop" or "pi")."""
        if profile == "desktop":
            return replace(
                self,
                headless=False,
                detect_every_n=1,
                camera=replace(self.camera, width=640, height=480),
            )
        if profile == "pi":
            # Smaller frames and detection every third frame keep a Pi 4 near
            # real time; the tracker maintains identities across skipped frames.
            return replace(
                self,
                headless=True,
                detect_every_n=3,
                camera=replace(self.camera, width=480, height=360, fps=15),
            )
        raise ValueError(f"Unknown profile {profile!r}; expected 'desktop' or 'pi'")


PROFILES = ("desktop", "pi")
