"""End-to-end loop behaviour, driven entirely by fakes.

These are the tests that would have caught the original design's central flaw:
the loop is exercised with a detector that always fires, and the assertion is
that the user is *not* told about it on every frame.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeFrameSource, RecordingAnnouncer, StubDetector

from safestep.alerts import AlertPolicy
from safestep.app import SafeStepApp
from safestep.config import AlertSettings, Settings
from safestep.detection.base import RawDetection
from safestep.geometry import BBox
from safestep.spatial import Proximity


def frames(count: int, height: int = 480, width: int = 640):
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)]


def person_at(x: int, y: int = 150, w: int = 80, h: int = 300) -> RawDetection:
    """A person box large enough to land in an announceable proximity band."""
    return RawDetection(bbox=BBox(x, y, w, h), label="person", confidence=0.9)


def build(
    detector,
    frame_count: int = 10,
    announcer=None,
    settings: Settings = None,
) -> tuple:
    announcer = announcer or RecordingAnnouncer()
    settings = settings or Settings(
        headless=True,
        alerts=AlertSettings(speech_enabled=False, earcons_enabled=False, min_hits=1),
    )
    app = SafeStepApp(
        settings=settings,
        detector=detector,
        announcer=announcer,
        frame_source=FakeFrameSource(frames(frame_count)),
        policy=AlertPolicy(
            cooldown_s=settings.alerts.cooldown_s,
            min_gap_s=settings.alerts.min_gap_s,
            min_hits=settings.alerts.min_hits,
            min_proximity=Proximity(settings.alerts.min_proximity),
        ),
    )
    return app, announcer


class TestLoopMechanics:
    def test_processes_every_frame(self):
        app, _ = build(StubDetector([[]]), frame_count=7)
        assert app.run().frames == 7

    def test_terminates_when_the_source_ends(self):
        app, _ = build(StubDetector([[]]), frame_count=3)
        stats = app.run()
        assert stats.frames == 3 and stats.elapsed_s >= 0

    def test_empty_source_is_handled(self):
        app, _ = build(StubDetector([[]]), frame_count=0)
        assert app.run().frames == 0

    def test_max_frames_stops_early(self):
        settings = Settings(
            headless=True,
            max_frames=4,
            alerts=AlertSettings(speech_enabled=False, earcons_enabled=False),
        )
        app, _ = build(StubDetector([[]]), frame_count=50, settings=settings)
        assert app.run().frames == 4

    def test_stop_requests_a_graceful_exit(self):
        app, _ = build(StubDetector([[]]), frame_count=50)
        app.stop()
        assert app.run().frames == 0

    def test_resources_are_released_on_exit(self):
        announcer = RecordingAnnouncer()
        app, _ = build(StubDetector([[]]), frame_count=3, announcer=announcer)
        app.run()
        assert app.frame_source.released is True
        assert announcer.closed is True

    def test_no_hardware_is_touched(self):
        # The whole point: this suite must pass with nothing plugged in.
        app, _ = build(StubDetector([[person_at(280)]]), frame_count=5)
        app.run()


class TestDetectionIntegration:
    def test_detections_are_counted(self):
        detector = StubDetector([[person_at(100), person_at(400)]])
        app, _ = build(detector, frame_count=3)
        stats = app.run()
        assert stats.detection_passes == 3
        assert stats.detections_seen == 6

    def test_detector_is_called_once_per_frame_by_default(self):
        detector = StubDetector([[]])
        app, _ = build(detector, frame_count=6)
        app.run()
        assert detector.calls == 6

    def test_frame_skipping_reduces_detector_calls(self):
        settings = Settings(
            headless=True,
            detect_every_n=3,
            alerts=AlertSettings(speech_enabled=False, earcons_enabled=False),
        )
        detector = StubDetector([[]])
        app, _ = build(detector, frame_count=9, settings=settings)
        stats = app.run()
        assert stats.frames == 9
        assert detector.calls == 3  # frames 0, 3, 6

    def test_tracking_assigns_stable_ids(self):
        detector = StubDetector([[person_at(300)]])
        app, _ = build(detector, frame_count=5)
        app.run()
        assert app.tracker.hits_for(1) == 5


class TestAlertBehaviour:
    def test_a_constant_obstacle_is_not_announced_every_frame(self):
        # The original code spoke on every frame containing a contour.
        detector = StubDetector([[person_at(280)]])
        app, announcer = build(detector, frame_count=30)
        stats = app.run()
        assert stats.detections_seen == 30
        assert stats.announcements <= 2

    def test_an_obstacle_is_announced_at_least_once(self):
        detector = StubDetector([[person_at(280)]])
        app, announcer = build(detector, frame_count=10)
        app.run()
        assert len(announcer.announcements) >= 1

    def test_nothing_is_announced_for_an_empty_scene(self):
        app, announcer = build(StubDetector([[]]), frame_count=20)
        stats = app.run()
        assert stats.announcements == 0
        assert announcer.announcements == []

    def test_announcement_text_names_the_object(self):
        detector = StubDetector([[person_at(280)]])
        app, announcer = build(detector, frame_count=5)
        app.run()
        assert announcer.announcements[0].label == "person"
        assert "person" in announcer.announcements[0].text

    def test_announcement_log_is_recorded_in_stats(self):
        detector = StubDetector([[person_at(280)]])
        app, _ = build(detector, frame_count=5)
        stats = app.run()
        assert len(stats.announcement_log) == stats.announcements

    def test_distant_obstacle_is_not_announced(self):
        # A small box implies a far object, below the default reporting floor.
        detector = StubDetector([[RawDetection(BBox(300, 200, 6, 18), "person", 0.9)]])
        app, announcer = build(detector, frame_count=10)
        assert app.run().announcements == 0
        assert announcer.announcements == []

    def test_min_hits_suppresses_a_single_frame_false_positive(self):
        settings = Settings(
            headless=True,
            alerts=AlertSettings(speech_enabled=False, earcons_enabled=False, min_hits=5),
        )
        detector = StubDetector([[person_at(280)], []])  # seen once, then gone
        app, _ = build(detector, frame_count=3, settings=settings)
        assert app.run().announcements == 0


class SlowStartFrameSource:
    """Frame source that stalls before yielding, mimicking camera warm-up."""

    def __init__(self, frames_list, startup_delay: float = 0.25) -> None:
        self._frames = frames_list
        self._delay = startup_delay
        self.released = False

    def frames(self):
        import time

        time.sleep(self._delay)
        for frame in self._frames:
            yield frame

    def release(self) -> None:
        self.released = True


class TestStats:
    def test_fps_is_reported(self):
        app, _ = build(StubDetector([[]]), frame_count=5)
        assert app.run().fps >= 0

    def test_fps_is_zero_for_an_empty_run(self):
        app, _ = build(StubDetector([[]]), frame_count=0)
        assert app.run().fps == 0.0

    def test_camera_startup_is_excluded_from_the_frame_rate(self):
        # Opening a camera can take tens of seconds on some Windows backends.
        # Counting that as processing time made the reported fps meaningless.
        settings = Settings(
            headless=True,
            alerts=AlertSettings(speech_enabled=False, earcons_enabled=False),
        )
        app = SafeStepApp(
            settings=settings,
            detector=StubDetector([[]]),
            announcer=RecordingAnnouncer(),
            frame_source=SlowStartFrameSource(frames(10), startup_delay=0.3),
        )
        stats = app.run()

        assert stats.startup_s >= 0.25
        assert stats.elapsed_s < 0.25       # processing was fast
        assert stats.total_s >= stats.startup_s

    def test_startup_is_recorded_even_when_no_frame_arrives(self):
        settings = Settings(
            headless=True,
            alerts=AlertSettings(speech_enabled=False, earcons_enabled=False),
        )
        app = SafeStepApp(
            settings=settings,
            detector=StubDetector([[]]),
            announcer=RecordingAnnouncer(),
            frame_source=SlowStartFrameSource([], startup_delay=0.15),
        )
        stats = app.run()

        assert stats.frames == 0
        assert stats.startup_s >= 0.15
        assert stats.elapsed_s == 0.0
        assert stats.fps == 0.0

    def test_total_time_is_startup_plus_processing(self):
        app, _ = build(StubDetector([[]]), frame_count=5)
        stats = app.run()
        assert stats.total_s == pytest.approx(stats.startup_s + stats.elapsed_s)
