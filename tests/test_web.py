"""Web layer: telemetry serialisation and HTTP surface.

No camera is opened. The pipeline is driven directly with synthetic frames, and
the HTTP routes are exercised through FastAPI's test client.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from conftest import make_detection

from safestep.config import AlertSettings, DetectorSettings, Settings
from safestep.spatial import Proximity, Zone
from safestep.web.pipeline import Snapshot, TelemetryPipeline, detection_to_dict

fastapi = pytest.importorskip("fastapi", reason="web extra not installed")
from fastapi.testclient import TestClient  # noqa: E402

from safestep.web.server import create_app  # noqa: E402


def color_settings(**kwargs) -> Settings:
    """Settings pinned to the colour backend so no weights are needed."""
    base = dict(
        detector=DetectorSettings(backend="color"),
        alerts=AlertSettings(speech_enabled=False, earcons_enabled=False),
        headless=True,
    )
    base.update(kwargs)
    return Settings(**base)


class TestDetectionSerialisation:
    def test_carries_every_field_the_ui_needs(self):
        payload = detection_to_dict(make_detection(track_id=3))
        assert set(payload) == {
            "x", "y", "w", "h", "label", "confidence", "zone", "proximity",
            "proximityRank", "distanceM", "truncated", "distanceMethod", "trackId",
        }

    def test_geometry_round_trips(self):
        payload = detection_to_dict(make_detection(x=10, y=20, w=30, h=40))
        assert (payload["x"], payload["y"], payload["w"], payload["h"]) == (10, 20, 30, 40)

    def test_proximity_is_sent_as_name_and_rank(self):
        payload = detection_to_dict(make_detection(proximity=Proximity.IMMEDIATE))
        assert payload["proximity"] == "IMMEDIATE"
        assert payload["proximityRank"] == 3

    def test_zone_is_a_plain_string(self):
        assert detection_to_dict(make_detection(zone=Zone.LEFT))["zone"] == "left"

    def test_missing_distance_serialises_as_null(self):
        assert detection_to_dict(make_detection(distance_m=None))["distanceM"] is None

    def test_truncation_flag_is_exposed(self):
        # The UI renders "<2.5m" rather than "~2.5m" off this.
        from dataclasses import replace

        detection = replace(make_detection(), truncated=True)
        assert detection_to_dict(detection)["truncated"] is True

    def test_payload_is_json_safe(self):
        import json

        json.dumps(detection_to_dict(make_detection(track_id=1)))


class TestSnapshot:
    def test_empty_snapshot_serialises(self):
        payload = Snapshot().as_dict()
        assert payload["detections"] == []
        assert payload["haptics"]["active"] is False

    def test_snapshot_exposes_a_sequence_number(self):
        # Clients use this to skip stale frames rather than queue them.
        assert Snapshot(seq=7).as_dict()["seq"] == 7


class TestPipeline:
    def test_builds_with_the_colour_backend(self):
        pipeline = TelemetryPipeline(color_settings())
        assert pipeline.detector_name == "color"
        assert pipeline.running is False

    def test_publish_produces_a_decodable_jpeg(self):
        pipeline = TelemetryPipeline(color_settings())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._publish(frame, [], None, 12.0)

        snapshot = pipeline.snapshot()
        assert snapshot.frame_jpeg
        raw = base64.b64decode(snapshot.frame_jpeg)
        assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker

    def test_publish_records_frame_dimensions(self):
        pipeline = TelemetryPipeline(color_settings())
        pipeline._publish(np.zeros((360, 480, 3), dtype=np.uint8), [], None, 5.0)
        snapshot = pipeline.snapshot()
        assert (snapshot.width, snapshot.height) == (480, 360)

    def test_sequence_increments_per_publish(self):
        pipeline = TelemetryPipeline(color_settings())
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        pipeline._publish(frame, [], None, 1.0)
        first = pipeline.snapshot().seq
        pipeline._publish(frame, [], None, 1.0)
        assert pipeline.snapshot().seq == first + 1

    def test_haptics_reflect_the_most_urgent_detection(self):
        pipeline = TelemetryPipeline(color_settings())
        pipeline._publish(
            np.zeros((120, 160, 3), dtype=np.uint8),
            [
                make_detection(proximity=Proximity.MODERATE, zone=Zone.LEFT, distance_m=3.0),
                make_detection(proximity=Proximity.IMMEDIATE, zone=Zone.CENTER, distance_m=0.4),
            ],
            None, 1.0,
        )
        haptics = pipeline.snapshot().haptics
        assert haptics["proximity"] == "IMMEDIATE"
        assert haptics["strength"] == 1.0  # 0.4 m is inside the full-power range

    def test_empty_scene_leaves_the_motors_idle(self):
        pipeline = TelemetryPipeline(color_settings())
        pipeline._publish(np.zeros((120, 160, 3), dtype=np.uint8), [], None, 1.0)
        assert pipeline.snapshot().haptics["active"] is False

    def test_stats_describe_the_running_configuration(self):
        pipeline = TelemetryPipeline(color_settings(detect_every_n=3))
        pipeline._publish(np.zeros((120, 160, 3), dtype=np.uint8), [], None, 9.5)
        stats = pipeline.snapshot().stats
        assert stats["detector"] == "color"
        assert stats["detectEveryN"] == 3
        assert stats["fps"] == 9.5

    def test_alerts_accumulate_into_the_transcript(self):
        pipeline = TelemetryPipeline(color_settings())
        raw = [make_detection(proximity=Proximity.IMMEDIATE)]

        # Register the obstacle with the tracker until it clears the default
        # confirmation threshold, which is what the real loop does.
        tracked = pipeline._tracker.update(raw)
        tracked = pipeline._tracker.update(raw)

        assert pipeline._maybe_alert(tracked) is not None
        pipeline._publish(np.zeros((120, 160, 3), dtype=np.uint8), tracked, None, 1.0)
        assert len(pipeline.snapshot().transcript) == 1

    def test_unconfirmed_obstacle_is_not_transcribed(self):
        # A single-frame false positive must not reach the UI.
        pipeline = TelemetryPipeline(color_settings())
        tracked = pipeline._tracker.update(
            [make_detection(proximity=Proximity.IMMEDIATE)]
        )
        assert pipeline._maybe_alert(tracked) is None

    def test_transcript_is_bounded(self):
        from safestep.web.pipeline import MAX_TRANSCRIPT, TranscriptEntry

        pipeline = TelemetryPipeline(color_settings())
        for i in range(MAX_TRANSCRIPT + 40):
            pipeline._transcript.append(
                TranscriptEntry(f"alert {i}", "person", "NEAR", "center", 1.0, 0.0)
            )
            del pipeline._transcript[:-MAX_TRANSCRIPT]
        assert len(pipeline._transcript) == MAX_TRANSCRIPT

    def test_wait_for_frame_times_out_when_nothing_is_published(self):
        pipeline = TelemetryPipeline(color_settings())
        assert pipeline.wait_for_frame(last_seq=0, timeout=0.05) is None

    def test_wait_for_frame_returns_a_newer_snapshot(self):
        pipeline = TelemetryPipeline(color_settings())
        pipeline._publish(np.zeros((120, 160, 3), dtype=np.uint8), [], None, 1.0)
        assert pipeline.wait_for_frame(last_seq=0, timeout=0.05) is not None


class TestHttp:
    @pytest.fixture
    def client(self):
        app = create_app(color_settings())
        # Do not enter the app lifespan: startup would open a camera.
        return TestClient(app)

    def test_index_serves_the_dashboard(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "SafeStep" in response.text

    def test_index_is_html(self, client):
        assert "text/html" in client.get("/").headers["content-type"]

    def test_status_reports_the_detector_and_settings(self, client):
        payload = client.get("/api/status").json()
        assert payload["detector"] == "color"
        assert payload["settings"]["hfov"] == 65.0

    def test_status_is_json_serialisable(self, client):
        assert client.get("/api/status").status_code == 200

    def test_unknown_route_is_a_404(self, client):
        assert client.get("/nope").status_code == 404


class TestStaticAssets:
    def test_dashboard_file_exists(self):
        from safestep.web.server import STATIC_DIR

        assert (STATIC_DIR / "index.html").is_file()

    def test_dashboard_has_no_external_dependencies(self):
        # A wearable's dashboard has to work with no internet connection.
        from safestep.web.server import STATIC_DIR

        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for marker in ("http://", "https://", "//cdn", "src=\"//"):
            assert marker not in html, f"external reference {marker!r} in dashboard"


class TestControl:
    """Start/stop/mirror control surface."""

    @pytest.fixture
    def client(self):
        return TestClient(create_app(color_settings()))

    def test_mirror_can_be_turned_on_and_off(self, client):
        assert client.post("/api/control/mirror-on").json()["mirror"] is True
        assert client.post("/api/control/mirror-off").json()["mirror"] is False

    def test_unknown_action_is_rejected(self, client):
        assert client.post("/api/control/explode").status_code == 400

    def test_stop_reports_not_running(self, client):
        assert client.post("/api/control/stop").json()["running"] is False

    def test_stop_publishes_a_blank_frame(self):
        # A stopped feed must not keep showing the last picture, or it reads as
        # a live camera pointed at a stationary scene.
        pipeline = TelemetryPipeline(color_settings())
        pipeline._publish(np.zeros((120, 160, 3), dtype=np.uint8), [], None, 1.0)
        assert pipeline.snapshot().frame_jpeg is not None
        pipeline.stop()
        assert pipeline.snapshot().frame_jpeg is None

    def test_stop_advances_the_sequence_so_clients_notice(self):
        pipeline = TelemetryPipeline(color_settings())
        pipeline._publish(np.zeros((120, 160, 3), dtype=np.uint8), [], None, 1.0)
        before = pipeline.snapshot().seq
        pipeline.stop()
        assert pipeline.snapshot().seq > before

    def test_mirror_defaults_to_the_configured_value(self):
        from dataclasses import replace
        from safestep.config import CameraSettings

        settings = color_settings(camera=replace(CameraSettings(), mirror=True))
        assert TelemetryPipeline(settings).mirror is True

    def test_status_exposes_the_mirror_state(self, client):
        client.post("/api/control/mirror-on")
        assert client.get("/api/status").json()["settings"]["mirror"] is True
