"""Configuration profiles, argument parsing and the detector factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from safestep.cli import build_parser, parse_source, settings_from_args
from safestep.config import CameraSettings, DetectorSettings, Settings
from safestep.detection import (
    BACKENDS,
    BackendUnavailable,
    ModelFilesMissing,
    build_detector,
)
from safestep.detection.color import ColorThresholdDetector
from safestep.detection.mobilenet_ssd import caffe_support_available


def parse(*argv):
    return settings_from_args(build_parser().parse_args(list(argv)))


class TestSourceParsing:
    def test_numeric_source_is_a_camera_index(self):
        assert parse_source("0") == 0
        assert parse_source("2") == 2

    def test_path_source_is_kept_as_a_string(self):
        assert parse_source("clips/walk.mp4") == "clips/walk.mp4"

    def test_camera_settings_know_which_they_are(self):
        assert CameraSettings(source=0).is_file is False
        assert CameraSettings(source="walk.mp4").is_file is True


class TestProfiles:
    def test_pi_profile_is_headless_and_skips_frames(self):
        settings = Settings().with_profile("pi")
        assert settings.headless is True
        assert settings.detect_every_n > 1

    def test_desktop_profile_shows_the_preview(self):
        settings = Settings().with_profile("desktop")
        assert settings.headless is False
        assert settings.detect_every_n == 1

    def test_pi_profile_lowers_the_capture_resolution(self):
        assert Settings().with_profile("pi").camera.width < Settings().camera.width

    def test_unknown_profile_is_rejected(self):
        with pytest.raises(ValueError):
            Settings().with_profile("toaster")

    def test_profiles_do_not_mutate_the_original(self):
        original = Settings()
        original.with_profile("pi")
        assert original.headless is False


class TestArgumentParsing:
    def test_defaults_are_sane(self):
        settings = parse()
        assert settings.camera.source == 0
        assert settings.detector.backend == "auto"
        assert settings.alerts.speech_enabled is True
        assert settings.alerts.earcons_enabled is True

    def test_video_file_source(self):
        assert parse("--source", "clip.mp4").camera.source == "clip.mp4"

    def test_disabling_audio_channels(self):
        settings = parse("--no-speech", "--no-earcons")
        assert settings.alerts.speech_enabled is False
        assert settings.alerts.earcons_enabled is False

    def test_headless_flag(self):
        assert parse("--headless").headless is True

    def test_detector_choice(self):
        assert parse("--detector", "color").detector.backend == "color"

    def test_class_filter(self):
        settings = parse("--classes", "person", "chair")
        assert settings.detector.classes_of_interest == ("person", "chair")

    def test_alert_tuning_flags(self):
        settings = parse("--cooldown", "9", "--min-gap", "3", "--min-hits", "4")
        assert settings.alerts.cooldown_s == 9
        assert settings.alerts.min_gap_s == 3
        assert settings.alerts.min_hits == 4

    def test_min_proximity_names_map_to_ranks(self):
        assert parse("--min-proximity", "far").alerts.min_proximity == 0
        assert parse("--min-proximity", "immediate").alerts.min_proximity == 3

    def test_fov_drives_distance_estimation(self):
        assert parse("--hfov", "78").horizontal_fov_deg == 78.0

    def test_max_frames(self):
        assert parse("--max-frames", "25").max_frames == 25

    def test_profile_applies_its_defaults(self):
        assert parse("--profile", "pi").detect_every_n > 1

    def test_explicit_flag_overrides_the_profile(self):
        # --profile pi sets frame skipping; an explicit value must win.
        assert parse("--profile", "pi", "--detect-every-n", "1").detect_every_n == 1

    def test_headless_survives_the_desktop_profile(self):
        assert parse("--profile", "desktop", "--headless").headless is True

    def test_show_preview_overrides_the_pi_profile(self):
        assert parse("--profile", "pi", "--show-preview").headless is False

    def test_explicit_resolution_overrides_the_profile(self):
        settings = parse("--profile", "pi", "--width", "1280", "--height", "720")
        assert (settings.camera.width, settings.camera.height) == (1280, 720)

    def test_profile_resolution_applies_when_not_overridden(self):
        assert parse("--profile", "pi").camera.width == 480

    def test_headless_and_show_preview_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--headless", "--show-preview"])

    def test_detect_every_n_must_be_positive(self):
        from safestep.cli import main

        with pytest.raises(SystemExit):
            main(["--detect-every-n", "0"])

    def test_invalid_detector_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--detector", "magic"])

    def test_help_renders(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--help"])
        out = capsys.readouterr().out
        assert "--source" in out
        # The safety caveat must be visible to anyone reading the help text.
        assert "not a certified mobility device" in out


class TestDetectorFactory:
    def test_color_backend_builds_without_weights(self):
        detector = build_detector(DetectorSettings(backend="color"))
        assert isinstance(detector, ColorThresholdDetector)

    def test_auto_falls_back_to_colour_when_no_weights_exist(self, tmp_path):
        detector = build_detector(DetectorSettings(backend="auto", model_dir=tmp_path))
        assert detector.name == "color"

    def test_explicit_neural_backend_fails_loudly_when_weights_are_missing(self, tmp_path):
        # Silently downgrading a user who asked for a real model would be wrong
        # for an assistive device.
        with pytest.raises(ModelFilesMissing) as excinfo:
            build_detector(DetectorSettings(backend="yolo", model_dir=tmp_path))
        assert "fetch_models.py" in str(excinfo.value)

    @pytest.mark.skipif(
        not caffe_support_available(), reason="OpenCV 5 removed the Caffe importer"
    )
    def test_missing_weights_error_names_the_file(self, tmp_path):
        with pytest.raises(ModelFilesMissing) as excinfo:
            build_detector(DetectorSettings(backend="ssd", model_dir=tmp_path))
        assert "MobileNetSSD_deploy.caffemodel" in str(excinfo.value)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError):
            build_detector(DetectorSettings(backend="nonsense"))

    def test_all_advertised_backends_either_build_or_fail_actionably(self, tmp_path):
        # Every backend must do one of: construct, say which file is missing,
        # or say why this environment cannot run it. Never an opaque crash.
        for backend in BACKENDS:
            settings = DetectorSettings(backend=backend, model_dir=tmp_path)
            try:
                build_detector(settings)
            except (ModelFilesMissing, BackendUnavailable) as exc:
                assert str(exc).strip()

    def test_auto_survives_a_backend_that_cannot_run_here(self, tmp_path):
        # On OpenCV 5 the SSD backend is unavailable rather than merely
        # unweighted; 'auto' must step past it instead of dying.
        assert build_detector(
            DetectorSettings(backend="auto", model_dir=tmp_path)
        ).name == "color"


class TestModelDir:
    def test_default_model_dir_is_inside_the_repo(self):
        assert Settings().detector.model_dir.name == "models"

    def test_model_dir_is_overridable(self, tmp_path: Path):
        assert DetectorSettings(model_dir=tmp_path).model_dir == tmp_path
