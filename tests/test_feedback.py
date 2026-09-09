"""Feedback channels: the non-blocking speech mailbox and earcon synthesis.

No audio device is touched. The speech worker thread is deliberately not
started, so the mailbox semantics can be asserted directly, and earcons are
driven through a fake playback backend.
"""

from __future__ import annotations

import subprocess
import sys
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from safestep.alerts import Announcement
from safestep.config import AlertSettings
from safestep.feedback import CompositeAnnouncer, build_announcer
from safestep.feedback.base import NullAnnouncer
from safestep.feedback.earcons import (
    SAMPLE_RATE,
    EarconPlayer,
    synthesize_pulse_train,
    to_wav_bytes,
)
from safestep.feedback.speech import Pyttsx3Speaker
from safestep.spatial import Proximity, Zone


def announcement(proximity=Proximity.NEAR, zone=Zone.CENTER, text="chair, close") -> Announcement:
    return Announcement(text=text, proximity=proximity, zone=zone, label="chair")


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.played = []
        self.closed = False

    def play(self, wav_bytes: bytes) -> None:
        self.played.append(wav_bytes)

    def close(self) -> None:
        self.closed = True


class TestImportIsSideEffectFree:
    def test_importing_safestep_does_not_construct_a_speech_engine(self):
        # The original code called pyttsx3.init() at module scope, so importing
        # it required a working speech driver and broke CI. Run in a subprocess
        # so an already-imported pyttsx3 cannot mask a regression.
        src = Path(__file__).resolve().parents[1] / "src"
        code = (
            f"import sys; sys.path.insert(0, {str(src)!r});"
            "import safestep, safestep.app, safestep.cli, safestep.feedback;"
            "print('pyttsx3' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"


class TestSpeechMailbox:
    def test_say_stores_a_pending_message(self):
        speaker = Pyttsx3Speaker(start=False)
        speaker.say("first")
        assert speaker._pending == "first"
        speaker.close()

    def test_newer_message_replaces_an_unspoken_one(self):
        # Queueing would deliver stale warnings; the newest is the only useful one.
        speaker = Pyttsx3Speaker(start=False)
        speaker.say("old obstacle")
        speaker.say("new obstacle")
        assert speaker._pending == "new obstacle"
        speaker.close()

    def test_superseded_messages_are_counted_as_dropped(self):
        speaker = Pyttsx3Speaker(start=False)
        speaker.say("a")
        speaker.say("b")
        speaker.say("c")
        assert speaker.dropped_count == 2
        speaker.close()

    def test_say_returns_without_blocking(self):
        import time

        speaker = Pyttsx3Speaker(start=False)
        started = time.monotonic()
        for _ in range(1000):
            speaker.say("obstacle ahead")
        assert time.monotonic() - started < 0.5
        speaker.close()

    def test_announce_delegates_to_say(self):
        speaker = Pyttsx3Speaker(start=False)
        speaker.announce(announcement(text="person, very close, ahead"))
        assert speaker._pending == "person, very close, ahead"
        speaker.close()

    def test_say_after_close_is_ignored(self):
        speaker = Pyttsx3Speaker(start=False)
        speaker.close()
        speaker.say("too late")
        assert speaker._pending is None


class TestToneSynthesis:
    def test_output_is_stereo(self):
        samples = synthesize_pulse_train(440.0, 1, 100.0, 0.0)
        assert samples.ndim == 2 and samples.shape[1] == 2

    def test_duration_matches_the_requested_shape(self):
        # Three 70 ms pulses with two 30 ms gaps = 270 ms.
        samples = synthesize_pulse_train(880.0, 3, 70.0, 30.0)
        expected = int(SAMPLE_RATE * 0.270)
        assert abs(len(samples) - expected) < SAMPLE_RATE * 0.01

    def test_samples_stay_within_range(self):
        samples = synthesize_pulse_train(880.0, 3, 70.0, 45.0, volume=1.0)
        assert np.all(np.abs(samples) <= 1.0)

    def test_left_pan_is_louder_in_the_left_channel(self):
        samples = synthesize_pulse_train(440.0, 1, 100.0, 0.0, pan=-1.0)
        assert np.abs(samples[:, 0]).max() > np.abs(samples[:, 1]).max()

    def test_right_pan_is_louder_in_the_right_channel(self):
        samples = synthesize_pulse_train(440.0, 1, 100.0, 0.0, pan=1.0)
        assert np.abs(samples[:, 1]).max() > np.abs(samples[:, 0]).max()

    def test_centre_pan_is_balanced(self):
        samples = synthesize_pulse_train(440.0, 1, 100.0, 0.0, pan=0.0)
        assert np.abs(samples[:, 0]).max() == pytest.approx(np.abs(samples[:, 1]).max(), rel=1e-5)

    def test_envelope_starts_and_ends_near_silence(self):
        # Without the fade, every pulse begins with an audible click.
        samples = synthesize_pulse_train(440.0, 1, 100.0, 0.0)
        assert abs(samples[0, 0]) < 0.01
        assert abs(samples[-1, 0]) < 0.01

    def test_zero_pulses_yields_empty_audio(self):
        assert len(synthesize_pulse_train(440.0, 0, 100.0, 0.0)) == 0


class TestWavEncoding:
    def test_produces_a_readable_wav(self):
        samples = synthesize_pulse_train(440.0, 1, 50.0, 0.0)
        with wave.open(BytesIO(to_wav_bytes(samples)), "rb") as handle:
            assert handle.getnchannels() == 2
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == SAMPLE_RATE

    def test_starts_with_the_riff_magic(self):
        assert to_wav_bytes(synthesize_pulse_train(440.0, 1, 50.0, 0.0))[:4] == b"RIFF"

    def test_out_of_range_samples_are_clipped_not_wrapped(self):
        loud = np.ones((100, 2), dtype=np.float32) * 5.0
        with wave.open(BytesIO(to_wav_bytes(loud)), "rb") as handle:
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        assert pcm.max() == 32767 and pcm.min() >= 0


class TestEarconPlayer:
    def test_caches_a_cue_per_proximity_and_zone(self):
        player = EarconPlayer(backend=FakeBackend())
        # Three audible bands times three zones; FAR is deliberately silent.
        for proximity in (Proximity.IMMEDIATE, Proximity.NEAR, Proximity.MODERATE):
            for zone in Zone:
                assert player.cue_for(proximity, zone) is not None

    def test_far_obstacles_have_no_cue(self):
        player = EarconPlayer(backend=FakeBackend())
        assert player.cue_for(Proximity.FAR, Zone.CENTER) is None

    def test_announcing_plays_a_cue(self):
        backend = FakeBackend()
        EarconPlayer(backend=backend).announce(announcement())
        assert len(backend.played) == 1

    def test_far_announcement_plays_nothing(self):
        backend = FakeBackend()
        EarconPlayer(backend=backend).announce(announcement(proximity=Proximity.FAR))
        assert backend.played == []

    def test_closer_cue_is_higher_pitched(self):
        player = EarconPlayer(backend=FakeBackend())
        immediate = player.cue_for(Proximity.IMMEDIATE, Zone.CENTER)
        moderate = player.cue_for(Proximity.MODERATE, Zone.CENTER)
        # Higher pitch and more pulses, so the buffers differ.
        assert immediate != moderate

    def test_left_and_right_cues_differ(self):
        player = EarconPlayer(backend=FakeBackend())
        left = player.cue_for(Proximity.NEAR, Zone.LEFT)
        right = player.cue_for(Proximity.NEAR, Zone.RIGHT)
        assert left != right

    def test_playback_failure_does_not_propagate(self):
        class ExplodingBackend(FakeBackend):
            def play(self, wav_bytes: bytes) -> None:
                raise RuntimeError("no audio device")

        # Audio trouble must never interrupt navigation.
        EarconPlayer(backend=ExplodingBackend()).announce(announcement())

    def test_close_releases_the_backend(self):
        backend = FakeBackend()
        player = EarconPlayer(backend=backend)
        player.close()
        assert backend.closed


class TestCompositeAnnouncer:
    def test_fans_out_to_every_channel(self, announcer):
        second = type(announcer)()
        CompositeAnnouncer([announcer, second]).announce(announcement())
        assert len(announcer.announcements) == 1
        assert len(second.announcements) == 1

    def test_a_failing_channel_does_not_silence_the_others(self, announcer):
        class Broken:
            def announce(self, a):
                raise RuntimeError("boom")

            def close(self):
                raise RuntimeError("boom")

        composite = CompositeAnnouncer([Broken(), announcer])
        composite.announce(announcement())
        composite.close()
        assert len(announcer.announcements) == 1

    def test_close_closes_every_channel(self, announcer):
        CompositeAnnouncer([announcer]).close()
        assert announcer.closed


class TestBuildAnnouncer:
    def test_all_channels_disabled_gives_a_null_announcer(self):
        settings = AlertSettings(speech_enabled=False, earcons_enabled=False)
        assert isinstance(build_announcer(settings), NullAnnouncer)

    def test_null_announcer_accepts_and_discards(self):
        null = NullAnnouncer()
        null.announce(announcement())
        null.close()
