"""Earcons -- short tonal cues that reach the user before speech can.

Speech is the wrong *primary* channel for obstacle warnings. A spoken sentence
takes well over a second; at a walking pace of ~1.4 m/s the user has covered
two metres before they learn anything. Earcons fire in tens of milliseconds and
carry the two facts that matter most:

* **Urgency** through pitch and pulse rate -- closer is higher and faster.
* **Direction** through stereo panning -- an obstacle on the left is louder in
  the left ear.

Speech then follows with the detail ("chair", "person"). This mirrors how
production mobility aids are built.

All cues are synthesised once at construction and cached, so emitting one is a
buffer hand-off rather than a computation. Playback backends are tried in order
and degrade to a logged no-op, so earcons never become an install blocker:

1. ``winsound`` -- Windows stdlib, plays an in-memory WAV asynchronously.
2. ``simpleaudio`` -- optional cross-platform dependency (``pip install safestep[audio]``).
3. no-op with a single warning.
"""

from __future__ import annotations

import io
import logging
import math
import sys
import wave
from typing import Dict, Optional, Tuple

import numpy as np

from ..alerts import Announcement
from ..spatial import Proximity, Zone

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
_FADE_MS = 6.0  # Attack/release ramp; without it each pulse starts with a click.

# (frequency Hz, pulse count, pulse duration ms, gap ms)
_CUE_SHAPE: Dict[Proximity, Tuple[float, int, float, float]] = {
    Proximity.IMMEDIATE: (880.0, 3, 70.0, 45.0),
    Proximity.NEAR: (660.0, 2, 90.0, 60.0),
    Proximity.MODERATE: (440.0, 1, 130.0, 0.0),
}

# Constant-power pan positions, -1.0 hard left to +1.0 hard right. Kept short of
# the extremes so a cue is never fully inaudible in one ear.
_PAN: Dict[Zone, float] = {
    Zone.LEFT: -0.7,
    Zone.CENTER: 0.0,
    Zone.RIGHT: 0.7,
}


def synthesize_pulse_train(
    frequency: float,
    pulses: int,
    pulse_ms: float,
    gap_ms: float,
    pan: float = 0.0,
    volume: float = 0.6,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Build a stereo pulse train as float32 samples in ``[-1, 1]``.

    Returns an ``(n, 2)`` array. ``pan`` uses the constant-power law, so a cue
    keeps roughly the same perceived loudness wherever it sits in the field.
    """
    if pulses <= 0 or pulse_ms <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    pulse_samples = max(1, int(sample_rate * pulse_ms / 1000.0))
    gap_samples = max(0, int(sample_rate * gap_ms / 1000.0))

    t = np.arange(pulse_samples, dtype=np.float32) / sample_rate
    tone = np.sin(2.0 * np.pi * frequency * t, dtype=np.float32)

    fade = min(int(sample_rate * _FADE_MS / 1000.0), pulse_samples // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]

    gap = np.zeros(gap_samples, dtype=np.float32)
    segments = []
    for index in range(pulses):
        segments.append(tone)
        if index < pulses - 1 and gap_samples:
            segments.append(gap)
    mono = np.concatenate(segments) * volume

    # Constant-power panning: angle sweeps 0..pi/2 across the stereo field.
    angle = (np.clip(pan, -1.0, 1.0) + 1.0) * (math.pi / 4.0)
    left_gain = math.cos(angle)
    right_gain = math.sin(angle)

    return np.stack([mono * left_gain, mono * right_gain], axis=1).astype(np.float32)


def to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode float32 ``(n, channels)`` samples as a 16-bit PCM WAV in memory."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(pcm.shape[1] if pcm.ndim > 1 else 1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class _WinsoundBackend:
    name = "winsound"

    def __init__(self) -> None:
        import winsound

        self._winsound = winsound
        self._flags = winsound.SND_MEMORY | winsound.SND_ASYNC

    def play(self, wav_bytes: bytes) -> None:
        # SND_ASYNC returns immediately; a new cue pre-empts one still playing,
        # which is the behaviour we want for a freshening obstacle report.
        self._winsound.PlaySound(wav_bytes, self._flags)

    def close(self) -> None:
        try:
            self._winsound.PlaySound(None, self._winsound.SND_PURGE)
        except Exception:  # noqa: BLE001 - best-effort silence on shutdown
            pass


class _SimpleaudioBackend:
    name = "simpleaudio"

    def __init__(self) -> None:
        import simpleaudio

        self._simpleaudio = simpleaudio
        self._last = None

    def play(self, wav_bytes: bytes) -> None:
        if self._last is not None and self._last.is_playing():
            self._last.stop()
        wave_obj = self._simpleaudio.WaveObject.from_wave_file(io.BytesIO(wav_bytes))
        self._last = wave_obj.play()

    def close(self) -> None:
        try:
            if self._last is not None:
                self._last.stop()
        except Exception:  # noqa: BLE001
            pass


class _NullBackend:
    name = "none"

    def play(self, wav_bytes: bytes) -> None:
        return None

    def close(self) -> None:
        return None


def _select_backend():
    """Pick the first usable playback backend."""
    if sys.platform == "win32":
        try:
            return _WinsoundBackend()
        except Exception:  # noqa: BLE001
            logger.debug("winsound backend unavailable", exc_info=True)

    try:
        return _SimpleaudioBackend()
    except Exception:  # noqa: BLE001
        logger.debug("simpleaudio backend unavailable", exc_info=True)

    logger.warning(
        "No audio backend available for earcons - proximity tones are disabled. "
        "Install the optional dependency with: pip install simpleaudio"
    )
    return _NullBackend()


class EarconPlayer:
    """Plays a cached proximity/direction tone for each announcement."""

    def __init__(self, volume: float = 0.6, backend=None) -> None:
        self._backend = backend if backend is not None else _select_backend()
        self._cache: Dict[Tuple[Proximity, Zone], bytes] = {}

        for proximity, (freq, pulses, pulse_ms, gap_ms) in _CUE_SHAPE.items():
            for zone, pan in _PAN.items():
                samples = synthesize_pulse_train(
                    frequency=freq,
                    pulses=pulses,
                    pulse_ms=pulse_ms,
                    gap_ms=gap_ms,
                    pan=pan,
                    volume=volume,
                )
                self._cache[(proximity, zone)] = to_wav_bytes(samples)

        logger.info("Earcons using '%s' backend (%d cues cached)", self.backend_name, len(self._cache))

    @property
    def backend_name(self) -> str:
        return getattr(self._backend, "name", "unknown")

    @property
    def enabled(self) -> bool:
        return not isinstance(self._backend, _NullBackend)

    def cue_for(self, proximity: Proximity, zone: Zone) -> Optional[bytes]:
        return self._cache.get((proximity, zone))

    def announce(self, announcement: Announcement) -> None:
        wav = self.cue_for(announcement.proximity, announcement.zone)
        if wav is None:
            return
        try:
            self._backend.play(wav)
        except Exception:  # noqa: BLE001 - audio failure must never stop navigation
            logger.debug("Earcon playback failed", exc_info=True)

    def close(self) -> None:
        self._backend.close()
