"""Non-blocking text-to-speech.

This module exists to fix the defect at the heart of the original SafeStep:
``engine.runAndWait()`` is synchronous, and calling it from the capture loop
stalled frame acquisition for the duration of every utterance.

Two design points matter:

**The engine is built on the worker thread that uses it.** ``pyttsx3`` drivers
(SAPI5 on Windows, NSSpeechSynthesizer on macOS, eSpeak on Linux) have thread
affinity and misbehave when driven from a thread other than their creator. The
import is also deferred to that point, so merely importing SafeStep does not
require a working speech driver -- which is what makes the package importable
in CI.

**The mailbox holds one message, and a new one replaces it.** Queueing would be
wrong: if speech is busy, the *newest* obstacle report is the only one worth
saying. An obstacle warning delivered three seconds late is worse than silence
because the user has already walked past it.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..alerts import Announcement

logger = logging.getLogger(__name__)


class Pyttsx3Speaker:
    """Speaks announcements on a background thread, dropping stale ones."""

    def __init__(self, rate: int = 165, volume: float = 1.0, start: bool = True) -> None:
        self.rate = rate
        self.volume = volume

        self._pending: Optional[str] = None
        self._condition = threading.Condition()
        self._stopping = False
        self._ready = threading.Event()
        self._failed: Optional[BaseException] = None
        self._spoken_count = 0
        self._dropped_count = 0

        self._thread = threading.Thread(target=self._run, name="safestep-speech", daemon=True)
        if start:
            self._thread.start()

    @property
    def spoken_count(self) -> int:
        return self._spoken_count

    @property
    def dropped_count(self) -> int:
        """Announcements superseded before they could be spoken."""
        return self._dropped_count

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until the engine is initialised. Returns False on timeout/failure."""
        ok = self._ready.wait(timeout)
        return ok and self._failed is None

    def announce(self, announcement: Announcement) -> None:
        self.say(announcement.text)

    def say(self, text: str) -> None:
        """Queue ``text``, replacing anything not yet spoken. Returns immediately."""
        with self._condition:
            if self._stopping:
                return
            if self._pending is not None:
                self._dropped_count += 1
            self._pending = text
            self._condition.notify()

    def close(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self) -> None:
        engine = None
        try:
            import pyttsx3  # Imported here: see module docstring.

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
        except Exception as exc:  # noqa: BLE001 - no speech driver must not kill the app
            self._failed = exc
            logger.warning(
                "Text-to-speech unavailable (%s). SafeStep will continue without speech; "
                "earcons, if enabled, are unaffected.",
                exc,
            )
            self._ready.set()
            return
        finally:
            self._ready.set()

        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    break
                text = self._pending
                self._pending = None

            try:
                engine.say(text)
                engine.runAndWait()
                self._spoken_count += 1
            except Exception:  # noqa: BLE001 - a failed utterance must not kill the thread
                logger.debug("Failed to speak %r", text, exc_info=True)

        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            logger.debug("Error stopping speech engine", exc_info=True)
