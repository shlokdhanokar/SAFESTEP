"""Feedback channels and the factory that assembles them."""

from __future__ import annotations

import logging
from typing import List

from ..alerts import Announcement
from ..config import AlertSettings
from .base import Announcer, NullAnnouncer

logger = logging.getLogger(__name__)


class CompositeAnnouncer:
    """Fans an announcement out to several channels, earcon first.

    Order is significant. The tone is emitted before speech is queued so the
    user gets the urgency and direction cue immediately, with the spoken
    identity arriving behind it.
    """

    def __init__(self, channels: List[Announcer]) -> None:
        self._channels = list(channels)

    @property
    def channels(self) -> List[Announcer]:
        return list(self._channels)

    def announce(self, announcement: Announcement) -> None:
        for channel in self._channels:
            try:
                channel.announce(announcement)
            except Exception:  # noqa: BLE001 - one dead channel must not silence the rest
                logger.debug("Announcer %r failed", channel, exc_info=True)

    def close(self) -> None:
        for channel in self._channels:
            try:
                channel.close()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing announcer %r", channel, exc_info=True)


def build_announcer(settings: AlertSettings) -> Announcer:
    """Assemble the configured feedback channels.

    Both channels are optional and both fail soft: a machine with no speech
    driver or no audio device still runs, just more quietly. If everything is
    disabled or unavailable the result is a :class:`NullAnnouncer`.
    """
    channels: List[Announcer] = []

    if settings.earcons_enabled:
        try:
            from .earcons import EarconPlayer

            channels.append(EarconPlayer())
        except Exception:  # noqa: BLE001
            logger.warning("Could not initialise earcons; continuing without tones", exc_info=True)

    if settings.speech_enabled:
        try:
            from .speech import Pyttsx3Speaker

            channels.append(
                Pyttsx3Speaker(rate=settings.speech_rate, volume=settings.speech_volume)
            )
        except Exception:  # noqa: BLE001
            logger.warning("Could not initialise speech; continuing without it", exc_info=True)

    if not channels:
        logger.warning("All audio feedback is disabled - SafeStep will run silently")
        return NullAnnouncer()

    return CompositeAnnouncer(channels)


__all__ = ["Announcer", "NullAnnouncer", "CompositeAnnouncer", "build_announcer"]
