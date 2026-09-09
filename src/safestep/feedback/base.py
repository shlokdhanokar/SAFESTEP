"""The feedback contract.

An announcer converts an :class:`~safestep.alerts.Announcement` into something
the user perceives. The single hard requirement: :meth:`Announcer.announce`
must return promptly and must never block the capture loop. Audio output is
slow (a spoken sentence is well over a second); every implementation that talks
to a device does so on its own thread.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..alerts import Announcement


@runtime_checkable
class Announcer(Protocol):
    """Anything that can convey an announcement to the user."""

    def announce(self, announcement: Announcement) -> None:
        """Convey ``announcement``. Must not block."""
        ...

    def close(self) -> None:
        """Release resources and stop any worker threads."""
        ...


class NullAnnouncer:
    """Discards everything. Used for silent runs, headless tests and CI."""

    def announce(self, announcement: Announcement) -> None:
        return None

    def close(self) -> None:
        return None
