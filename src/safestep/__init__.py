"""SafeStep - real-time obstacle detection and audio guidance.

Importing this package is deliberately side-effect free: no camera is opened,
no model is loaded and no text-to-speech engine is constructed. Every component
with hardware or driver requirements is built explicitly through a factory
(:func:`safestep.detection.build_detector`, :func:`safestep.feedback.build_announcer`)
so the package can be imported and unit-tested on a machine with no camera,
no display and no speech driver.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
