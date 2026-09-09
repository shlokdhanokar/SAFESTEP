"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .config import (
    DEFAULT_MODEL_DIR,
    PROFILES,
    AlertSettings,
    DetectorSettings,
    Settings,
)
from .detection import BACKENDS

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        stream=sys.stderr,
    )


def parse_source(value: str):
    """Interpret ``--source`` as a camera index when numeric, else a file path."""
    try:
        return int(value)
    except ValueError:
        return value


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Shows defaults, except for flags whose default is "inherit from profile"."""

    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safestep",
        description=(
            "Real-time obstacle detection with audio guidance for visually impaired users."
        ),
        epilog=(
            "SAFETY: SafeStep is an experimental aid, not a certified mobility device. "
            "It does not replace a white cane, a guide dog, or orientation and mobility "
            "training. It can and will miss obstacles."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"SafeStep {__version__}")

    source = parser.add_argument_group("input")
    source.add_argument(
        "--source", type=parse_source, default=0,
        help="Camera index (e.g. 0) or path to a video file",
    )
    source.add_argument(
        "--width", type=int, default=None,
        help="Requested capture width (default: 640, or 480 under --profile pi)",
    )
    source.add_argument(
        "--height", type=int, default=None,
        help="Requested capture height (default: 480, or 360 under --profile pi)",
    )
    source.add_argument(
        "--fps", type=int, default=None,
        help="Requested capture frame rate (default: 30, or 15 under --profile pi)",
    )
    source.add_argument(
        "--hfov", type=float, default=65.0,
        help="Camera horizontal field of view in degrees; drives distance estimates",
    )
    source.add_argument(
        "--mirror", action="store_true",
        help=(
            "Flip the image horizontally. Use with a front-facing (laptop or phone) "
            "camera, where left and right are otherwise reported reversed"
        ),
    )

    detect = parser.add_argument_group("detection")
    detect.add_argument(
        "--detector", choices=BACKENDS, default="auto",
        help="Detection backend. 'auto' prefers yolo, then ssd, then the colour fallback",
    )
    detect.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory holding model weights",
    )
    detect.add_argument("--confidence", type=float, default=0.45, help="Minimum detection score")
    detect.add_argument("--nms", type=float, default=0.45, help="Non-maximum-suppression IoU threshold")
    detect.add_argument(
        "--classes", nargs="*", default=[], metavar="CLASS",
        help="Only announce these classes (e.g. --classes person chair). Default: all",
    )
    detect.add_argument(
        "--detect-every-n", type=int, default=None,
        help=(
            "Run detection every Nth frame; raise on slow hardware "
            "(default: 1, or 3 under --profile pi)"
        ),
    )

    alerts = parser.add_argument_group("alerts")
    alerts.add_argument("--no-speech", action="store_true", help="Disable spoken announcements")
    alerts.add_argument("--no-earcons", action="store_true", help="Disable proximity tones")
    alerts.add_argument("--cooldown", type=float, default=4.0, help="Seconds before re-announcing an object")
    alerts.add_argument("--min-gap", type=float, default=1.2, help="Minimum seconds between any two alerts")
    alerts.add_argument(
        "--min-hits", type=int, default=2,
        help="Detection passes an object must appear in before it is announced",
    )
    alerts.add_argument("--speech-rate", type=int, default=165, help="Speech rate in words per minute")
    alerts.add_argument(
        "--min-proximity", choices=("far", "moderate", "near", "immediate"), default="moderate",
        help="Quietest distance band worth announcing",
    )

    web = parser.add_argument_group("web ui")
    web.add_argument(
        "--web", action="store_true",
        help="Serve the browser dashboard instead of the desktop preview window",
    )
    web.add_argument("--host", default="127.0.0.1", help="Web UI bind address")
    web.add_argument("--port", type=int, default=8000, help="Web UI port")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--profile", choices=PROFILES, default=None,
        help="Apply hardware defaults ('pi' is headless with frame skipping)",
    )
    preview = runtime.add_mutually_exclusive_group()
    preview.add_argument(
        "--headless", action="store_true", help="Run without the preview window",
    )
    preview.add_argument(
        "--show-preview", action="store_true",
        help="Force the preview window on, overriding --profile pi",
    )
    runtime.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames")
    runtime.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging verbosity",
    )

    return parser


_PROXIMITY_BY_NAME = {"far": 0, "moderate": 1, "near": 2, "immediate": 3}


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Translate parsed arguments into a :class:`Settings`.

    Resolution order is: base defaults, then ``--profile``, then any flag the
    user actually typed. Profile-overridable flags default to ``None`` rather
    than to their nominal value, which is what lets ``--profile pi
    --detect-every-n 1`` mean what it says -- a literal ``1`` is otherwise
    indistinguishable from the default.
    """
    base = Settings()
    if args.profile:
        base = base.with_profile(args.profile)

    def pick(explicit, inherited):
        return inherited if explicit is None else explicit

    headless = base.headless
    if args.headless:
        headless = True
    elif args.show_preview:
        headless = False

    return replace(
        base,
        camera=replace(
            base.camera,
            source=args.source,
            width=pick(args.width, base.camera.width),
            height=pick(args.height, base.camera.height),
            fps=pick(args.fps, base.camera.fps),
            mirror=args.mirror,
        ),
        detector=DetectorSettings(
            backend=args.detector,
            model_dir=args.model_dir,
            confidence=args.confidence,
            nms_threshold=args.nms,
            classes_of_interest=tuple(args.classes),
        ),
        alerts=AlertSettings(
            cooldown_s=args.cooldown,
            min_gap_s=args.min_gap,
            min_hits=args.min_hits,
            speech_enabled=not args.no_speech,
            earcons_enabled=not args.no_earcons,
            speech_rate=args.speech_rate,
            min_proximity=_PROXIMITY_BY_NAME[args.min_proximity],
        ),
        headless=headless,
        detect_every_n=pick(args.detect_every_n, base.detect_every_n),
        horizontal_fov_deg=args.hfov,
        log_level=args.log_level,
        max_frames=args.max_frames,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(args.log_level)
    logger = logging.getLogger("safestep")

    if args.detect_every_n is not None and args.detect_every_n < 1:
        parser.error("--detect-every-n must be at least 1")

    settings = settings_from_args(args)

    logger.info("SafeStep %s starting", __version__)
    logger.warning(
        "SafeStep is an experimental aid, not a certified mobility device. "
        "Do not rely on it in place of a cane, guide dog, or your own judgement."
    )

    if args.web:
        return _serve_web(settings, args, logger)

    from .app import run

    return run(settings)


def _serve_web(settings: Settings, args: argparse.Namespace, logger: logging.Logger) -> int:
    """Launch the browser dashboard, or explain why it cannot start."""
    try:
        from .web.server import serve
    except ImportError:
        logger.error(
            "The web UI needs FastAPI and uvicorn, which are not installed.\n"
            "  pip install 'safestep[web]'\n"
            "or\n"
            "  pip install fastapi 'uvicorn[standard]'"
        )
        return 2

    # The dashboard replaces the desktop preview; leaving both on would fight
    # over the camera and open a stray OpenCV window next to the browser.
    settings = replace(settings, headless=True)

    logger.info("Dashboard: http://%s:%d", args.host, args.port)
    try:
        serve(settings, host=args.host, port=args.port, log_level=args.log_level)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")
    return 0


def _entrypoint() -> None:  # pragma: no cover - thin process wrapper
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _entrypoint()


__all__: List[str] = ["main", "build_parser", "settings_from_args", "configure_logging"]
