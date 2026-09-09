"""Web UI for SafeStep.

Imported lazily by the CLI so that FastAPI and uvicorn remain optional: the
core application runs on a headless Raspberry Pi with neither installed.
"""

from __future__ import annotations

__all__ = ["create_app", "serve", "TelemetryPipeline"]


def __getattr__(name: str):
    if name in ("create_app", "serve"):
        from .server import create_app, serve

        return {"create_app": create_app, "serve": serve}[name]
    if name == "TelemetryPipeline":
        from .pipeline import TelemetryPipeline

        return TelemetryPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
