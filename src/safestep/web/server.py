"""FastAPI server exposing the SafeStep pipeline to the browser.

One WebSocket carries everything: the JPEG frame and the telemetry that
describes it travel in the same message, so bounding boxes can never drift out
of sync with the picture.

Audio is *not* streamed. The browser synthesises the earcons itself from the
haptic telemetry using the Web Audio API, which gives sample-accurate cues with
no buffering delay, a real analyser node to visualise, and no audio plumbing
between processes.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from .pipeline import TelemetryPipeline

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


async def stream_telemetry(websocket: WebSocket, pipeline: TelemetryPipeline) -> None:
    """Push snapshots to one client until it goes away."""
    await websocket.accept()
    last_seq = 0
    loop = asyncio.get_running_loop()

    try:
        while True:
            # The blocking wait runs on a worker thread so the event loop keeps
            # serving other clients. A slow consumer simply misses frames
            # rather than accumulating a backlog of stale ones.
            snapshot = await loop.run_in_executor(
                None, pipeline.wait_for_frame, last_seq, 1.0
            )

            if snapshot is None:
                await websocket.send_json(
                    {"type": "idle", "error": pipeline.error, "running": pipeline.running}
                )
                continue

            last_seq = snapshot.seq
            payload = snapshot.as_dict()
            payload["type"] = "frame"
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.debug("Telemetry client disconnected")
    except Exception:  # noqa: BLE001 - a dead socket must not take down the server
        logger.debug("Telemetry socket closed", exc_info=True)


def apply_control(pipeline: TelemetryPipeline, action: str) -> JSONResponse:
    """Start, stop or restart capture, and toggle the mirror.

    Stopping releases the camera outright rather than merely pausing the
    stream: a device that reports itself off must not still hold the lens.
    """
    if action == "start":
        pipeline.start()
    elif action == "stop":
        pipeline.stop()
    elif action == "restart":
        pipeline.restart()
    elif action in ("mirror-on", "mirror-off"):
        pipeline.set_mirror(action == "mirror-on")
    else:
        return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)

    return JSONResponse(
        {"running": pipeline.running, "mirror": pipeline.mirror, "error": pipeline.error}
    )


def create_app(settings: Settings) -> FastAPI:
    """Build the FastAPI application around a telemetry pipeline."""
    pipeline = TelemetryPipeline(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        pipeline.start()
        logger.info("Pipeline started with the '%s' detector", pipeline.detector_name)
        yield
        pipeline.stop()

    app = FastAPI(title="SafeStep", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse(
            {
                "running": pipeline.running,
                "detector": pipeline.detector_name,
                "error": pipeline.error,
                "settings": {
                    "source": str(settings.camera.source),
                    "mirror": pipeline.mirror,
                    "hfov": settings.horizontal_fov_deg,
                    "detectEveryN": settings.detect_every_n,
                    "centerFraction": settings.center_fraction,
                    "cooldownS": settings.alerts.cooldown_s,
                    "minGapS": settings.alerts.min_gap_s,
                },
            }
        )

    @app.post("/api/control/{action}")
    def control(action: str) -> JSONResponse:
        return apply_control(pipeline, action)

    @app.websocket("/ws")
    async def telemetry(websocket: WebSocket) -> None:
        await stream_telemetry(websocket, pipeline)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8000,
          log_level: Optional[str] = None) -> None:
    """Run the web UI. Blocks until interrupted."""
    import uvicorn

    app = create_app(settings)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=(log_level or settings.log_level).lower(),
        access_log=False,
    )
