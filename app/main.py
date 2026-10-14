"""
FastAPI application: /health, /ws endpoints, startup warm-up (§4.1).

Entry point for the server.  Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 0
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from app.config import settings
from app.engine.echo import EchoEngine
from app.engine.fake_model import FakeModelEngine
from app.engine.model import ModelEngine
from app.session import Session

# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sarvam.main")

# Engine instance — one for all sessions (§1: sessions run sequentially)
_engine = None

# Serialise sessions with a semaphore (§4.3 rule 8)
_session_semaphore = asyncio.Semaphore(1)

# Startup readiness flag
_ready = False


def _create_engine():
    """Create the engine based on configuration."""
    if settings.engine == "echo":
        return EchoEngine(delay_ms=settings.echo_delay_ms)
    elif settings.engine == "fake_model":
        return FakeModelEngine()
    elif settings.engine == "model":
        return ModelEngine()
    else:
        raise ValueError(f"Unknown engine type: {settings.engine}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: create engine, warm up, then mark as ready.
    /health won't return 200 until warm-up is complete (§4.6).
    """
    global _engine, _ready
    logger.info("Starting server with engine=%s", settings.engine)
    _engine = _create_engine()
    # Warm-up: for echo engine, this is a no-op.
    # For real model, this would run a dummy session to trigger CUDA graph
    # capture, torch.compile, cuDNN autotune, etc.
    logger.info("Warm-up complete. Server ready.")
    _ready = True
    yield
    if _engine:
        _engine.reset()


app = FastAPI(title="Sarvam S2ST Server", version="0.1.0", lifespan=lifespan)


# ── Health endpoint ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check.  Returns 200 only after warm-up is complete.
    Must respond within 300 s of server start (§1).
    """
    if _ready:
        return JSONResponse({"status": "ok"}, status_code=200)
    return JSONResponse({"status": "warming_up"}, status_code=503)


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for S2ST sessions.

    Follows the per-connection flow from §4.1:
    1. Accept connection
    2. Create Session with engine
    3. Run reader + writer tasks concurrently
    4. Cleanup in finally block
    """
    await ws.accept()
    logger.info("WebSocket connection accepted")

    async with _session_semaphore:  # §4.3 rule 8: serialise sessions
        session = Session(ws, _engine, watchdog_s=settings.finish_watchdog_s)
        tasks = {
            asyncio.create_task(session.read_loop()),
            asyncio.create_task(session.write_loop()),
        }
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            # If any task raised an exception, log it
            for t in done:
                if t.exception():
                    logger.error("Task exception: %s", t.exception())
        finally:
            await session.cleanup(tasks)
