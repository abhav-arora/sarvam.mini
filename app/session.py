"""
Per-connection session state machine, reader/writer tasks (§4.1–§4.3).

State machine: WAIT_START → STREAMING → INPUT_ENDED → DRAINING → COMPLETE
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import time
from typing import TYPE_CHECKING

from starlette.websockets import WebSocket, WebSocketDisconnect

from app.metrics import LoopLagCanary, SessionMetrics
from app.protocol import (
    BYTES_PER_SAMPLE,
    MAX_OUT_BYTES,
    MSG_SESSION_COMPLETE,
    InputEndMessage,
    SessionStartMessage,
    parse_text_message,
)

if TYPE_CHECKING:
    from app.engine.base import Engine

logger = logging.getLogger("sarvam.session")


class State(enum.Enum):
    """Session state machine states (§4.2)."""
    WAIT_START = "WAIT_START"
    STREAMING = "STREAMING"
    INPUT_ENDED = "INPUT_ENDED"
    DRAINING = "DRAINING"
    COMPLETE = "COMPLETE"


class Session:
    """
    Manages a single WebSocket session.

    Implements the per-connection flow from §4.1:
    - reader task: receives frames, timestamps, enqueues to engine
    - writer task: pulls from engine output, sends audio chunks
    - watchdog: forces completion at input_end + watchdog_s
    """

    def __init__(
        self,
        ws: WebSocket,
        engine: Engine,
        watchdog_s: float = 25.0,
    ) -> None:
        self.ws = ws
        self.engine = engine
        self.watchdog_s = watchdog_s

        self.state = State.WAIT_START
        self.metrics = SessionMetrics()
        self.canary = LoopLagCanary()

        # Single-shot guard for session_complete (§4.3 rule 6)
        self._complete_sent = False
        self._complete_lock = asyncio.Lock()

        # Watchdog task handle
        self._watchdog_task: asyncio.Task | None = None

    # ── Central emit function (§4.3 rule 5) ──────────────────────────────

    async def emit(self, pcm: bytes) -> None:
        """
        Send an audio chunk to the client.

        Enforces:
        - Chunk size ≤ MAX_OUT_BYTES (48,000 B = 1 s)
        - Even byte length (whole samples)
        - No sending after session_complete
        """
        if self._complete_sent:
            logger.warning("Attempted to send audio after session_complete")
            return

        # Split into MAX_OUT_BYTES chunks if needed
        for offset in range(0, len(pcm), MAX_OUT_BYTES):
            chunk = pcm[offset : offset + MAX_OUT_BYTES]

            # Ensure even byte length (whole samples)
            if len(chunk) % BYTES_PER_SAMPLE != 0:
                chunk = chunk[: len(chunk) - (len(chunk) % BYTES_PER_SAMPLE)]

            if not chunk:
                continue

            if self.metrics.t_first_output == 0.0:
                self.metrics.t_first_output = time.perf_counter()

            try:
                await self.ws.send_bytes(chunk)
                self.metrics.chunks_sent += 1
                self.metrics.bytes_sent += len(chunk)
            except Exception as e:
                logger.error("Failed to send audio chunk: %s", e)
                return

    # ── Send session_complete (§4.3 rule 6) ──────────────────────────────

    async def send_session_complete(self) -> None:
        """
        Send session_complete exactly once, then close the socket.

        Guarded by a lock so concurrent triggers (writer done + watchdog)
        cannot send it twice.
        """
        async with self._complete_lock:
            if self._complete_sent:
                return
            self._complete_sent = True

        self.state = State.COMPLETE
        self.metrics.t_session_complete = time.perf_counter()

        try:
            msg = json.dumps({"type": MSG_SESSION_COMPLETE})
            await self.ws.send_text(msg)
            logger.info("Sent session_complete session=%s", self.metrics.session_id)
        except Exception as e:
            logger.error("Failed to send session_complete: %s", e)

        # Close the socket right after sending (§4.3 rule 6)
        try:
            await self.ws.close()
        except Exception:
            pass

    # ── Reader task (§4.3 rules 1–4) ─────────────────────────────────────

    async def read_loop(self) -> None:
        """
        Single receive loop: branch on bytes vs text.

        - Minimal work: parse, timestamp, enqueue.  No slow await.
        - Keeps reading after input_end (§4.3 rule 3).
        """
        try:
            while True:
                message = await self.ws.receive()
                msg_type = message.get("type")

                if msg_type == "websocket.disconnect":
                    logger.info("Client disconnected session=%s", self.metrics.session_id)
                    return

                t_now = time.perf_counter()

                # ── Binary frame: audio data ─────────────────────────
                if msg_type == "websocket.receive" and "bytes" in message and message["bytes"]:
                    pcm = message["bytes"]

                    if self.state == State.WAIT_START:
                        logger.warning("Received audio before session_start, ignoring")
                        continue

                    if self.state in (State.STREAMING, State.INPUT_ENDED, State.DRAINING):
                        if self.metrics.t_first_frame == 0.0:
                            self.metrics.t_first_frame = t_now

                        self.metrics.frames_received += 1
                        self.metrics.bytes_received += len(pcm)

                        # Only push to engine if we're still streaming
                        if self.state == State.STREAMING:
                            self.engine.push(pcm, t_now)

                    continue

                # ── Text frame: JSON control message ─────────────────
                if msg_type == "websocket.receive" and "text" in message and message["text"]:
                    raw = message["text"]
                    parsed = parse_text_message(raw)

                    if parsed is None:
                        logger.warning(
                            "Malformed/unknown text message, ignoring: %s",
                            raw[:200],
                        )
                        continue

                    # session_start
                    if isinstance(parsed, SessionStartMessage):
                        if self.state != State.WAIT_START:
                            logger.warning(
                                "Unexpected session_start in state %s, ignoring",
                                self.state.value,
                            )
                            continue

                        self.metrics.session_id = parsed.session_id
                        self.metrics.t_session_start = t_now
                        self.engine.start(parsed.session_id)
                        self.state = State.STREAMING
                        self.canary.start()
                        logger.info("Session started: %s", parsed.session_id)
                        continue

                    # input_end
                    if isinstance(parsed, InputEndMessage):
                        if self.state != State.STREAMING:
                            logger.warning(
                                "Unexpected input_end in state %s, ignoring",
                                self.state.value,
                            )
                            continue

                        self.metrics.t_input_end = t_now
                        self.metrics.valid_samples = parsed.valid_samples
                        self.engine.end_input(parsed.valid_samples)
                        self.state = State.INPUT_ENDED
                        logger.info(
                            "Input ended: valid_samples=%d session=%s",
                            parsed.valid_samples,
                            self.metrics.session_id,
                        )

                        # Start watchdog (§4.3 rule 7)
                        self._watchdog_task = asyncio.create_task(
                            self._watchdog()
                        )
                        continue

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected session=%s", self.metrics.session_id)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Reader error: %s", e, exc_info=True)

    # ── Writer task ──────────────────────────────────────────────────────

    async def write_loop(self) -> None:
        """
        Pull from engine.output() and emit chunks.

        Finishes when the engine's output iterator is exhausted,
        then sends session_complete.
        """
        try:
            # Wait until streaming has started
            while self.state == State.WAIT_START:
                await asyncio.sleep(0.01)

            async for chunk in self.engine.output():
                if self._complete_sent:
                    break
                await self.emit(chunk)

            # Engine output exhausted — send session_complete
            if not self._complete_sent:
                await self.send_session_complete()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Writer error: %s", e, exc_info=True)
            # Try to send session_complete even on error
            if not self._complete_sent:
                await self.send_session_complete()

    # ── Watchdog (§4.3 rule 7) ───────────────────────────────────────────

    async def _watchdog(self) -> None:
        """
        If the engine hasn't finished by input_end + watchdog_s,
        force session_complete.  Truncated output scores better than
        a failed session.
        """
        try:
            await asyncio.sleep(self.watchdog_s)
            if not self._complete_sent:
                logger.warning(
                    "Watchdog fired at %.1fs for session=%s, forcing session_complete",
                    self.watchdog_s,
                    self.metrics.session_id,
                )
                await self.send_session_complete()
        except asyncio.CancelledError:
            pass

    # ── Cleanup (§4.3 rule 9) ────────────────────────────────────────────

    async def cleanup(self, tasks: set[asyncio.Task]) -> None:
        """
        Cancel remaining tasks, reset engine, log summary.
        Called from the finally block. Always runs.
        """
        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Cancel watchdog
        if self._watchdog_task:
            self._watchdog_task.cancel()

        # Stop canary and collect samples
        lag_samples = self.canary.stop()
        self.metrics.loop_lag_samples = lag_samples

        # Reset engine state (§4.3 rule 9)
        try:
            self.engine.reset()
        except Exception as e:
            logger.error("Engine reset error: %s", e)

        # Ensure session_complete was sent
        if not self._complete_sent and self.state != State.WAIT_START:
            try:
                await self.send_session_complete()
            except Exception:
                pass

        # Log summary
        self.metrics.log_summary()

        # Wait for tasks to finish cancelling
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        logger.info("Session cleanup complete session=%s", self.metrics.session_id)
