"""
Echo engine (§4.5): delayed echo for demo and baseline.

Each pushed frame gets release_time = t_arrival + D.
A pump coroutine sleeps until the head frame's release time, then yields it.
Done condition: input ended and queue empty.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator

logger = logging.getLogger("sarvam.engine.echo")


class EchoEngine:
    """
    Delayed-echo engine.

    Replays every received frame after a fixed delay D.
    Schedule by arrival timestamps, not by counting frames —
    so the echo stays correct even if the reader jitters.
    """

    def __init__(self, delay_ms: int = 2000) -> None:
        self._delay_s: float = delay_ms / 1000.0
        self._queue: deque[tuple[float, bytes]] = deque()
        self._wake: asyncio.Event = asyncio.Event()
        self._input_ended: bool = False
        self._session_id: str = ""

    def start(self, session_id: str) -> None:
        """Initialise for a new session."""
        self._session_id = session_id
        self._input_ended = False
        self._queue.clear()
        self._wake.clear()
        logger.info(
            "EchoEngine started session=%s delay=%.3fs",
            session_id,
            self._delay_s,
        )

    def push(self, pcm: bytes, t_arrival: float) -> None:
        """
        Accept one input frame (non-blocking).

        The frame is scheduled for release at t_arrival + delay.
        """
        release_time = t_arrival + self._delay_s
        self._queue.append((release_time, pcm))
        self._wake.set()

    def end_input(self, valid_samples: int) -> None:
        """Signal that the real audio input has ended."""
        self._input_ended = True
        self._wake.set()
        logger.info(
            "EchoEngine input ended session=%s valid_samples=%d",
            self._session_id,
            valid_samples,
        )

    async def output(self) -> AsyncIterator[bytes]:
        """
        Yield output audio chunks.

        Sleeps until each frame's release time, then yields.
        Finishes when input has ended and the queue is empty.
        """
        while True:
            # Wait for data or input_end signal
            while not self._queue and not self._input_ended:
                self._wake.clear()
                await self._wake.wait()

            # If queue is empty and input ended → we're done
            if not self._queue and self._input_ended:
                logger.info("EchoEngine output complete session=%s", self._session_id)
                return

            # Pop the head frame and sleep until its release time
            release_time, pcm = self._queue.popleft()
            now = time.perf_counter()
            wait = release_time - now
            if wait > 0:
                await asyncio.sleep(wait)

            yield pcm

    def reset(self) -> None:
        """Reset all state for the next session."""
        self._queue.clear()
        self._wake.clear()
        self._input_ended = False
        self._session_id = ""
        logger.info("EchoEngine reset")
