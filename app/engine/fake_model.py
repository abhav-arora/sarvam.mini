"""
Fake model engine (§4.6): slow fake "model" for threading tests.

Emits tone bursts after a configurable per-step compute time.
Used to validate the threading model before the real checkpoint exists.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator

import numpy as np

from app.protocol import FRAME_SAMPLES, SAMPLE_RATE

logger = logging.getLogger("sarvam.engine.fake_model")


class FakeModelEngine:
    """
    Fake model engine that simulates inference latency.

    For each input frame, it "processes" for `step_time_ms` and then
    produces a short tone burst as output.
    """

    def __init__(self, step_time_ms: int = 50) -> None:
        self._step_time_s: float = step_time_ms / 1000.0
        self._input_queue: deque[bytes] = deque()
        self._output_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._input_ended: bool = False
        self._session_id: str = ""
        self._wake: asyncio.Event = asyncio.Event()
        self._process_task: asyncio.Task | None = None

    def start(self, session_id: str) -> None:
        self._session_id = session_id
        self._input_ended = False
        self._input_queue.clear()
        self._output_queue = asyncio.Queue()
        self._wake.clear()
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(
            "FakeModelEngine started session=%s step_time=%.3fs",
            session_id,
            self._step_time_s,
        )

    def push(self, pcm: bytes, t_arrival: float) -> None:
        self._input_queue.append(pcm)
        self._wake.set()

    def end_input(self, valid_samples: int) -> None:
        self._input_ended = True
        self._wake.set()
        logger.info("FakeModelEngine input ended session=%s", self._session_id)

    async def _process_loop(self) -> None:
        """Simulate per-step inference with configurable latency."""
        try:
            while True:
                while not self._input_queue and not self._input_ended:
                    self._wake.clear()
                    await self._wake.wait()

                if not self._input_queue and self._input_ended:
                    await self._output_queue.put(None)  # sentinel
                    return

                _frame = self._input_queue.popleft()

                # Simulate compute time
                await asyncio.sleep(self._step_time_s)

                # Generate a short 440 Hz tone burst
                t = np.arange(FRAME_SAMPLES) / SAMPLE_RATE
                tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
                await self._output_queue.put(tone.tobytes())

        except asyncio.CancelledError:
            pass

    async def output(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._output_queue.get()
            if chunk is None:
                return
            yield chunk

    def reset(self) -> None:
        if self._process_task:
            self._process_task.cancel()
            self._process_task = None
        self._input_queue.clear()
        self._output_queue = asyncio.Queue()
        self._wake.clear()
        self._input_ended = False
        self._session_id = ""
        logger.info("FakeModelEngine reset")
