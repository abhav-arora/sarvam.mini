"""
Metrics, timestamps, event-loop-lag canary, and JSONL session log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("sarvam.metrics")


@dataclass
class SessionMetrics:
    """Collects per-session timing data."""

    session_id: str = ""
    t_session_start: float = 0.0
    t_first_frame: float = 0.0
    t_input_end: float = 0.0
    t_first_output: float = 0.0
    t_session_complete: float = 0.0
    frames_received: int = 0
    bytes_received: int = 0
    chunks_sent: int = 0
    bytes_sent: int = 0
    valid_samples: int = 0
    loop_lag_samples: list[float] = field(default_factory=list)

    def log_summary(self) -> None:
        """Write a JSON summary line for this session."""
        summary = {
            "session_id": self.session_id,
            "duration_input_s": self.t_input_end - self.t_session_start
            if self.t_input_end
            else None,
            "duration_total_s": self.t_session_complete - self.t_session_start
            if self.t_session_complete
            else None,
            "time_to_first_output_s": self.t_first_output - self.t_session_start
            if self.t_first_output
            else None,
            "frames_received": self.frames_received,
            "bytes_received": self.bytes_received,
            "chunks_sent": self.chunks_sent,
            "bytes_sent": self.bytes_sent,
            "valid_samples": self.valid_samples,
            "loop_lag_p50_ms": _percentile(self.loop_lag_samples, 0.5) * 1000
            if self.loop_lag_samples
            else None,
            "loop_lag_p99_ms": _percentile(self.loop_lag_samples, 0.99) * 1000
            if self.loop_lag_samples
            else None,
        }
        logger.info("SESSION_SUMMARY %s", json.dumps(summary))


def _percentile(data: list[float], q: float) -> float:
    """Simple percentile without importing numpy."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(q * (len(s) - 1))
    return s[idx]


class LoopLagCanary:
    """
    Event-loop-lag canary (§7).

    A background task that sleeps for a short interval and measures how much
    the actual sleep overshot.  High overshoot means the event loop is blocked.
    """

    INTERVAL_S = 0.01  # 10 ms target

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self.samples: list[float] = []

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                t0 = time.perf_counter()
                await asyncio.sleep(self.INTERVAL_S)
                overshoot = time.perf_counter() - t0 - self.INTERVAL_S
                self.samples.append(overshoot)
                if overshoot > 0.05:  # 50 ms is concerning
                    logger.warning("Event-loop lag: %.1f ms", overshoot * 1000)
        except asyncio.CancelledError:
            pass

    def stop(self) -> list[float]:
        if self._task:
            self._task.cancel()
        result = self.samples.copy()
        self.samples.clear()
        return result
