"""
test_engine_echo.py (§8): Echo engine unit tests.
"""

from __future__ import annotations

import time

import pytest

from app.engine.echo import EchoEngine
from app.protocol import FRAME_BYTES


@pytest.fixture
def engine():
    return EchoEngine(delay_ms=200)  # 200ms for fast tests


class TestEchoEngine:
    @pytest.mark.asyncio
    async def test_basic_echo(self, engine):
        """Push a frame and verify it comes back after the delay."""
        engine.start("test-session")

        frame = b"\x01" * FRAME_BYTES
        t_push = time.perf_counter()
        engine.push(frame, t_push)
        engine.end_input(valid_samples=1920)

        output_frames = []
        async for chunk in engine.output():
            output_frames.append(chunk)

        assert len(output_frames) == 1
        assert output_frames[0] == frame

    @pytest.mark.asyncio
    async def test_delay_timing(self, engine):
        """Verify that frames are released approximately after the configured delay."""
        engine.start("test-timing")

        frame = b"\x02" * FRAME_BYTES
        t_push = time.perf_counter()
        engine.push(frame, t_push)
        engine.end_input(valid_samples=1920)

        async for chunk in engine.output():
            t_output = time.perf_counter()
            elapsed_from_push = t_output - t_push
            # Should be approximately 200ms (with some tolerance)
            assert elapsed_from_push >= 0.15, f"Output too early: {elapsed_from_push:.3f}s"
            assert elapsed_from_push < 0.5, f"Output too late: {elapsed_from_push:.3f}s"

    @pytest.mark.asyncio
    async def test_multiple_frames(self, engine):
        """Multiple frames should all be echoed back."""
        engine.start("test-multi")

        n_frames = 5
        for i in range(n_frames):
            frame = bytes([i % 256]) * FRAME_BYTES
            engine.push(frame, time.perf_counter())

        engine.end_input(valid_samples=1920 * n_frames)

        output_frames = []
        async for chunk in engine.output():
            output_frames.append(chunk)

        assert len(output_frames) == n_frames

    @pytest.mark.asyncio
    async def test_empty_session(self, engine):
        """End input with no frames — output should be empty."""
        engine.start("test-empty")
        engine.end_input(valid_samples=0)

        output_frames = []
        async for chunk in engine.output():
            output_frames.append(chunk)

        assert len(output_frames) == 0

    @pytest.mark.asyncio
    async def test_reset(self, engine):
        """After reset, engine should be clean for the next session."""
        engine.start("session-1")
        engine.push(b"\x00" * FRAME_BYTES, time.perf_counter())
        engine.reset()

        # Start new session
        engine.start("session-2")
        engine.end_input(valid_samples=0)

        output_frames = []
        async for chunk in engine.output():
            output_frames.append(chunk)

        # Should not see frame from session-1
        assert len(output_frames) == 0

    @pytest.mark.asyncio
    async def test_frame_ordering(self, engine):
        """Frames should come out in the same order they were pushed."""
        engine.start("test-order")

        frames_in = []
        for i in range(3):
            frame = bytes([i + 1]) * FRAME_BYTES
            frames_in.append(frame)
            engine.push(frame, time.perf_counter())

        engine.end_input(valid_samples=1920 * 3)

        frames_out = []
        async for chunk in engine.output():
            frames_out.append(chunk)

        assert frames_out == frames_in
