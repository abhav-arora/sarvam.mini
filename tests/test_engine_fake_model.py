"""
test_engine_fake_model.py: Tests for FakeModelEngine (§4.6 / Phase 4).

Validates:
- Starts and accepts input frames
- Emits tone burst chunks with simulated latency
- Handles end_input and empty session cleanly
- Reset works properly between sessions
"""

from __future__ import annotations

import pytest

from app.engine.fake_model import FakeModelEngine
from app.protocol import FRAME_BYTES


class TestFakeModelEngine:
    @pytest.mark.asyncio
    async def test_basic_fake_model(self):
        engine = FakeModelEngine(step_time_ms=10)
        engine.start("test-fake-001")

        # Push 3 frames
        for _ in range(3):
            engine.push(b"\x00" * FRAME_BYTES, 0.0)

        engine.end_input(1920 * 3)

        output_chunks = []
        async for chunk in engine.output():
            output_chunks.append(chunk)

        assert len(output_chunks) == 3
        for chunk in output_chunks:
            assert len(chunk) == FRAME_BYTES

    @pytest.mark.asyncio
    async def test_empty_session(self):
        engine = FakeModelEngine(step_time_ms=10)
        engine.start("test-fake-empty")
        engine.end_input(0)

        output_chunks = []
        async for chunk in engine.output():
            output_chunks.append(chunk)

        assert len(output_chunks) == 0

    @pytest.mark.asyncio
    async def test_reset(self):
        engine = FakeModelEngine(step_time_ms=10)
        engine.start("test-fake-reset")
        engine.push(b"\x00" * FRAME_BYTES, 0.0)
        engine.reset()

        assert len(engine._input_queue) == 0
        assert engine._session_id == ""
