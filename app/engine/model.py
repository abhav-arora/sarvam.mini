"""
Real model adapter (§4.6): stub for the actual checkpoint.

This module will be filled in Phase 5 after the checkpoint is released.
For now it raises NotImplementedError to make accidental use obvious.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

logger = logging.getLogger("sarvam.engine.model")


class ModelEngine:
    """
    Adapter for the real S2ST model checkpoint.

    Design assumptions (from §4.6, to be confirmed against released code):
    - Inference in one dedicated thread owning the GPU and all model state.
    - Frame-synchronous stepping: one 80ms input → one step → output.
    - Emit per step, never batch.
    - Warm-up before /health returns 200.
    """

    def start(self, session_id: str) -> None:
        raise NotImplementedError(
            "ModelEngine is a stub. The real checkpoint is not yet available."
        )

    def push(self, pcm: bytes, t_arrival: float) -> None:
        raise NotImplementedError

    def end_input(self, valid_samples: int) -> None:
        raise NotImplementedError

    async def output(self) -> AsyncIterator[bytes]:
        raise NotImplementedError
        # Make this a generator to satisfy the type
        yield b""  # pragma: no cover

    def reset(self) -> None:
        pass
