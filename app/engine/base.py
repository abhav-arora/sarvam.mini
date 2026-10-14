"""
Engine protocol (§4.4).

This is the only interface that changes when the real model arrives.
All engines must implement this protocol.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    """
    Abstract engine interface.

    The server interacts with the model exclusively through this protocol.
    Implementations:
      - EchoEngine: delayed echo for demo / baseline (Phase 0)
      - FakeModelEngine: configurable-latency stub for threading tests (Phase 4)
      - ModelEngine: real checkpoint adapter (Phase 5)
    """

    def start(self, session_id: str) -> None:
        """Initialise for a new session."""
        ...

    def push(self, pcm: bytes, t_arrival: float) -> None:
        """
        Accept one input frame.  Must be non-blocking.
        Called from the reader task on the event loop.
        """
        ...

    def end_input(self, valid_samples: int) -> None:
        """Signal that the real audio input has ended."""
        ...

    def output(self) -> AsyncIterator[bytes]:
        """
        Yield output audio chunks (≤ MAX_OUT_BYTES, even byte length).
        The iterator finishes when the engine has no more output.
        """
        ...

    def reset(self) -> None:
        """
        Reset all state for the next session.
        Called in the finally block after every session.
        """
        ...
