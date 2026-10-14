"""
test_session.py (§8): Integration tests for the WebSocket session handler.

Tests the full server by spawning a uvicorn server and connecting with
the websockets client library.  Uses a short echo delay for fast tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time

import pytest
import uvicorn
import websockets

# Set echo delay low BEFORE importing the app
os.environ["ECHO_DELAY_MS"] = "100"
os.environ["ENGINE"] = "echo"

from app.main import app
from app.protocol import (
    BYTES_PER_SAMPLE,
    FRAME_BYTES,
    MAX_OUT_BYTES,
    MSG_INPUT_END,
    MSG_SESSION_COMPLETE,
    MSG_SESSION_START,
)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    """Start a uvicorn server in a background thread for integration tests."""
    port = _get_free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        ws_ping_interval=None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(50):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.1)
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)

    yield f"ws://127.0.0.1:{port}/ws"

    server.should_exit = True
    thread.join(timeout=5)


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self, server_url: str):
        import httpx
        http_url = server_url.replace("ws://", "http://").replace("/ws", "/health")
        resp = httpx.get(http_url)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestWebSocketSession:
    """Integration tests for the /ws endpoint."""

    @pytest.mark.asyncio
    async def test_valid_session(self, server_url: str):
        """A complete valid session: start → frames → input_end → session_complete."""
        async with websockets.connect(server_url, max_size=2**20) as ws:
            # Send session_start
            await ws.send(json.dumps({
                "type": MSG_SESSION_START,
                "session_id": "test-valid-001",
            }))

            # Send a few frames
            n_frames = 3
            for _ in range(n_frames):
                await ws.send(b"\x00" * FRAME_BYTES)

            # Send input_end
            await ws.send(json.dumps({
                "type": MSG_INPUT_END,
                "valid_samples": 1920 * n_frames,
            }))

            # Collect server responses
            received_chunks = []
            session_complete_received = False

            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        received_chunks.append(msg)
                        assert len(msg) <= MAX_OUT_BYTES
                        assert len(msg) % BYTES_PER_SAMPLE == 0
                    elif isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == MSG_SESSION_COMPLETE:
                            session_complete_received = True
                            break
            except websockets.exceptions.ConnectionClosed:
                pass

            assert session_complete_received, "session_complete not received"
            assert len(received_chunks) > 0, "No audio chunks received"

    @pytest.mark.asyncio
    async def test_malformed_json(self, server_url: str):
        """Server must not crash on malformed JSON."""
        # Wait for previous session cleanup
        await asyncio.sleep(0.5)

        async with websockets.connect(server_url, max_size=2**20) as ws:
            # Send garbage text
            await ws.send("{not valid json!!!")

            # Now send a valid session
            await ws.send(json.dumps({
                "type": MSG_SESSION_START,
                "session_id": "test-malformed-001",
            }))
            await ws.send(json.dumps({
                "type": MSG_INPUT_END,
                "valid_samples": 0,
            }))

            completed = False
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == MSG_SESSION_COMPLETE:
                            completed = True
                            break
            except websockets.exceptions.ConnectionClosed:
                pass

            assert completed, "Server should still complete after malformed JSON"

    @pytest.mark.asyncio
    async def test_zero_valid_samples(self, server_url: str):
        """Session with valid_samples=0 should complete cleanly."""
        await asyncio.sleep(0.5)

        async with websockets.connect(server_url, max_size=2**20) as ws:
            await ws.send(json.dumps({
                "type": MSG_SESSION_START,
                "session_id": "test-zero-samples",
            }))
            await ws.send(json.dumps({
                "type": MSG_INPUT_END,
                "valid_samples": 0,
            }))

            completed = False
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == MSG_SESSION_COMPLETE:
                            completed = True
                            break
            except websockets.exceptions.ConnectionClosed:
                pass

            assert completed
