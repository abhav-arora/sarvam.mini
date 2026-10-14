"""
client.py (§5.2): Paced, harness-like WebSocket client.

Sends audio frames on an absolute schedule so sleep errors don't accumulate.
Records server frame arrival times for analyze.py.

Usage:
    python tools/client.py clips/c01.wav
    python tools/client.py clips/c01.wav --delay 2000 --url ws://localhost:8000/ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid

import soundfile as sf

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.protocol import (
    FRAME_BYTES,
    FRAME_MS,
    MSG_INPUT_END,
    MSG_SESSION_COMPLETE,
    MSG_SESSION_START,
    SAMPLE_RATE,
)


async def run_session(
    url: str,
    clip_path: str,
    output_path: str | None = None,
) -> dict:
    """
    Run a single harness-like session against the server.

    Returns a trace dict suitable for analyze.py.
    """
    import websockets

    # Load clip as raw PCM bytes
    data, sr = sf.read(clip_path, dtype="int16")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Clip must be {SAMPLE_RATE} Hz, got {sr}")
    if data.ndim > 1:
        data = data[:, 0]  # mono

    pcm_bytes = data.tobytes()
    total_samples = len(data)

    # Slice into frames, zero-pad last
    frames: list[bytes] = []
    for offset in range(0, len(pcm_bytes), FRAME_BYTES):
        chunk = pcm_bytes[offset : offset + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
        frames.append(chunk)

    valid_samples = total_samples
    session_id = str(uuid.uuid4())

    # Trace data
    trace = {
        "session_id": session_id,
        "clip": clip_path,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "n_frames": len(frames),
        "sent_frames": [],
        "received_chunks": [],
        "t_session_start": 0.0,
        "t_input_end": 0.0,
        "t_session_complete": 0.0,
    }

    received_chunks: list[tuple[float, int]] = []
    session_complete_event = asyncio.Event()

    async with websockets.connect(url, max_size=2**20) as ws:
        t_base = time.perf_counter()

        # ── Receive task ─────────────────────────────────────────────
        async def receiver():
            try:
                async for msg in ws:
                    t_arr = time.perf_counter() - t_base
                    if isinstance(msg, bytes):
                        received_chunks.append((t_arr, len(msg)))
                    elif isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == MSG_SESSION_COMPLETE:
                            trace["t_session_complete"] = t_arr
                            session_complete_event.set()
                            return
            except Exception:
                session_complete_event.set()

        recv_task = asyncio.create_task(receiver())

        # ── Send session_start ───────────────────────────────────────
        await ws.send(json.dumps({
            "type": MSG_SESSION_START,
            "session_id": session_id,
        }))
        trace["t_session_start"] = time.perf_counter() - t_base

        # ── Send frames on absolute schedule (§5.2 step 3) ───────────
        t0 = time.perf_counter()
        for k, frame in enumerate(frames):
            target = t0 + k * (FRAME_MS / 1000.0)
            now = time.perf_counter()
            wait = target - now
            if wait > 0:
                await asyncio.sleep(wait)
            await ws.send(frame)
            trace["sent_frames"].append({
                "k": k,
                "t": time.perf_counter() - t_base,
                "nbytes": len(frame),
            })

        # ── Send input_end ───────────────────────────────────────────
        await ws.send(json.dumps({
            "type": MSG_INPUT_END,
            "valid_samples": valid_samples,
        }))
        trace["t_input_end"] = time.perf_counter() - t_base

        # ── Continue sending silent frames until session_complete ─────
        k_silence = len(frames)
        while not session_complete_event.is_set():
            target = t0 + k_silence * (FRAME_MS / 1000.0)
            now = time.perf_counter()
            wait = target - now
            if wait > 0:
                await asyncio.sleep(wait)

            silent_frame = b"\x00" * FRAME_BYTES
            try:
                await ws.send(silent_frame)
            except Exception:
                break
            k_silence += 1

            if session_complete_event.is_set():
                break

        # Wait for receiver to finish
        try:
            await asyncio.wait_for(recv_task, timeout=5.0)
        except (TimeoutError, Exception):
            recv_task.cancel()

    # Store received chunks
    trace["received_chunks"] = [
        {"t_arrival": t, "nbytes": n} for t, n in received_chunks
    ]

    return trace


def main():
    parser = argparse.ArgumentParser(description="Harness-like S2ST client")
    parser.add_argument("clip", help="Path to 24 kHz mono s16 WAV clip")
    parser.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket URL")
    parser.add_argument("--output", "-o", default=None, help="Output JSON trace path")
    args = parser.parse_args()

    if args.output is None:
        clip_name = os.path.splitext(os.path.basename(args.clip))[0]
        os.makedirs("out", exist_ok=True)
        args.output = f"out/{clip_name}.json"

    trace = asyncio.run(run_session(args.url, args.clip, args.output))

    with open(args.output, "w") as f:
        json.dump(trace, f, indent=2)

    n_recv = len(trace["received_chunks"])
    total_recv = sum(c["nbytes"] for c in trace["received_chunks"])
    print(f"Session {trace['session_id']}")
    print(f"  Sent {trace['n_frames']} frames")
    print(f"  Received {n_recv} chunks ({total_recv} bytes)")
    print(f"  Trace saved to {args.output}")


if __name__ == "__main__":
    main()
