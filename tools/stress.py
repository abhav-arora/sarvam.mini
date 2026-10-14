"""
stress.py (§8): Fault injection and stress testing.

Scenarios:
- disconnect: Client disconnects mid-session
- concurrent: Two simultaneous connections
- burst: Frames sent faster than real time
- slow_engine: Artificially slow engine (requires fake_model engine)
- endurance: Many back-to-back sessions

Usage:
    python tools/stress.py --scenario disconnect --clip clips/c01.wav
    python tools/stress.py --scenario concurrent --clip clips/c01.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.protocol import (
    FRAME_BYTES,
    FRAME_MS,
    MSG_INPUT_END,
    MSG_SESSION_COMPLETE,
    MSG_SESSION_START,
)


async def scenario_disconnect(url: str, clip_path: str) -> dict:
    """Client disconnects mid-session, then a new session runs cleanly."""
    import soundfile as sf
    import websockets

    data, sr = sf.read(clip_path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    pcm = data.tobytes()

    frames = []
    for offset in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[offset : offset + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk += b"\x00" * (FRAME_BYTES - len(chunk))
        frames.append(chunk)

    mid = len(frames) // 2
    result = {"scenario": "disconnect", "phases": []}

    # Phase 1: connect, send half, disconnect abruptly
    print("Phase 1: Connecting and disconnecting mid-session...")
    try:
        async with websockets.connect(url, max_size=2**20) as ws:
            await ws.send(json.dumps({
                "type": MSG_SESSION_START,
                "session_id": str(uuid.uuid4()),
            }))

            for frame in frames[:mid]:
                await ws.send(frame)
                await asyncio.sleep(0.01)  # fast send

            # Abrupt disconnect (just exit the context manager)
        result["phases"].append({"phase": 1, "status": "disconnected_at_frame", "frame": mid})
    except Exception as e:
        result["phases"].append({"phase": 1, "status": "error", "error": str(e)})

    # Wait for server cleanup
    await asyncio.sleep(3.0)

    # Phase 2: new clean session
    print("Phase 2: Running a clean session...")
    from tools.client import run_session

    try:
        trace = await run_session(url, clip_path)
        n_chunks = len(trace.get("received_chunks", []))
        result["phases"].append({
            "phase": 2,
            "status": "completed",
            "chunks_received": n_chunks,
            "session_complete": trace.get("t_session_complete", 0) > 0,
        })
    except Exception as e:
        result["phases"].append({"phase": 2, "status": "error", "error": str(e)})

    return result


async def scenario_concurrent(url: str, clip_path: str) -> dict:
    """Two simultaneous connections. Verify serialisation or rejection."""
    from tools.client import run_session

    result = {"scenario": "concurrent", "sessions": []}

    async def run_one(idx: int):
        try:
            trace = await run_session(url, clip_path)
            return {
                "session": idx,
                "status": "completed",
                "session_complete": trace.get("t_session_complete", 0) > 0,
            }
        except Exception as e:
            return {"session": idx, "status": "error", "error": str(e)}

    results = await asyncio.gather(run_one(1), run_one(2))
    result["sessions"] = list(results)
    return result


async def scenario_burst(url: str, clip_path: str) -> dict:
    """Frames sent faster than real time."""
    import soundfile as sf
    import websockets

    data, sr = sf.read(clip_path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    pcm = data.tobytes()

    frames = []
    for offset in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[offset : offset + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk += b"\x00" * (FRAME_BYTES - len(chunk))
        frames.append(chunk)

    result = {"scenario": "burst", "n_frames": len(frames)}
    session_id = str(uuid.uuid4())

    async with websockets.connect(url, max_size=2**20) as ws:
        # Start session
        await ws.send(json.dumps({
            "type": MSG_SESSION_START,
            "session_id": session_id,
        }))

        # Send ALL frames as fast as possible (no pacing)
        t0 = time.perf_counter()
        for frame in frames:
            await ws.send(frame)
        burst_time = time.perf_counter() - t0

        result["burst_time_s"] = burst_time
        result["expected_realtime_s"] = len(frames) * FRAME_MS / 1000

        # Send input_end
        await ws.send(json.dumps({
            "type": MSG_INPUT_END,
            "valid_samples": len(data),
        }))

        # Wait for session_complete
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("type") == MSG_SESSION_COMPLETE:
                        result["session_complete"] = True
                        break
        except Exception:
            result["session_complete"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="Fault injection / stress testing")
    parser.add_argument(
        "--scenario",
        choices=["disconnect", "concurrent", "burst"],
        required=True,
    )
    parser.add_argument("--clip", required=True, help="Path to test clip")
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--output", "-o", default="out/stress_results.json")
    args = parser.parse_args()

    if args.scenario == "disconnect":
        result = asyncio.run(scenario_disconnect(args.url, args.clip))
    elif args.scenario == "concurrent":
        result = asyncio.run(scenario_concurrent(args.url, args.clip))
    elif args.scenario == "burst":
        result = asyncio.run(scenario_burst(args.url, args.clip))

    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
