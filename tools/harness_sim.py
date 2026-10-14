"""
harness_sim.py (§5.4): Run N sequential sessions with 2 s gaps.

Asserts all protocol rules each time.  Catches state leaking between sessions.

Usage:
    python tools/harness_sim.py --sessions 5 --clip clips/c01.wav
    python tools/harness_sim.py --sessions 10 --clip clips/c01.wav --url ws://localhost:8000/ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.protocol import BYTES_PER_SAMPLE, COMPLETION_DEADLINE_S, MAX_OUT_BYTES
from tools.client import run_session


def check_rules(trace: dict) -> list[str]:
    """Check all protocol rules for a session trace. Returns list of failures."""
    failures = []

    # Rule: session_complete received
    if trace.get("t_session_complete", 0) <= 0:
        failures.append("No session_complete received")

    # Rule: all chunks ≤ MAX_OUT_BYTES
    for i, chunk in enumerate(trace.get("received_chunks", [])):
        if chunk["nbytes"] > MAX_OUT_BYTES:
            failures.append(f"Chunk {i} exceeds MAX_OUT_BYTES: {chunk['nbytes']}")

    # Rule: all chunks have even byte length
    for i, chunk in enumerate(trace.get("received_chunks", [])):
        if chunk["nbytes"] % BYTES_PER_SAMPLE != 0:
            failures.append(f"Chunk {i} has odd byte length: {chunk['nbytes']}")

    # Rule: completion within 30 s of input_end
    t_complete = trace.get("t_session_complete", 0)
    t_input_end = trace.get("t_input_end", 0)
    if t_complete and t_input_end:
        completion_time = t_complete - t_input_end
        if completion_time > COMPLETION_DEADLINE_S:
            failures.append(
                f"Completion time {completion_time:.2f}s exceeds {COMPLETION_DEADLINE_S}s"
            )

    return failures


async def run_harness(
    n_sessions: int,
    clip_path: str,
    url: str,
    gap_s: float = 2.0,
) -> list[dict]:
    """Run N sessions sequentially with gaps, checking rules."""
    results = []

    for i in range(n_sessions):
        print(f"\n{'='*60}")
        print(f"Session {i+1}/{n_sessions}")
        print(f"{'='*60}")

        t_start = time.perf_counter()

        try:
            trace = await run_session(url, clip_path)
            failures = check_rules(trace)

            result = {
                "session_num": i + 1,
                "session_id": trace.get("session_id"),
                "duration_s": time.perf_counter() - t_start,
                "n_frames_sent": trace.get("n_frames", 0),
                "n_chunks_received": len(trace.get("received_chunks", [])),
                "rule_failures": failures,
                "passed": len(failures) == 0,
            }

            if failures:
                print("  [FAILED]:")
                for f in failures:
                    print(f"    - {f}")
            else:
                print(f"  [PASSED] ({result['duration_s']:.2f}s)")

        except Exception as e:
            result = {
                "session_num": i + 1,
                "session_id": None,
                "duration_s": time.perf_counter() - t_start,
                "error": str(e),
                "rule_failures": [f"Exception: {e}"],
                "passed": False,
            }
            print(f"  ✗ ERROR: {e}")

        results.append(result)

        # Gap between sessions (§1)
        if i < n_sessions - 1:
            print(f"  Waiting {gap_s}s before next session...")
            await asyncio.sleep(gap_s)

    return results


def main():
    parser = argparse.ArgumentParser(description="Harness simulator")
    parser.add_argument("--sessions", "-n", type=int, default=5, help="Number of sessions")
    parser.add_argument("--clip", required=True, help="Path to test clip")
    parser.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket URL")
    parser.add_argument("--gap", type=float, default=2.0, help="Gap between sessions (s)")
    parser.add_argument("--output", "-o", default="out/harness_results.json", help="Output path")
    args = parser.parse_args()

    results = asyncio.run(
        run_harness(args.sessions, args.clip, args.url, args.gap)
    )

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"HARNESS SUMMARY: {passed}/{total} sessions passed")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
