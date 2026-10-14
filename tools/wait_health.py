"""
wait_health.py (§1 / §6.3): Wait for server /health endpoint to return 200.

Measures cold-start time and verifies startup readiness (§4.6).

Usage:
    python tools/wait_health.py
    python tools/wait_health.py --url http://localhost:8000/health --timeout 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def wait_for_health(url: str, timeout_s: float = 300.0, poll_interval_s: float = 0.5) -> float:
    """
    Poll url until HTTP 200 with status ok is received.
    Returns elapsed time in seconds.
    Raises TimeoutError if timeout_s is exceeded.
    """
    start_time = time.perf_counter()
    deadline = start_time + timeout_s

    while time.perf_counter() < deadline:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wait_health/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode("utf-8"))
                    if body.get("status") == "ok":
                        elapsed = time.perf_counter() - start_time
                        return elapsed
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            pass

        time.sleep(poll_interval_s)

    raise TimeoutError(f"Server at {url} did not become healthy within {timeout_s} seconds.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for server /health endpoint to return 200")
    parser.add_argument("--url", default="http://localhost:8000/health", help="Health check URL")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout in seconds")
    args = parser.parse_args()

    print(f"Waiting for {args.url} (timeout: {args.timeout}s)...")
    try:
        elapsed = wait_for_health(args.url, timeout_s=args.timeout)
        print(f"Server ready in {elapsed:.3f} s (HTTP 200)")
    except TimeoutError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
