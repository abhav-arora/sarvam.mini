"""
analyze.py (§5.3): Playback-timeline simulator and metrics.

Reconstructs what the listener hears using the PS rule:
a frame starts when it arrives or when the previous finishes, whichever is later.

Usage:
    python tools/analyze.py out/c01.json
    python tools/analyze.py out/c01.json --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.protocol import BYTES_PER_SAMPLE, MAX_OUT_BYTES, SAMPLE_RATE


def simulate_playback(chunks: list[dict]) -> list[dict]:
    """
    Reconstruct the playback timeline.

    A frame starts when it arrives or when the previous finishes,
    whichever is later.
    """
    cursor = 0.0
    timeline = []

    for chunk in chunks:
        t_arr = chunk["t_arrival"]
        nbytes = chunk["nbytes"]
        duration = nbytes / BYTES_PER_SAMPLE / SAMPLE_RATE

        start = max(t_arr, cursor)
        end = start + duration
        cursor = end

        timeline.append({
            "t_arrival": t_arr,
            "t_play_start": start,
            "t_play_end": end,
            "nbytes": nbytes,
            "duration_s": duration,
        })

    return timeline


def compute_metrics(trace: dict, timeline: list[dict]) -> dict:
    """Compute all metrics from §5.3."""
    metrics = {}

    valid_samples = trace.get("valid_samples", 0)
    input_duration = valid_samples / SAMPLE_RATE if valid_samples else 0

    # First-audio delay: first output playback start minus session start
    if timeline:
        metrics["first_audio_delay_s"] = timeline[0]["t_play_start"]
    else:
        metrics["first_audio_delay_s"] = None

    # Tail latency: end of output playback minus end of valid input
    if timeline:
        last_play_end = timeline[-1]["t_play_end"]
        t_input_end = trace.get("t_input_end", 0)
        metrics["tail_latency_s"] = last_play_end - t_input_end
    else:
        metrics["tail_latency_s"] = None

    # Completion time: session_complete arrival minus input_end send time
    t_complete = trace.get("t_session_complete", 0)
    t_input_end = trace.get("t_input_end", 0)
    if t_complete and t_input_end:
        metrics["completion_time_s"] = t_complete - t_input_end
    else:
        metrics["completion_time_s"] = None

    # Rule checks
    rules = {
        "all_chunks_le_max": True,
        "all_chunks_even_bytes": True,
        "session_complete_received": trace.get("t_session_complete", 0) > 0,
        "completion_within_30s": True,
    }

    for chunk in trace.get("received_chunks", []):
        if chunk["nbytes"] > MAX_OUT_BYTES:
            rules["all_chunks_le_max"] = False
        if chunk["nbytes"] % BYTES_PER_SAMPLE != 0:
            rules["all_chunks_even_bytes"] = False

    if metrics["completion_time_s"] is not None:
        rules["completion_within_30s"] = metrics["completion_time_s"] <= 30.0

    metrics["rules"] = rules

    # Total output duration
    if timeline:
        metrics["total_output_duration_s"] = sum(e["duration_s"] for e in timeline)
    else:
        metrics["total_output_duration_s"] = 0

    metrics["input_duration_s"] = input_duration
    metrics["n_output_chunks"] = len(trace.get("received_chunks", []))

    return metrics


def plot_timeline(trace: dict, timeline: list[dict], output_path: str) -> None:
    """Generate the showcase figure: input and output waveforms on one time axis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Input timeline
    ax_in = axes[0]
    frame_ms = 80
    for sf_info in trace.get("sent_frames", []):
        t = sf_info["t"]
        ax_in.barh(0, frame_ms / 1000, left=t, height=0.5, color="#2196F3", alpha=0.7)
    ax_in.set_ylabel("Input")
    ax_in.set_yticks([])
    ax_in.set_title("Session Timeline")

    # Mark input_end
    t_ie = trace.get("t_input_end", 0)
    if t_ie:
        ax_in.axvline(t_ie, color="red", linestyle="--", label="input_end")
        ax_in.legend()

    # Output timeline
    ax_out = axes[1]
    for entry in timeline:
        duration = entry["duration_s"]
        start = entry["t_play_start"]
        ax_out.barh(0, duration, left=start, height=0.5, color="#4CAF50", alpha=0.7)
    ax_out.set_ylabel("Output")
    ax_out.set_xlabel("Time (s)")
    ax_out.set_yticks([])

    # Mark session_complete
    t_sc = trace.get("t_session_complete", 0)
    if t_sc:
        ax_out.axvline(t_sc, color="red", linestyle="--", label="session_complete")
        ax_out.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Playback-timeline analyzer")
    parser.add_argument("trace", help="Path to JSON trace from client.py")
    parser.add_argument("--plot", action="store_true", help="Generate timeline plot")
    parser.add_argument("--plot-output", default=None, help="Plot output path")
    args = parser.parse_args()

    with open(args.trace) as f:
        trace = json.load(f)

    chunks = trace.get("received_chunks", [])
    timeline = simulate_playback(chunks)
    metrics = compute_metrics(trace, timeline)

    print("=" * 60)
    print(f"Session: {trace.get('session_id', 'unknown')}")
    print(f"Clip: {trace.get('clip', 'unknown')}")
    print("=" * 60)
    print(f"Input duration:      {metrics['input_duration_s']:.3f} s")
    print(f"Output chunks:       {metrics['n_output_chunks']}")
    print(f"Output duration:     {metrics['total_output_duration_s']:.3f} s")
    print(f"First-audio delay:   {metrics['first_audio_delay_s']:.3f} s" if metrics['first_audio_delay_s'] else "First-audio delay:   N/A")
    print(f"Tail latency:        {metrics['tail_latency_s']:.3f} s" if metrics['tail_latency_s'] else "Tail latency:        N/A")
    print(f"Completion time:     {metrics['completion_time_s']:.3f} s" if metrics['completion_time_s'] else "Completion time:     N/A")
    print("-" * 60)
    print("Rule checks:")
    for rule, passed in metrics.get("rules", {}).items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {rule}: {status}")
    print("=" * 60)

    if args.plot:
        plot_path = args.plot_output
        if not plot_path:
            base = os.path.splitext(args.trace)[0]
            plot_path = f"{base}_timeline.png"
        plot_timeline(trace, timeline, plot_path)


if __name__ == "__main__":
    main()
