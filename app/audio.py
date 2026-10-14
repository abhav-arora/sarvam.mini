"""
Audio utilities: PCM ↔ numpy, frame slicing, trim to valid_samples.
"""

from __future__ import annotations

import numpy as np

from app.protocol import BYTES_PER_SAMPLE, FRAME_BYTES, SAMPLE_RATE


def pcm_to_numpy(pcm: bytes) -> np.ndarray:
    """Convert raw PCM s16le bytes to a float32 numpy array in [-1, 1]."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def numpy_to_pcm(arr: np.ndarray) -> bytes:
    """Convert a float32 numpy array in [-1, 1] to raw PCM s16le bytes."""
    clamped = np.clip(arr, -1.0, 1.0)
    return (clamped * 32767).astype(np.int16).tobytes()


def slice_into_frames(pcm: bytes) -> list[bytes]:
    """
    Slice a PCM buffer into FRAME_BYTES-sized chunks.
    The last frame is zero-padded if necessary.
    """
    frames: list[bytes] = []
    for offset in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[offset : offset + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
        frames.append(chunk)
    return frames


def trim_to_valid_samples(pcm: bytes, valid_samples: int) -> bytes:
    """
    Trim PCM audio to exactly `valid_samples` samples.
    Returns bytes of length valid_samples * BYTES_PER_SAMPLE.
    """
    valid_bytes = valid_samples * BYTES_PER_SAMPLE
    return pcm[:valid_bytes]


def compute_valid_samples(total_pcm_bytes: int) -> int:
    """Compute the number of valid samples from total PCM byte length."""
    return total_pcm_bytes // BYTES_PER_SAMPLE


def duration_seconds(n_samples: int) -> float:
    """Duration of n_samples at the protocol sample rate."""
    return n_samples / SAMPLE_RATE


def ensure_even_bytes(pcm: bytes) -> bytes:
    """Ensure PCM data has an even number of bytes (whole samples)."""
    if len(pcm) % BYTES_PER_SAMPLE != 0:
        return pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)]
    return pcm
