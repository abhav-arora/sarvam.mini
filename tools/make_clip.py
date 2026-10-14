"""
make_clip.py (§5.1): Convert any audio recording to the wire format.

Output: mono, 24 kHz, 16-bit signed PCM (s16le), as a .wav file.

Usage:
    python tools/make_clip.py input.m4a clips/c01.wav
    python tools/make_clip.py input.m4a clips/c01.wav --variant noisy --snr 10
    python tools/make_clip.py input.m4a clips/c01.wav --variant telephone
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 24_000


def load_and_convert(input_path: str) -> np.ndarray:
    """Load any audio file and convert to mono 24 kHz float32."""
    data, sr = sf.read(input_path, dtype="float32", always_2d=True)

    # Convert to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample to 24 kHz if needed
    if sr != SAMPLE_RATE:
        from math import gcd

        g = gcd(sr, SAMPLE_RATE)
        up = SAMPLE_RATE // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.float32)

    return data


def add_noise(data: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Add white noise at a given SNR."""
    power = np.mean(data ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(data)).astype(np.float32) * np.sqrt(noise_power)
    return data + noise


def telephone_band(data: np.ndarray) -> np.ndarray:
    """Apply telephone band-limiting: 300–3400 Hz."""
    from scipy.signal import butter, sosfilt

    nyq = SAMPLE_RATE / 2
    low = 300.0 / nyq
    high = 3400.0 / nyq
    sos = butter(5, [low, high], btype="band", output="sos")
    return sosfilt(sos, data).astype(np.float32)


def generate_synthetic(duration_s: float = 10.0, with_pauses: bool = False) -> np.ndarray:
    """
    Generate synthetic speech-like audio with harmonics and envelope modulation.
    Optionally include realistic conversational pauses (e.g. 1-2s pauses).
    """
    n_samples = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)

    # Fundamental + formants
    f0 = 160.0  # F0 around 160 Hz
    harmonics = [1.0 * np.sin(2 * np.pi * f0 * t),
                 0.6 * np.sin(2 * np.pi * 2 * f0 * t),
                 0.4 * np.sin(2 * np.pi * 3 * f0 * t),
                 0.3 * np.sin(2 * np.pi * 5 * f0 * t),
                 0.2 * np.sin(2 * np.pi * 8 * f0 * t)]
    signal = sum(harmonics)

    # Syllabic amplitude envelope (modulation at 3-5 Hz)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    signal = signal * envelope

    if with_pauses:
        # Insert 1.5s pauses every ~4s
        pause_mask = (t % 5.0) < 3.5  # 3.5s speech, 1.5s pause
        signal = signal * pause_mask.astype(np.float32)

    # Normalize to -3 dB peak (~0.7)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = (signal / peak) * 0.7
    return signal.astype(np.float32)


def save_pcm_wav(data: np.ndarray, output_path: str) -> None:
    """Save as 24 kHz mono s16 WAV."""
    sf.write(output_path, data, SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audio to 24 kHz mono s16le PCM WAV, or generate synthetic test clips"
    )
    parser.add_argument("input", nargs="?", default=None, help="Input audio file (optional if --synthetic)")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument(
        "--synthetic",
        type=float,
        default=None,
        help="Generate synthetic clip with specified duration in seconds",
    )
    parser.add_argument(
        "--pauses",
        action="store_true",
        help="Include pauses in synthetic audio",
    )
    parser.add_argument(
        "--variant",
        choices=["clean", "noisy", "telephone"],
        default="clean",
        help="Variant to generate",
    )
    parser.add_argument("--snr", type=float, default=10.0, help="SNR in dB (for noisy variant)")
    args = parser.parse_args()

    if args.synthetic is not None:
        print(f"Generating synthetic clip ({args.synthetic}s, pauses={args.pauses})...")
        data = generate_synthetic(duration_s=args.synthetic, with_pauses=args.pauses)
    elif args.input:
        print(f"Loading {args.input}...")
        data = load_and_convert(args.input)
    else:
        parser.error("Either input file or --synthetic <duration> must be provided.")

    if args.variant == "noisy":
        print(f"Adding noise at SNR={args.snr} dB...")
        data = add_noise(data, snr_db=args.snr)
    elif args.variant == "telephone":
        print("Applying telephone band-limit (300–3400 Hz)...")
        data = telephone_band(data)

    duration = len(data) / SAMPLE_RATE
    print(f"Duration: {duration:.2f}s, samples: {len(data)}")

    save_pcm_wav(data, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
