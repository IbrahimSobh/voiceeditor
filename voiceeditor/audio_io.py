"""Loading and saving audio.

Every recording is converted to mono, resampled to one working rate and
peak-normalised, so the rest of the pipeline can assume a clean input.
"""

from __future__ import annotations

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

WORK_FS = 16000        # 16 kHz is plenty for speech and keeps WORLD fast
MAX_SECONDS = 30.0     # longer files are cut to keep analysis responsive


def load_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes (WAV, FLAC, OGG, MP3) into a mono float64 signal at WORK_FS."""
    x, fs = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    x = x.mean(axis=1)                                  # stereo -> mono
    if fs != WORK_FS:
        g = gcd(int(fs), WORK_FS)
        x = resample_poly(x, WORK_FS // g, int(fs) // g)
    x = x[: int(MAX_SECONDS * WORK_FS)]
    return normalize(x), WORK_FS


def normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale the signal so its loudest sample is at `peak`."""
    m = np.max(np.abs(x)) if x.size else 0.0
    return x * (peak / m) if m > 0 else x


def to_wav_bytes(x: np.ndarray, fs: int) -> bytes:
    """Encode a signal as 16-bit WAV bytes (for download buttons)."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), fs, subtype="PCM_16", format="WAV")
    return buf.getvalue()
