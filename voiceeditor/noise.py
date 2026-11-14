"""Adding and removing background noise.

Adding: generate a noise of a chosen colour and mix it at a chosen SNR
(signal-to-noise ratio, in dB). The speech level is measured only on frames
where someone is talking, so silence doesn't make the SNR look better.

Removing: estimate the noise spectrum from the quietest frames, then reduce
every frame by it. Two classic methods:
- spectral subtraction: |clean| = |noisy| - a * |noise|
- Wiener filter:        gain = SNR / (1 + SNR), a smoother rule with fewer "beeps"
A spectral floor keeps a little of the original in every bin, which limits
"musical noise" (short random tones left behind by the subtraction).
Both work well for steady noise (fan, hiss, hum) and poorly for changing noise.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft

NOISE_TYPES = ["White", "Pink", "Brown", "Hum 50 Hz", "Hum 60 Hz"]
REMOVE_METHODS = ["Spectral subtraction", "Wiener filter"]
NPERSEG, NOVERLAP = 512, 384          # 32 ms frames, 75 % overlap at 16 kHz (8 ms hop)
WIENER_SMOOTH_FRAMES = 5


@dataclass(frozen=True)
class AddNoise:
    enabled: bool = False
    kind: str = "White"
    snr_db: float = 10.0

    def key(self) -> tuple:
        return astuple(self)


@dataclass(frozen=True)
class RemoveNoise:
    enabled: bool = False
    method: str = "Spectral subtraction"
    strength: float = 1.0          # over-subtraction factor: 1 = remove the estimated noise once
    floor: float = 0.05            # keep at least this fraction of each bin's magnitude
    quiet_fraction: float = 0.10   # quietest share of frames used as the noise estimate

    def key(self) -> tuple:
        return astuple(self)


# ---------- adding noise ----------

def make_noise(kind: str, n: int, fs: int, seed: int = 0) -> np.ndarray:
    """Unit-power noise of the given kind."""
    rng = np.random.default_rng(seed)
    if kind.startswith("Hum"):
        base = 50.0 if "50" in kind else 60.0
        t = np.arange(n) / fs
        noise = sum(np.sin(2 * np.pi * base * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 9))
    else:
        white = rng.standard_normal(n)
        if kind == "White":
            noise = white
        else:
            # Shape white noise in the frequency domain: pink falls 3 dB/octave, brown 6 dB/octave
            spec = np.fft.rfft(white)
            f = np.fft.rfftfreq(n, 1 / fs)
            f[0] = f[1]
            spec /= np.sqrt(f) if kind == "Pink" else f
            noise = np.fft.irfft(spec, n)
    return noise / np.sqrt(np.mean(noise ** 2))


def speech_power(x: np.ndarray, fs: int) -> float:
    """Mean power of the active-speech frames.

    Active = 20 ms frames above 10 % of the 95th-percentile frame RMS. The percentile (not the
    maximum) keeps a single click or pop, common at the start of a recording, from setting the scale.
    """
    hop = int(0.02 * fs)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    active = rms > 0.1 * np.percentile(rms, 95) if rms.size else rms
    return float(np.mean(frames[active] ** 2)) if active.any() else float(np.mean(x ** 2))


def noise_parts(x: np.ndarray, fs: int, s: AddNoise) -> tuple[np.ndarray, np.ndarray]:
    """The speech and the noise inside the noisy mix, so that speech + noise = add_noise(x, fs, s).

    If the mix would clip, both are scaled down by the same gain. Keeping them apart gives the exact
    clean reference for the quality metrics, and the true "Noise alone" (x itself isn't, after a rescale).
    """
    if not s.enabled:
        return x, np.zeros_like(x)
    noise = make_noise(s.kind, x.size, fs) * np.sqrt(speech_power(x, fs) / 10 ** (s.snr_db / 10))
    peak = np.max(np.abs(x + noise))
    gain = 0.95 / peak if peak > 1.0 else 1.0           # rescale only if it would clip
    return x * gain, noise * gain


def add_noise(x: np.ndarray, fs: int, s: AddNoise) -> np.ndarray:
    """Mix noise into x so that speech power / noise power = s.snr_db."""
    if not s.enabled:
        return x
    speech, noise = noise_parts(x, fs, s)
    return speech + noise


# ---------- removing noise ----------

def removal_gain(x: np.ndarray, fs: int, s: RemoveNoise) -> np.ndarray:
    """The gain (0..1) that noise removal gives every frequency of every STFT frame of x."""
    _, _, X = stft(x, fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    mag = np.abs(X)

    energy = (mag ** 2).sum(axis=0)
    n_quiet = max(1, int(s.quiet_fraction * mag.shape[1]))
    # Average the noise *power*: averaging magnitudes underestimates it (by ~20 % for random noise)
    noise_pow = (mag[:, np.argsort(energy)[:n_quiet]] ** 2).mean(axis=1, keepdims=True)
    noise = np.sqrt(noise_pow)

    if s.method == "Wiener filter":
        # Smooth power over 5 frames (40 ms) first: the raw per-frame SNR jumps randomly and lets noise through
        smoothed = uniform_filter1d(mag ** 2, WIENER_SMOOTH_FRAMES, axis=1)
        snr_post = smoothed / (s.strength * noise_pow + 1e-12)
        snr_prior = np.maximum(snr_post - 1.0, 0.0)
        return np.maximum(snr_prior / (1.0 + snr_prior), s.floor)
    # spectral subtraction, written as a gain: |clean| / |noisy|
    clean = np.maximum(mag - s.strength * noise, s.floor * mag)
    return np.divide(clean, mag, out=np.ones_like(mag), where=mag > 0)


def apply_gain(x: np.ndarray, fs: int, gain: np.ndarray) -> np.ndarray:
    """Multiply x's STFT by the gain (keeping its phase) and rebuild a signal of the same length."""
    _, _, X = stft(x, fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    _, y = istft(gain * X, fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    y = y[: x.size]
    return np.pad(y, (0, x.size - y.size))


def remove_noise(x: np.ndarray, fs: int, s: RemoveNoise) -> np.ndarray:
    """Reduce steady background noise with spectral subtraction or a Wiener filter."""
    if not s.enabled or x.size < NPERSEG:
        return x
    return apply_gain(x, fs, removal_gain(x, fs, s))


def split_result(speech: np.ndarray, noise: np.ndarray, fs: int, s: RemoveNoise) -> tuple[np.ndarray, np.ndarray]:
    """What noise removal does to the speech and to the noise, separately.

    The gains are computed from the noisy mix, as usual. Applying a fixed gain to an STFT is linear,
    so the same gains applied to the speech and to the noise alone give two signals that add up
    exactly to remove_noise(speech + noise). Then:
      damage to the voice = speech_out - speech,   leftover noise = noise_out.
    """
    if not s.enabled or speech.size < NPERSEG:
        return speech, noise
    gain = removal_gain(speech + noise, fs, s)
    return apply_gain(speech, fs, gain), apply_gain(noise, fs, gain)
