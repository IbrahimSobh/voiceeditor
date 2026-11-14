"""Quality metrics: how close a processed signal is to the clean original, in numbers.

All four compare an estimate (noisy or denoised speech) with a clean reference of the
same length and timing, so they can only be used when the clean original is known: here,
when the app adds the noise itself. Higher is better for the three SNRs, lower for LSD.

- SNR            whole-signal signal-to-error ratio (classic, dominated by loud parts)
- Segmental SNR  the same per 32 ms frame, clamped to [-10, 35] dB and averaged over speech
- SI-SDR         scale-invariant SDR: the estimate is first rescaled to best match the reference
- LSD            log-spectral distance: average dB difference between the two spectra

Frames match noise.py (32 ms, 75 % overlap), and "speech frames" use the same rule as
noise.speech_power, so all numbers describe the parts where someone is talking.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from .noise import NOVERLAP, NPERSEG

SPEECH_DB = -20.0                        # speech frame: RMS within 20 dB of the 95th-percentile frame
SEG_MIN_DB, SEG_MAX_DB = -10.0, 35.0     # usual clamp for segmental SNR (Hansen and Pellom, 1998)
EPS = 1e-12

METRICS = {  # key: (label, unit, higher is better, one-line help)
    "snr": ("SNR", "dB", True, "Power of the clean speech / power of the error, over the whole signal."),
    "seg_snr": ("Segmental SNR", "dB", True, "SNR per 32 ms frame, clamped to −10…35 dB, averaged over speech frames. "
                                            "Quiet syllables count as much as loud ones."),
    "si_sdr": ("SI-SDR", "dB", True, "Scale-invariant SDR: rescales the estimate to best fit the clean speech, "
                                     "then compares. A change of volume alone doesn't count as an error."),
    "lsd": ("LSD", "dB", False, "Log-spectral distance: RMS difference of the two spectra in dB, averaged over "
                                "speech frames. Phase is ignored. Lower is better."),
}


def _frames(x: np.ndarray) -> np.ndarray:
    """Cut x into 32 ms frames with 75 % overlap, one frame per row (the same grid as noise.py's STFT)."""
    hop = NPERSEG - NOVERLAP
    if x.size < NPERSEG:
        x = np.pad(x, (0, NPERSEG - x.size))
    n = 1 + (x.size - NPERSEG) // hop
    idx = np.arange(NPERSEG)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def speech_frames(ref: np.ndarray) -> np.ndarray:
    """True for frames where the reference is talking: RMS above 10 % (-20 dB) of the 95th-percentile frame."""
    rms = np.sqrt(np.mean(_frames(ref) ** 2, axis=1))
    return rms > 10 ** (SPEECH_DB / 20) * np.percentile(rms, 95) if rms.any() else np.zeros(rms.size, bool)


def snr(ref: np.ndarray, est: np.ndarray) -> float:
    """10 log10( sum s^2 / sum (s - s_hat)^2 )."""
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + EPS) + EPS))


def segmental_snr_track(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame SNR (dB, clamped) and the speech-frame mask, for plotting and averaging."""
    s, e = _frames(ref), _frames(ref - est)
    track = 10 * np.log10(np.sum(s ** 2, axis=1) / (np.sum(e ** 2, axis=1) + EPS) + EPS)
    return np.clip(track, SEG_MIN_DB, SEG_MAX_DB), speech_frames(ref)


def segmental_snr(ref: np.ndarray, est: np.ndarray) -> float:
    """Mean of the clamped per-frame SNR over speech frames."""
    track, speech = segmental_snr_track(ref, est)
    return float(track[speech].mean()) if speech.any() else float("nan")


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR (Le Roux et al., 2019).

    Project the estimate on the reference: target = a*s with a = <s_hat, s> / ||s||^2,
    everything else is error. Means are removed first, so a DC offset isn't counted either.
    """
    s, s_hat = ref - ref.mean(), est - est.mean()
    a = np.dot(s_hat, s) / (np.dot(s, s) + EPS)
    target = a * s
    return float(10 * np.log10(np.sum(target ** 2) / (np.sum((s_hat - target) ** 2) + EPS) + EPS))


def log_spectral_distance(ref: np.ndarray, est: np.ndarray, fs: int) -> float:
    """sqrt( mean over frequency of (10 log10 |S|^2 - 10 log10 |S_hat|^2)^2 ), averaged over speech frames.

    Powers are floored 80 dB below the loudest reference bin, so empty bins (digital silence,
    or frequencies a filter removed completely) give a large but finite distance.
    """
    if ref.size < NPERSEG:                  # one whole frame at least (scipy would shrink the frame and fail)
        ref, est = np.pad(ref, (0, NPERSEG - ref.size)), np.pad(est, (0, NPERSEG - est.size))
    _, _, S = stft(ref, fs, nperseg=NPERSEG, noverlap=NOVERLAP, boundary=None, padded=False)
    _, _, E = stft(est, fs, nperseg=NPERSEG, noverlap=NOVERLAP, boundary=None, padded=False)
    p_ref, p_est = np.abs(S) ** 2, np.abs(E) ** 2
    floor = p_ref.max() * 1e-8 + EPS
    diff = 10 * np.log10(np.maximum(p_ref, floor)) - 10 * np.log10(np.maximum(p_est, floor))
    per_frame = np.sqrt(np.mean(diff ** 2, axis=0))
    speech = speech_frames(ref)[: per_frame.size]
    return float(per_frame[speech].mean()) if speech.any() else float("nan")


def quality(ref: np.ndarray, est: np.ndarray, fs: int) -> dict[str, float]:
    """All four metrics of `est` against the clean `ref` (same length, same timing)."""
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    return {
        "snr": snr(ref, est),
        "seg_snr": segmental_snr(ref, est),
        "si_sdr": si_sdr(ref, est),
        "lsd": log_spectral_distance(ref, est, fs),
    }
