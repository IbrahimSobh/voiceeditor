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


# ---------- the numbers in plain words ----------
#
# "Mess" = everything in the result that isn't the clean voice: leftover noise plus damage to the voice.
# dB are turned into "times stronger" (power ratio 10^(dB/10)). The verdict bands follow the SNR
# lesson (20 dB mildly noisy, 5 dB clearly noisy, 0 dB equal); the LSD bands are a rough guide only.

SNR_BANDS = [  # upper limit (dB), verdict, level 0 (bad) … 4 (good)
    (0.0, "noise wins", 0),
    (10.0, "heavy noise", 1),
    (20.0, "noticeable noise", 2),
    (30.0, "slight noise", 3),
    (np.inf, "almost clean", 4),
]
LSD_BANDS = [(2.0, "almost identical", 4), (5.0, "close", 3), (10.0, "clearly different", 2),
             (np.inf, "very different", 0)]


def times(db: float) -> str:
    """A power ratio in dB as words: 13 dB -> "20×"."""
    r = 10 ** (db / 10)
    if r >= 1e6:
        return "over a million times"
    return f"{r:,.0f}×" if r >= 9.95 else f"{r:.1f}×"


def verdict(key: str, value: float) -> tuple[str, int]:
    """A short verdict for one number, and its level from 0 (bad) to 4 (good)."""
    for limit, word, level in LSD_BANDS if key == "lsd" else SNR_BANDS:
        if value < limit:
            return word, level
    return SNR_BANDS[-1][1], SNR_BANDS[-1][2]


def in_words(key: str, value: float) -> str:
    """One plain sentence saying what a number means."""
    if not np.isfinite(value):
        return "No speech to measure."
    if key == "lsd":
        return (f"Look at which frequencies are present, frame by frame: on average each one is {value:.1f} dB "
                "away from the clean voice. 0 dB would mean the same sound colour.")
    strength = (f"the voice is **{times(value)} stronger** than the mess" if value >= 0
                else f"the mess is **{times(-value)} stronger** than the voice")
    return {
        "snr": f"Over the whole recording, {strength}.",
        "seg_snr": f"Checking each short piece of speech (32 ms) on its own, so quiet syllables count too: "
                   f"typically {strength}.",
        "si_sdr": f"Like SNR, but the result may be louder or quieter without penalty: {strength}.",
    }[key]


def change_in_words(key: str, before: float, after: float) -> str:
    """One plain sentence about a change, e.g. from the noisy input to the denoised result."""
    d = after - before
    if not np.isfinite(d):
        return ""
    if key == "lsd":
        return (f"{abs(d):.1f} dB {'closer to' if d < 0 else 'further from'} the clean voice's sound colour "
                f"({'better' if d < 0 else 'worse'}).")
    return (f"The mess got **{times(d)} weaker** (better)." if d >= 0
            else f"The mess got **{times(-d)} stronger** (worse).")


DB_CHEAT_SHEET = ("Reading dB: +3 dB ≈ 2× stronger · +10 dB = 10× · +20 dB = 100× · 0 dB = equal · "
                  "negative = the mess is stronger than the voice.")


# ---------- going deeper: what the mess is made of, the math, the worst moment ----------

def mess_breakdown(speech: np.ndarray, speech_out: np.ndarray, noise_out: np.ndarray) -> dict[str, float]:
    """Split the mess into leftover noise and damage to the voice (see noise.split_result).

    Powers are summed over the whole signal. The two parts are nearly independent, so their powers
    add up to the mess power (within a few %); the shares are given out of their sum.
    """
    damage, leftover = speech_out - speech, noise_out
    p_s, p_d, p_n = (float(np.sum(v ** 2)) for v in (speech, damage, leftover))
    total = p_d + p_n + EPS
    return {
        "noise_share": p_n / total,
        "damage_share": p_d / total,
        "voice_vs_noise_db": float(10 * np.log10(p_s / (p_n + EPS) + EPS)),    # higher = less noise left
        "voice_vs_damage_db": float(10 * np.log10(p_s / (p_d + EPS) + EPS)),    # higher = less damage
    }


def worked_numbers(ref: np.ndarray, est: np.ndarray, fs: int) -> dict[str, float]:
    """The intermediate numbers behind each metric, to show the math with real values."""
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    track, speech = segmental_snr_track(ref, est)
    raw = 10 * np.log10(np.sum(_frames(ref) ** 2, axis=1) / (np.sum(_frames(ref - est) ** 2, axis=1) + EPS) + EPS)
    s, s_hat = ref - ref.mean(), est - est.mean()
    a = np.dot(s_hat, s) / (np.dot(s, s) + EPS)
    return {
        "p_speech": float(np.sum(ref ** 2)),
        "p_error": float(np.sum((ref - est) ** 2)),
        "frames": int(track.size),
        "speech_frames": int(speech.sum()),
        "clamped_low": int(np.sum(raw[speech] < SEG_MIN_DB)),
        "clamped_high": int(np.sum(raw[speech] > SEG_MAX_DB)),
        "seg_min": float(track[speech].min()) if speech.any() else float("nan"),
        "seg_max": float(track[speech].max()) if speech.any() else float("nan"),
        "alpha": float(a),
        "p_target": float(np.sum((a * s) ** 2)),
        "p_distortion": float(np.sum((s_hat - a * s) ** 2)),
        "bins": NPERSEG // 2 + 1,
        "hz_per_bin": fs / NPERSEG,
    }


def worst_moment(ref: np.ndarray, est: np.ndarray, fs: int) -> tuple[float, float] | None:
    """Time (s, frame centre) and SNR (dB) of the worst speech frame, or None without speech."""
    track, speech = segmental_snr_track(ref, est)
    if not speech.any():
        return None
    i = int(np.flatnonzero(speech)[np.argmin(track[speech])])
    return (i * (NPERSEG - NOVERLAP) + NPERSEG / 2) / fs, float(track[i])


def disagreements(q: dict[str, float], base: dict[str, float] | None = None) -> list[str]:
    """Plain-words notes on why the four numbers don't tell the same story, when they don't."""
    notes = []
    if q["si_sdr"] - q["snr"] > 1.0:
        notes.append(f"**SI-SDR is {q['si_sdr'] - q['snr']:.1f} dB above SNR.** Part of what SNR counts as mess is "
                     "only a change of loudness: noise removal also turns the voice down a little. SI-SDR "
                     "forgives that, SNR doesn't.")
    if q["seg_snr"] < q["snr"] - 2.0:
        notes.append(f"**Segmental SNR is {q['snr'] - q['seg_snr']:.1f} dB below SNR.** The quiet parts of the "
                     "speech are messier than the loud ones; one global SNR hides that because loud vowels "
                     "dominate it.")
    elif q["seg_snr"] > q["snr"] + 2.0:
        notes.append(f"**Segmental SNR is {q['seg_snr'] - q['snr']:.1f} dB above SNR.** The mess sits mostly in a "
                     "few loud moments; most syllables are cleaner than the global number suggests.")
    if base is not None and q["si_sdr"] > base["si_sdr"] and q["lsd"] > base["lsd"]:
        notes.append(f"**The SNRs improved but LSD got worse ({base['lsd']:.1f} → {q['lsd']:.1f} dB).** The noise "
                     "only touched a few frequencies (hum does this), so the sound colour was already close. "
                     "Noise removal turned many frequencies down a little, which the SNRs barely notice and LSD, "
                     "which treats every frequency equally, does.")
    if base is not None and q["si_sdr"] < base["si_sdr"]:
        notes.append("**The result is worse than the noisy input.** The removal damages more voice than the noise "
                     "it takes away: lower the Strength or raise the spectral floor.")
    return notes
