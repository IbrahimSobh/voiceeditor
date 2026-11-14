"""Analysis: split speech into editable parameters, and measure it for display.

Two tools are used:
- WORLD (pyworld) splits speech into F0 (pitch), spectral envelope (vocal-tract
  shape) and aperiodicity (breathiness). These are the parameters we edit,
  and WORLD rebuilds speech from them.
- Praat (parselmouth) tracks the formants F1-F3, used only for display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth
import pyworld as pw

FRAME_PERIOD_MS = 5.0
F0_FLOOR, F0_CEIL = 60.0, 600.0


@dataclass
class WorldParams:
    f0: np.ndarray   # (frames,) Hz, 0 where unvoiced
    sp: np.ndarray   # (frames, bins) spectral envelope (power)
    ap: np.ndarray   # (frames, bins) aperiodicity 0..1
    fs: int
    frame_period: float = FRAME_PERIOD_MS

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.f0)) * self.frame_period / 1000.0


def world_analyze(x: np.ndarray, fs: int) -> WorldParams:
    """Decompose a signal with WORLD (Harvest pitch, CheapTrick envelope, D4C aperiodicity)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, fs, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_PERIOD_MS)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    return WorldParams(f0=f0, sp=sp, ap=ap, fs=fs)


def world_synthesize(p: WorldParams) -> np.ndarray:
    """Rebuild a waveform from (possibly edited) WORLD parameters."""
    return pw.synthesize(
        np.ascontiguousarray(p.f0, dtype=np.float64),
        np.ascontiguousarray(p.sp, dtype=np.float64),
        np.ascontiguousarray(p.ap, dtype=np.float64),
        p.fs,
        p.frame_period,
    )


@dataclass
class FormantTracks:
    times: np.ndarray       # (T,)
    values: np.ndarray      # (3, T) Hz for F1, F2, F3; NaN where undefined


def formant_ceiling(f0: np.ndarray) -> float:
    """Praat's usual advice: search formants up to 5000 Hz for low (male) voices, 5500 Hz for higher ones.

    After a formant edit by factor k, the ceiling must be multiplied by k too,
    otherwise Praat mislabels the shifted peaks.
    """
    v = f0[f0 > 0]
    return 5000.0 if v.size == 0 or np.median(v) < 160 else 5500.0


def formant_tracks(x: np.ndarray, fs: int, max_formant: float = 5500.0) -> FormantTracks:
    """Track F1-F3 with Praat's Burg method (10 ms steps)."""
    max_formant = min(max_formant, 0.95 * fs / 2)   # can't search above the Nyquist frequency
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=fs)
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=max_formant)
    times = np.asarray(fm.ts())
    values = np.array([[fm.get_value_at_time(i, t) for t in times] for i in (1, 2, 3)])
    return FormantTracks(times=times, values=values)


def voiced_at(times: np.ndarray, f0: np.ndarray, f0_times: np.ndarray) -> np.ndarray:
    """True where the nearest WORLD frame is voiced (used to hide formants in silence)."""
    idx = np.clip(np.searchsorted(f0_times, times), 0, len(f0) - 1)
    return f0[idx] > 0


def summary(p: WorldParams, ft: FormantTracks, n_samples: int) -> dict:
    """Numbers for the metric cards: mean F0, F0 range, median F1, duration."""
    v = p.f0[p.f0 > 0]
    mask = voiced_at(ft.times, p.f0, p.times)
    f1 = ft.values[0][mask]
    f1 = f1[~np.isnan(f1)]
    return {
        "mean_f0": float(v.mean()) if v.size else None,
        "f0_range": float(np.percentile(v, 95) - np.percentile(v, 5)) if v.size else None,
        "median_f1": float(np.median(f1)) if f1.size else None,
        "duration": n_samples / p.fs,
    }
