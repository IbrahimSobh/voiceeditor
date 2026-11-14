"""Edits on WORLD parameters, and the presets built from them.

Each edit changes one parameter and leaves the others alone:
- pitch       scales F0 (how fast the vocal cords vibrate)
- pitch_range widens or flattens the F0 curve around its mean (liveliness)
- formant     stretches the spectral envelope along frequency (vocal-tract size)
- speed       shortens or lengthens the parameter frames in time (pitch unchanged)
A preset is a stored set of these numbers; the user can fine-tune it after.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, replace

import numpy as np

from .analysis import WorldParams


@dataclass(frozen=True)
class EditSettings:
    pitch: float = 1.0
    pitch_range: float = 1.0
    formant: float = 1.0
    speed: float = 1.0

    def key(self) -> tuple:
        return tuple(round(v, 4) for v in astuple(self))

    def is_neutral(self) -> bool:
        return all(abs(v - 1.0) < 1e-9 for v in astuple(self))


# Rough rules from phonetics: female voices have ~1.6-1.8x higher F0 and
# ~15-20 % higher formants than male voices. Happy speech: higher, livelier
# pitch and faster rate; sad speech: lower, flatter pitch and slower rate.
PRESETS: dict[str, EditSettings] = {
    "Neutral (original)": EditSettings(),
    "More female": EditSettings(pitch=1.6, pitch_range=1.1, formant=1.17),
    "More male": EditSettings(pitch=0.65, pitch_range=0.9, formant=0.88),
    "Happy": EditSettings(pitch=1.15, pitch_range=1.5, formant=1.03, speed=1.12),
    "Sad": EditSettings(pitch=0.9, pitch_range=0.4, formant=0.98, speed=0.82),
}


def shift_pitch(f0: np.ndarray, factor: float, range_factor: float) -> np.ndarray:
    """Scale F0 and its spread, working on log-F0 so changes sound even (musical)."""
    out = f0.copy()
    voiced = out > 0
    if voiced.any():
        log_f0 = np.log(out[voiced])
        mean = log_f0.mean()
        out[voiced] = np.exp(mean + (log_f0 - mean) * range_factor) * factor
        out[voiced] = np.clip(out[voiced], 40.0, 1200.0)
    return out


def warp_formants(sp: np.ndarray, factor: float) -> np.ndarray:
    """Stretch the spectral envelope along frequency: peak at f moves to f * factor.

    Interpolation is done on log power so the peaks keep their shape.
    """
    if factor == 1.0:
        return sp
    bins = sp.shape[1]
    src = np.clip(np.arange(bins) / factor, 0, bins - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, bins - 1)
    w = src - i0
    log_sp = np.log(np.maximum(sp, 1e-16))
    return np.exp((1 - w) * log_sp[:, i0] + w * log_sp[:, i1])


def stretch_time(p: WorldParams, speed: float) -> WorldParams:
    """Resample the frame sequence: speed 2 keeps every other frame, 0.5 repeats frames."""
    if speed == 1.0:
        return p
    n = len(p.f0)
    idx = np.linspace(0, n - 1, max(2, int(round(n / speed))))
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0)[:, None]
    f0 = p.f0[np.round(idx).astype(int)]     # nearest frame: never blends voiced with unvoiced
    sp = (1 - w) * p.sp[i0] + w * p.sp[i1]
    ap = (1 - w) * p.ap[i0] + w * p.ap[i1]
    return replace(p, f0=f0, sp=sp, ap=ap)


def apply_edits(p: WorldParams, s: EditSettings) -> WorldParams:
    """Apply pitch, pitch-range, formant and speed edits (noise is handled earlier, in noise.py)."""
    p = replace(
        p,
        f0=shift_pitch(p.f0, s.pitch, s.pitch_range),
        sp=warp_formants(p.sp, s.formant),
    )
    return stretch_time(p, s.speed)
