"""Quick checks that each edit does what it claims, measured on a real sample.

Run:  .venv\\Scripts\\python check_pipeline.py
"""

from pathlib import Path

import numpy as np

from voiceeditor.analysis import (
    formant_ceiling, formant_tracks, summary, world_analyze, world_synthesize,
)
from voiceeditor import plots
from voiceeditor.audio_io import load_audio
from voiceeditor.edits import PRESETS, EditSettings, apply_edits
from voiceeditor.noise import (
    NOISE_TYPES, REMOVE_METHODS, AddNoise, RemoveNoise, add_noise, make_noise, remove_noise, speech_power,
)

SAMPLE = Path(__file__).parent / "samples" / "sample_male_en.wav"


def measure(y, fs, edited, ceiling):
    """Formants and duration are measured on the rebuilt audio. F0 numbers come from the
    edited pitch curve itself, because re-tracking pitch on rebuilt audio adds tracker errors."""
    return summary(edited, formant_tracks(y, fs, ceiling), y.size)


def main():
    x, fs = load_audio(SAMPLE.read_bytes())
    p = world_analyze(x, fs)
    ceiling = formant_ceiling(p.f0)
    base = measure(world_synthesize(p), fs, p, ceiling)
    print(f"original : F0 {base['mean_f0']:.0f} Hz, range {base['f0_range']:.0f} Hz, "
          f"F1 {base['median_f1']:.0f} Hz, {base['duration']:.2f} s")

    checks = {
        "pitch x1.5": (EditSettings(pitch=1.5), "mean_f0", 1.5),
        "range x0.4": (EditSettings(pitch_range=0.4), "f0_range", 0.4),
        "formant x1.2": (EditSettings(formant=1.2), "median_f1", 1.2),
        "speed x1.5": (EditSettings(speed=1.5), "duration", 1 / 1.5),
    }
    ok = True
    for name, (s, key, expected) in checks.items():
        e = apply_edits(p, s)
        m = measure(world_synthesize(e), fs, e, ceiling * s.formant)
        ratio = m[key] / base[key]
        good = abs(ratio - expected) / expected < 0.15
        ok &= good
        print(f"{name:13}: {key} ratio {ratio:.2f} (expected {expected:.2f}) {'OK' if good else 'FAIL'}")

    for name, s in PRESETS.items():
        y = world_synthesize(apply_edits(p, s))
        assert np.isfinite(y).all(), name
    print("presets  : all render finite audio")

    # Added noise must hit the requested SNR (speech power / noise power).
    for kind in NOISE_TYPES:
        noisy = add_noise(x, fs, AddNoise(True, kind, 10.0))
        noise = noisy - x
        snr = 10 * np.log10(speech_power(x, fs) / np.mean(noise ** 2))
        good = abs(snr - 10.0) < 0.5
        ok &= good
        print(f"add {kind:10}: SNR {snr:5.2f} dB (asked 10) {'OK' if good else 'FAIL'}")

    # Removal must cut the noise left in the pauses by more than half, for both methods.
    noisy = add_noise(x, fs, AddNoise(True, "White", 10.0))
    quiet = np.abs(x) < 0.01                      # samples where the speech is silent
    before = np.sqrt(np.mean(noisy[quiet] ** 2))
    for method in REMOVE_METHODS:
        clean = remove_noise(noisy, fs, RemoveNoise(True, method))
        after = np.sqrt(np.mean(clean[quiet] ** 2))
        good = after < 0.5 * before
        ok &= good
        print(f"remove ({method}): noise in pauses {before:.3f} -> {after:.3f} {'OK' if good else 'FAIL'}")

    # A click at a known (time, frequency) must map back to that time and frequency.
    ft = formant_tracks(x, fs, ceiling)
    img = plots.voice_panel(x, fs, p.f0, p.times, ft, True, True, x.size / fs)
    h, w = img.pixels.shape[:2]
    b = img.boxes["spec"]
    t_true, f_true = 1.5, 2000.0
    px = (b.left + (t_true - b.xlim[0]) / (b.xlim[1] - b.xlim[0]) * (b.right - b.left)) * w
    py = (b.bottom - (f_true - b.ylim[0]) / (b.ylim[1] - b.ylim[0]) * (b.bottom - b.top)) * h
    c = plots.click_to_data(img, {"x": px / 2, "y": py / 2, "width": w / 2, "height": h / 2})  # shown at half size
    good = c is not None and c.area == "spec" and abs(c.time - t_true) < 0.01 and abs(c.freq - f_true) < 20
    ok &= good
    print(f"click    : ({t_true}s, {f_true:.0f}Hz) -> ({c.time:.3f}s, {c.freq:.0f}Hz) {'OK' if good else 'FAIL'}")

    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
