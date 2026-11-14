"""Generate the before/after audio examples and spectrogram images shown in README.md.

Run:  .venv\\Scripts\\python make_examples.py
Writes docs/audio/*.wav and docs/img/example_*.png. Same code paths as the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from voiceeditor import plots
from voiceeditor.analysis import WorldParams, formant_ceiling, formant_tracks, world_analyze, world_synthesize
from voiceeditor.audio_io import load_audio, normalize, to_wav_bytes
from voiceeditor.edits import PRESETS, EditSettings, apply_edits
from voiceeditor.noise import AddNoise, RemoveNoise, add_noise, remove_noise

ROOT = Path(__file__).parent
AUDIO, IMG = ROOT / "docs" / "audio", ROOT / "docs" / "img"


@dataclass
class Example:
    name: str
    sample: str
    voice: EditSettings = EditSettings()
    add: AddNoise = AddNoise()
    remove: RemoveNoise = RemoveNoise()
    title: str = ""
    before: str = ""          # file the "before" player uses; default: the original sample


ORIGINALS = {  # sample file -> name of its copy in docs/audio
    "sample_male_ar.wav": "original_male_ar",
    "sample_male_en.wav": "original_male",
    "sample_female_en.wav": "original_female",
}


def before_name(ex: Example) -> str:
    return ex.before or ORIGINALS[ex.sample]


EXAMPLES = [
    Example("ar_male_to_female", "sample_male_ar.wav", PRESETS["More female"],
            title="Arabic: male → female (More female: F0 ×1.6, range ×1.1, formants ×1.17)"),
    Example("ar_sad", "sample_male_ar.wav", PRESETS["Sad"], title="Arabic: sad (lower, flatter pitch, 18 % slower)"),
    Example("ar_happy", "sample_male_ar.wav", PRESETS["Happy"],
            title="Arabic: happy (higher, livelier pitch, 12 % faster)"),
    Example("male_to_female", "sample_male_en.wav", PRESETS["More female"],
            title="Male → female (More female: F0 ×1.6, range ×1.1, formants ×1.17)"),
    Example("female_to_male", "sample_female_en.wav", PRESETS["More male"],
            title="Female → male (More male: F0 ×0.65, range ×0.9, formants ×0.88)"),
    Example("happy", "sample_female_en.wav", PRESETS["Happy"], title="Happy: higher, livelier pitch, 12 % faster"),
    Example("sad", "sample_female_en.wav", PRESETS["Sad"], title="Sad: lower, flatter pitch, 18 % slower"),
    Example("slower_0.75x", "sample_male_en.wav", EditSettings(speed=0.75), title="Slower, speed ×0.75 (same pitch)"),
    Example("faster_1.5x", "sample_male_en.wav", EditSettings(speed=1.5), title="Faster, speed ×1.5 (same pitch)"),
    Example("noisy_white_10dB", "sample_male_en.wav", add=AddNoise(True, "White", 10.0),
            title="Add white noise at 10 dB SNR"),
    Example("denoised_spectral_subtraction", "sample_male_en.wav", add=AddNoise(True, "White", 10.0),
            remove=RemoveNoise(True, "Spectral subtraction"), title="Remove it: spectral subtraction",
            before="noisy_white_10dB"),
    Example("denoised_wiener", "sample_male_en.wav", add=AddNoise(True, "White", 10.0),
            remove=RemoveNoise(True, "Wiener filter"), title="Remove it: Wiener filter", before="noisy_white_10dB"),
    Example("hum_50Hz_10dB", "sample_female_en.wav", add=AddNoise(True, "Hum 50 Hz", 10.0),
            title="Add 50 Hz mains hum at 10 dB SNR"),
    Example("hum_removed_wiener", "sample_female_en.wav", add=AddNoise(True, "Hum 50 Hz", 10.0),
            remove=RemoveNoise(True, "Wiener filter", strength=1.5), title="Remove the hum: Wiener filter, strength ×1.5",
            before="hum_50Hz_10dB"),
]
IMAGES = {"ar_male_to_female", "male_to_female", "denoised_wiener"}     # before/after spectrogram pictures for these


def process(x: np.ndarray, fs: int, ex: Example):
    """Same chain as the app: add noise -> remove noise -> voice edits (WORLD only if needed)."""
    y = remove_noise(add_noise(x, fs, ex.add), fs, ex.remove)
    p = world_analyze(y, fs)
    if ex.voice.is_neutral():
        return y, p.f0
    edited = apply_edits(p, ex.voice)
    return normalize(world_synthesize(edited), float(np.max(np.abs(y)))), edited.f0


def picture(x, y, f0_x, f0_y, fs, ex: Example, path: Path) -> None:
    ceiling = formant_ceiling(f0_x)
    x_max = max(x.size, y.size) / fs
    tiles = []
    for sig, f0, c in ((x, f0_x, ceiling), (y, f0_y, ceiling * ex.voice.formant)):
        t = np.arange(len(f0)) * 0.005
        tiles.append(plots.voice_panel(sig, fs, f0, t, formant_tracks(sig, fs, c), True, True, x_max,
                                       size=(6.2, 3.6)).pixels)
    gap = np.full((tiles[0].shape[0], 20, 3), 255, dtype=np.uint8)
    Image.fromarray(np.hstack([tiles[0], gap, tiles[1]])).save(path, optimize=True)


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    originals = {}
    for name, copy in ORIGINALS.items():
        x, fs = load_audio((ROOT / "samples" / name).read_bytes())
        originals[name] = (x, fs, world_analyze(x, fs).f0)
        (AUDIO / f"{copy}.wav").write_bytes(to_wav_bytes(x, fs))
    for ex in EXAMPLES:
        x, fs, f0_x = originals[ex.sample]
        y, f0_y = process(x, fs, ex)
        (AUDIO / f"{ex.name}.wav").write_bytes(to_wav_bytes(y, fs))
        if ex.name in IMAGES:
            before = add_noise(x, fs, ex.add) if ex.remove.enabled else x   # denoising: compare with the noisy input
            f0_b = world_analyze(before, fs).f0 if ex.remove.enabled else f0_x
            picture(before, y, f0_b, f0_y, fs, ex, IMG / f"example_{ex.name}.png")
        print("wrote", ex.name)


if __name__ == "__main__":
    main()
