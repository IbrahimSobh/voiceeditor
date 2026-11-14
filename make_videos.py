"""Turn each before/after audio example into a short MP4 for the README.

GitHub shows a play button only for videos uploaded through its web editor, never for
audio, so each example becomes a video: the before and after spectrograms side by side,
the "before" sound with a moving playhead on the left, a short pause, then the "after"
sound with the playhead on the right.

Run after make_examples.py:  .venv\\Scripts\\python make_videos.py
Writes docs/video/<example>.mp4. Needs the imageio-ffmpeg package (it provides ffmpeg).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from make_examples import AUDIO, EXAMPLES, ROOT, before_name
from voiceeditor import plots
from voiceeditor.analysis import formant_ceiling, formant_tracks, world_analyze
from voiceeditor.audio_io import WORK_FS

VIDEO = ROOT / "docs" / "video"
FPS = 12
GAP_S = 0.6               # silence between before and after
HEADER = 64
PANEL_SIZE = (6.2, 3.6)   # inches at 110 dpi -> 682 x 396 pixels per panel
ACCENT, MUTED, INK = (124, 108, 242), (150, 150, 150), (30, 30, 30)


def font(size: int, bold: bool = False):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def panel(x: np.ndarray, fs: int, x_max: float, formant_factor: float, ceiling: float):
    f0 = world_analyze(x, fs).f0
    ft = formant_tracks(x, fs, ceiling * formant_factor)
    return plots.voice_panel(x, fs, f0, np.arange(len(f0)) * 0.005, ft, True, True, x_max, size=PANEL_SIZE)


def make_video(ex) -> Path:
    xb, fs = sf.read(AUDIO / f"{before_name(ex)}.wav")
    xa, _ = sf.read(AUDIO / f"{ex.name}.wav")
    assert fs == WORK_FS
    x_max = max(xb.size, xa.size) / fs
    ceiling = formant_ceiling(world_analyze(xb, fs).f0)
    pb = panel(xb, fs, x_max, 1.0, ceiling)
    pa = panel(xa, fs, x_max, ex.voice.formant, ceiling)

    ph, pw = pb.pixels.shape[:2]
    gap = 24
    width = 2 * pw + gap + (2 * pw + gap) % 2            # H.264 needs even sizes
    height = HEADER + ph + (HEADER + ph) % 2
    base = Image.new("RGB", (width, height), "white")
    base.paste(Image.fromarray(pb.pixels), (0, HEADER))
    base.paste(Image.fromarray(pa.pixels), (pw + gap, HEADER))
    title_font, label_font = font(20, bold=True), font(17)
    ImageDraw.Draw(base).text((16, 8), ex.title, fill=INK, font=title_font)

    def x_of(p, t: float, offset: int) -> int:
        b = p.boxes["spec"]
        frac = b.left + (t - b.xlim[0]) / (b.xlim[1] - b.xlim[0]) * (b.right - b.left)
        return offset + int(frac * pw)

    def top_bottom(p):
        return HEADER + int(p.boxes["wave"].top * ph), HEADER + int(p.boxes["spec"].bottom * ph)

    tb, ta = xb.size / fs, xa.size / fs
    total = tb + GAP_S + ta
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i in range(int(np.ceil(total * FPS))):
            t = i / FPS
            img = base.copy()
            d = ImageDraw.Draw(img)
            playing_before, playing_after = t < tb, t >= tb + GAP_S
            d.text((16, 38), ("▶ " if playing_before else "") + "Before", font=label_font,
                   fill=ACCENT if playing_before else MUTED)
            d.text((pw + gap + 16, 38), ("▶ " if playing_after else "") + "After", font=label_font,
                   fill=ACCENT if playing_after else MUTED)
            if playing_before:
                x, (y0, y1) = x_of(pb, t, 0), top_bottom(pb)
                d.line([(x, y0), (x, y1)], fill=(220, 40, 40), width=2)
            elif playing_after:
                x, (y0, y1) = x_of(pa, t - tb - GAP_S, pw + gap), top_bottom(pa)
                d.line([(x, y0), (x, y1)], fill=(220, 40, 40), width=2)
            img.save(tmp / f"f{i:05d}.png")
        sound = np.concatenate([xb, np.zeros(int(GAP_S * fs)), xa])
        sf.write(tmp / "sound.wav", sound, fs, subtype="PCM_16")
        out = VIDEO / f"{ex.name}.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-framerate", str(FPS), "-i", str(tmp / "f%05d.png"), "-i", str(tmp / "sound.wav"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "slow",
                        "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(out)],
                       check=True)
    return out


def main() -> None:
    """All videos, or only the examples named on the command line."""
    import sys
    VIDEO.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[1:])
    for ex in EXAMPLES:
        if wanted and ex.name not in wanted:
            continue
        out = make_video(ex)
        print(f"wrote {out.name} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
