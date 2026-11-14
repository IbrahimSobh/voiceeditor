"""Build the README / GitHub social-preview banner (docs/img/banner.png, 1280 x 640).

Run after make_screenshots.py:  .venv\\Scripts\\python make_banner.py
The banner is composed from real screenshots of the app, framed as windows, under a title
with glowing waveforms. Fonts: Bahnschrift and Segoe UI (shipped with Windows).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent
IMG = ROOT / "docs" / "img"
W, H = 1280, 640
CYAN, VIOLET, WHITE = (34, 199, 240), (139, 108, 255), (244, 247, 255)
FONTS = Path(r"C:\Windows\Fonts")


def font(name: str, size: int, variation: str | None = None):
    f = ImageFont.truetype(str(FONTS / name), size)
    if variation:
        f.set_variation_by_name(variation)
    return f


def background() -> Image.Image:
    y, x = np.mgrid[0:H, 0:W]
    top, bottom = np.array([7, 14, 30]), np.array([11, 24, 52])
    img = top + (bottom - top) * (y / H)[..., None]
    for cx, cy, color, r in ((260, 120, CYAN, 420), (1040, 120, VIOLET, 420), (640, 640, CYAN, 520)):
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        img += np.array(color) * (np.clip(1 - d / r, 0, 1) ** 2 * 0.16)[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def waves(img: Image.Image) -> Image.Image:
    """Glowing speech-like waveforms on the left (cyan) and right (violet) of the title."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x0, x1, color, seed in ((0, 315, CYAN, 1), (965, W, VIOLET, 2)):
        rng = np.random.default_rng(seed)
        xs = np.arange(x0, x1)
        u = (xs - x0) / (x1 - x0)
        env = np.sin(np.pi * u) ** 1.5 * (0.6 + 0.4 * np.sin(2 * np.pi * u * 3 + seed))
        ys = 110 + 62 * env * np.sin(2 * np.pi * u * 26 + rng.uniform(0, 6)) * np.sin(2 * np.pi * u * 4.3)
        pts = list(zip(xs.tolist(), ys.tolist()))
        d.line(pts, fill=color + (255,), width=3)
    glow = layer.filter(ImageFilter.GaussianBlur(9))
    out = img.convert("RGBA")
    for _ in range(2):
        out = Image.alpha_composite(out, glow)
    return Image.alpha_composite(out, layer)


def window(shot: Path, crop: tuple, width: int, title: str) -> Image.Image:
    """A screenshot crop framed as an app window: dark title bar, three dots, rounded corners."""
    body = Image.open(shot).convert("RGB").crop(crop)
    body = body.resize((width, int(body.height * width / body.width)), Image.LANCZOS)
    bar = 30
    win = Image.new("RGBA", (width, body.height + bar), (17, 26, 46, 255))
    win.paste(body, (0, bar))
    d = ImageDraw.Draw(win)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((14 + i * 20, 10, 26 + i * 20, 22), fill=c)
    d.text((82, 6), title, fill=(220, 228, 245), font=font("segoeuib.ttf", 15))
    mask = Image.new("L", win.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, win.width - 1, win.height - 1), radius=12, fill=255)
    win.putalpha(mask)
    ImageDraw.Draw(win).rounded_rectangle((0, 0, win.width - 1, win.height - 1), radius=12,
                                          outline=(60, 80, 120, 255), width=2)
    return win


def place(canvas: Image.Image, win: Image.Image, xy: tuple, glow=CYAN) -> None:
    shadow = Image.new("RGBA", (win.width + 60, win.height + 60), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((30, 30, 30 + win.width, 30 + win.height), radius=14,
                                             fill=glow + (90,))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    canvas.alpha_composite(shadow, (xy[0] - 30, xy[1] - 30))
    canvas.alpha_composite(win, xy)


def main() -> None:
    img = waves(background())
    d = ImageDraw.Draw(img)
    title_font = font("bahnschrift.ttf", 96, "Bold")
    w1 = d.textlength("VOICE ", font=title_font)
    w2 = d.textlength("EDITOR", font=title_font)
    x = (W - w1 - w2) / 2
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((x + w1, 28), "EDITOR", font=title_font, fill=CYAN + (255,))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(img)
    d.text((x, 28), "VOICE ", font=title_font, fill=WHITE)
    d.text((x + w1, 28), "EDITOR", font=title_font, fill=CYAN)
    sub_font = font("seguisbi.ttf", 30)
    sub = "Classical Speech Processing  ·  No AI"
    d.text(((W - d.textlength(sub, font=sub_font)) / 2, 138), sub, font=sub_font, fill=(200, 212, 240))

    editor = window(IMG / "editor_noise.png", (372, 186, 1368, 560), 760, "Voice Editor · Editor")
    inspect = window(IMG / "editor_inspect.png", (96, 60, 1344, 590), 470, "Inspect a moment")
    learn = window(IMG / "learn_world.png", (584, 1420, 1360, 1838), 400, "Learn · WORLD, decomposed")
    place(img, editor, (40, 214))
    place(img, learn, (850, 196), VIOLET)
    place(img, inspect, (770, 400))
    out = IMG / "banner.png"
    img.convert("RGB").save(out, optimize=True)
    print("wrote", out, img.size)


if __name__ == "__main__":
    main()
