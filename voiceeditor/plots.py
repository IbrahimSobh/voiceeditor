"""Figures: waveform + spectrogram panels (clickable), and the spectrum at a clicked time.

Figures are built with matplotlib's object interface (Figure + its own Agg canvas),
never pyplot: Streamlit can run several reruns at once (e.g. while dragging), and
pyplot's shared global state breaks under that.

Panels are rendered to images so the app can read where the user clicked. Each
panel comes with a layout: where its axes sit inside the image (as fractions)
and which time/frequency ranges they show, so a click converts back to data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import spectrogram

from .analysis import FormantTracks, voiced_at
from .theme import ENVELOPE_COLOR, F0_COLOR, FORMANT_COLOR, MUTED

SPEC_MAX_HZ = 5000
F0_AXIS_MAX_HZ = 500
CURSOR_COLOR = "#ffffff"


@dataclass
class AxesBox:
    left: float      # fractions of image width/height, origin at the image's top-left
    right: float
    top: float
    bottom: float
    xlim: tuple
    ylim: tuple


@dataclass
class PanelImage:
    pixels: np.ndarray               # RGB image
    boxes: dict[str, AxesBox]        # "wave" and "spec"


@dataclass
class Click:
    area: str                        # "wave" or "spec"
    time: float
    freq: float | None               # only for clicks on the spectrogram


def spectrogram_db(x: np.ndarray, fs: int):
    """25 ms Hann window, 90 % overlap: shows both harmonics and formants."""
    nper = int(0.025 * fs)
    f, t, sxx = spectrogram(x, fs, window="hann", nperseg=nper, noverlap=int(nper * 0.9), nfft=1024)
    return f, t, 10 * np.log10(sxx + 1e-12)


def voice_panel(
    x: np.ndarray,
    fs: int,
    f0: np.ndarray,
    f0_times: np.ndarray,
    formants: FormantTracks,
    show_f0: bool,
    show_formants: bool,
    x_max: float,
    cursor: Click | None = None,
    size: tuple = (10, 3.6),
    wave_color: str = MUTED,
) -> PanelImage:
    """Waveform (top) and spectrogram (bottom) sharing one time axis."""
    fig = _new_figure(size)
    ax_w, ax_s = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.06})
    t = np.arange(x.size) / fs
    ax_w.plot(t, x, color=wave_color, linewidth=0.5)
    ax_w.set_ylim(-1, 1)
    ax_w.set_yticks([])
    ax_w.set_ylabel("wave", fontsize=8)

    f, ts, db = spectrogram_db(x, fs)
    keep = f <= SPEC_MAX_HZ
    ax_s.pcolormesh(ts, f[keep], db[keep], shading="auto", cmap="magma", vmin=db.max() - 70, vmax=db.max())
    ax_s.set_ylim(0, SPEC_MAX_HZ)
    ax_s.set_ylabel("Hz", fontsize=8)
    ax_s.set_xlabel("time (s)", fontsize=8)
    ax_s.set_xlim(0, x_max)

    if show_formants:
        mask = voiced_at(formants.times, f0, f0_times)
        for row in formants.values:
            ax_s.scatter(formants.times[mask], row[mask], s=4, color=FORMANT_COLOR, linewidths=0)

    if cursor is not None:
        ax_w.axvline(cursor.time, color="#ff5c5c", linewidth=1)
        ax_s.axvline(cursor.time, color=CURSOR_COLOR, linewidth=1, linestyle="--")
        if cursor.freq is not None:
            ax_s.axhline(cursor.freq, color=CURSOR_COLOR, linewidth=0.8, linestyle=":")
            ax_s.scatter([cursor.time], [cursor.freq], marker="+", s=140, color=CURSOR_COLOR, linewidths=1.5)

    if show_f0:
        ax_f0 = ax_s.twinx()                    # F0 gets its own 0-500 Hz scale on the right
        ax_f0.plot(f0_times, np.where(f0 > 0, f0, np.nan), color=F0_COLOR, linewidth=2)
        ax_f0.set_ylim(0, F0_AXIS_MAX_HZ)
        ax_f0.set_ylabel("F0 Hz", fontsize=8, color=F0_COLOR)
        ax_f0.tick_params(labelsize=7, colors=F0_COLOR)

    for ax in (ax_w, ax_s):
        ax.tick_params(labelsize=7)
    w, h = size                                  # fixed margins in inches, whatever the figure size
    fig.subplots_adjust(left=0.65 / w, right=1 - 0.65 / w, top=1 - 0.1 / h, bottom=0.45 / h)
    return _to_image(fig, {"wave": ax_w, "spec": ax_s})


def _to_image(fig, axes: dict) -> PanelImage:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()   # RGB is enough, smaller to send
    boxes = {}
    for name, ax in axes.items():
        bb = ax.get_window_extent()             # pixels, origin bottom-left
        boxes[name] = AxesBox(bb.x0 / w, bb.x1 / w, 1 - bb.y1 / h, 1 - bb.y0 / h, ax.get_xlim(), ax.get_ylim())
    return PanelImage(pixels, boxes)


def click_to_data(panel: PanelImage, value: dict) -> Click | None:
    """Convert a click (pixels in the displayed image) to time, and frequency on the spectrogram."""
    if not value or not value.get("width") or not value.get("height"):
        return None
    fx, fy = value["x"] / value["width"], value["y"] / value["height"]
    for name, b in panel.boxes.items():
        if b.left <= fx <= b.right and b.top <= fy <= b.bottom:
            t = b.xlim[0] + (fx - b.left) / (b.right - b.left) * (b.xlim[1] - b.xlim[0])
            freq = None
            if name == "spec":
                freq = b.ylim[0] + (b.bottom - fy) / (b.bottom - b.top) * (b.ylim[1] - b.ylim[0])
            return Click(name, float(t), None if freq is None else float(freq))
    return None


def frame_spectrum(x: np.ndarray, fs: int, t: float, win_s: float = 0.04):
    """Magnitude spectrum (dB) of a 40 ms Hann-windowed frame centred on time t."""
    n = int(win_s * fs)
    c = int(t * fs)
    seg = x[max(0, c - n // 2): c + n // 2]
    if seg.size < 16:
        return None, None
    seg = seg * np.hanning(seg.size)
    spec = np.fft.rfft(seg, 2048)
    return np.fft.rfftfreq(2048, 1 / fs), 20 * np.log10(np.abs(spec) + 1e-9)


def spectrum_slice_figure(curves: list[dict], cursor_freq: float | None):
    """Spectrum at the clicked time: harmonics (FFT) and, when available, the WORLD envelope.

    Each curve: {"label", "freqs", "db", "color", optional "env_freqs", "env_db", "formants"}.
    """
    n = len(curves)
    fig = _new_figure((10, 2.5))
    axes = fig.subplots(1, n, sharey=True, squeeze=False)
    top = max(np.max(c["db"]) for c in curves)
    for ax, c in zip(axes[0], curves):          # one small chart per panel, same dB scale
        ax.plot(c["freqs"], c["db"], color=c["color"], linewidth=0.8)
        if c.get("env_db") is not None:
            ax.plot(c["env_freqs"], c["env_db"], color=ENVELOPE_COLOR, linewidth=1.6, alpha=0.8)
        for fval in c.get("formants") or []:
            if np.isfinite(fval):
                ax.axvline(fval, color=FORMANT_COLOR, linewidth=1, linestyle=":")
        if cursor_freq is not None:
            ax.axvline(cursor_freq, color="#ff5c5c", linewidth=1)
        ax.set_title(c["label"], fontsize=8, color=c["color"])
        ax.set_xlim(0, SPEC_MAX_HZ)
        ax.set_ylim(top - 80, top + 5)
        ax.set_xlabel("Hz", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("dB", fontsize=7)
    fig.tight_layout()
    return fig


def _new_figure(size) -> Figure:
    fig = Figure(figsize=size, dpi=110)
    FigureCanvasAgg(fig)          # private canvas: safe when several reruns draw at once
    return fig


def close(fig) -> None:
    """Nothing to release: figures made without pyplot are freed like normal objects."""



def quality_track_chart(tracks: list[dict], hop_s: float, lo: float, hi: float, height: int = 300):
    """Segmental SNR over time as an interactive web chart (Altair, which ships with Streamlit).

    Hover a line for its value, drag or scroll to zoom in time, double-click to reset, and click a
    legend entry to highlight one signal. Frames without speech are left out: each run of speech frames
    is its own line piece, so the lines break at pauses instead of jumping across them.
    Each track: {"label", "db" (per frame), "speech" (bool mask), "color", optional "mean"}.
    """
    import altair as alt
    import pandas as pd

    parts = []
    for tr in tracks:
        name = tr["label"] + (f' · mean {tr["mean"]:.1f} dB' if tr.get("mean") is not None else "")
        speech = np.asarray(tr["speech"], bool)
        run = np.cumsum(np.diff(speech.astype(int), prepend=0) == 1)       # 1, 2, 3 … per speech run
        frames = np.flatnonzero(speech)
        parts.append(pd.DataFrame({"time (s)": frames * hop_s, "frame SNR (dB)": tr["db"][frames],
                                   "signal": name, "piece": run[frames]}))
    data = pd.concat(parts, ignore_index=True)
    names = [p["signal"].iat[0] for p in parts if len(p)]
    colors = [tr["color"] for tr, p in zip(tracks, parts) if len(p)]

    pick = alt.selection_point(fields=["signal"], bind="legend")
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    return (
        alt.Chart(data, height=height)
        .mark_line(strokeWidth=1.6)
        .encode(
            x=alt.X("time (s):Q"),
            y=alt.Y("frame SNR (dB):Q", scale=alt.Scale(domain=[lo - 2, hi + 2])),
            color=alt.Color("signal:N", scale=alt.Scale(domain=names, range=colors),
                            legend=alt.Legend(orient="top", title=None, labelLimit=400, symbolStrokeWidth=3)),
            detail="piece:N",
            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.15)),
            tooltip=["signal:N", alt.Tooltip("time (s):Q", format=".2f"),
                     alt.Tooltip("frame SNR (dB):Q", format=".1f")],
        )
        .add_params(pick, zoom)
    )


def mess_share_chart(rows: list[tuple[str, dict]], bar: int = 36):
    """One bar per signal: what share of its mess is leftover noise and what share is damage to the voice."""
    import altair as alt
    import pandas as pd

    data = pd.DataFrame([{"signal": name, "part": part, "share": m[key],
                          "vs voice (dB)": m["voice_vs_noise_db" if key == "noise_share" else "voice_vs_damage_db"]}
                         for name, m in rows
                         for part, key in (("Leftover noise", "noise_share"), ("Damage to the voice", "damage_share"))])
    return (
        alt.Chart(data, height=alt.Step(bar))     # a fixed height per bar: safe inside tabs that start hidden
        .mark_bar()
        .encode(
            y=alt.Y("signal:N", sort=[n for n, _ in rows], title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%", title="share of the mess")),
            color=alt.Color("part:N", scale=alt.Scale(domain=["Leftover noise", "Damage to the voice"],
                                                      range=["#ff9f2e", "#ff5c8a"]),
                            legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("part:N", sort="descending"),
            tooltip=["signal:N", "part:N", alt.Tooltip("share:Q", format=".0%"),
                     alt.Tooltip("vs voice (dB):Q", format=".1f", title="voice is this many dB stronger")],
        )
    )
