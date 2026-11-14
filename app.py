"""Voice editor - Streamlit app (WORLD engine).

Run:  .venv\\Scripts\\streamlit run app.py

Processing order: input -> add noise -> remove noise -> voice edits (WORLD).

Settings can also come from the web address, for demos and screenshots, e.g.
  ?sample=sample_male_en.wav&preset=More+female&add=White&snr=15&remove=Wiener+filter&click=1.6,1800
  ?view=learn&lesson=3
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from voiceeditor import lessons, plots
from voiceeditor.analysis import WorldParams, formant_ceiling, formant_tracks, summary, world_analyze, world_synthesize
from voiceeditor.audio_io import load_audio, normalize, to_wav_bytes
from voiceeditor.edits import PRESETS, EditSettings, apply_edits
from voiceeditor.metrics import METRICS, SEG_MAX_DB, SEG_MIN_DB, quality, segmental_snr_track
from voiceeditor.noise import (
    NOISE_TYPES, NOVERLAP, NPERSEG, REMOVE_METHODS, AddNoise, RemoveNoise, noise_parts, remove_noise,
)
from voiceeditor.theme import METHOD_COLORS, PANEL_COLORS

SAMPLES = Path(__file__).parent / "samples"
SLIDERS = {  # key: (label, min, max, step)
    "pitch": ("Pitch (F0) ×", 0.5, 2.0, 0.05),
    "pitch_range": ("Pitch range ×", 0.2, 2.0, 0.05),
    "formant": ("Formants ×", 0.8, 1.3, 0.01),
    "speed": ("Speed ×", 0.5, 2.0, 0.05),
}

st.set_page_config(page_title="Voice Editor", page_icon="〰️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:ital,wght@0,700;0,800;1,500&display=swap');
.block-container {padding-top: 2.2rem; padding-bottom: 1.5rem; max-width: 1500px;}
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {padding-top: 1.2rem;}
section[data-testid="stSidebar"] {border-right: 1px solid #1f2a44;}
div[data-testid="stVerticalBlock"] {gap: 0.6rem;}
.ve-brand {display: flex; align-items: center; gap: .9rem; margin-bottom: .9rem;}
.ve-logo {width: 54px; height: 54px; flex: none; border-radius: 12px; display: flex; align-items: center; justify-content: center;
          background: linear-gradient(135deg, #22c7f0 0%, #8b6cff 100%); box-shadow: 0 0 18px rgba(34,199,240,.35);}
.ve-title {font-family: 'Exo 2', sans-serif; font-weight: 800; font-size: 2.1rem !important; letter-spacing: .02em;
           line-height: 1; margin: 0; color: #f4f7ff;}
.ve-title span {background: linear-gradient(90deg, #22c7f0, #6fd8ff); -webkit-background-clip: text;
                background-clip: text; color: transparent;}
.ve-sub {font-family: 'Exo 2', sans-serif; font-style: italic; font-weight: 500; color: #9fb0d0;
         font-size: 1rem !important; margin: .3rem 0 0;}
.ve-sec {font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: #22c7f0;
         margin: .9rem 0 .1rem; font-weight: 700;}
.ve-panel {font-weight: 650; font-size: .92rem; margin: .1rem 0 .2rem; padding: .35rem .75rem;
           border-radius: 8px 8px 0 0; border: 1px solid; border-bottom-width: 3px;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.ve-panel span {font-weight: 400; opacity: .75;}
.ve-chips {display: flex; gap: .35rem; justify-content: flex-end;}
.ve-chip {border: 1px solid #26324d; background: #111a2e; border-radius: 999px; padding: .2rem .65rem;
          font-size: .78rem; white-space: nowrap; color: #dbe4f5;}
.ve-chip i {font-style: normal; color: #8a97b2;}
.ve-chip b {color: #22c7f0;}
audio {height: 36px; width: 100%;}
</style>
""", unsafe_allow_html=True)


# ---------- cached processing (re-runs only when inputs change) ----------

@st.cache_data(show_spinner=False)
def load(data: bytes):
    return load_audio(data)


@st.cache_data(show_spinner=False)
def noisy_parts(data: bytes, add_key: tuple):
    """The clean speech and the noise inside the noisy mix (both scaled the same if the mix was rescaled)."""
    x, fs = load(data)
    return noise_parts(x, fs, AddNoise(*add_key))


@st.cache_data(show_spinner=False)
def noisy_signal(data: bytes, add_key: tuple):
    speech, noise = noisy_parts(data, add_key)
    return speech + noise


@st.cache_data(show_spinner=False)
def cleaned_signal(data: bytes, add_key: tuple, remove_key: tuple):
    x, fs = load(data)
    return remove_noise(noisy_signal(data, add_key), fs, RemoveNoise(*remove_key))


@st.cache_data(show_spinner=False)
def world(x: np.ndarray, fs: int):
    p = world_analyze(x, fs)
    return p.f0, p.sp, p.ap


@st.cache_data(show_spinner=False)
def formants(x: np.ndarray, fs: int, ceiling: float):
    return formant_tracks(x, fs, ceiling)


@st.cache_data(show_spinner=False)
def render(data: bytes, add_key: tuple, remove_key: tuple, voice_key: tuple):
    """Final signal. Without voice edits WORLD is skipped, so noise results are heard exactly."""
    _, fs = load(data)
    x = cleaned_signal(data, add_key, remove_key)
    s = EditSettings(*voice_key)
    f0, sp, ap = world(x, fs)
    if s.is_neutral():
        return x, f0, sp
    edited = apply_edits(WorldParams(f0=f0, sp=sp, ap=ap, fs=fs), s)
    y = normalize(world_synthesize(edited), float(np.max(np.abs(x))) or 0.95)   # keep the input's peak level
    return y, edited.f0, edited.sp


@st.cache_data(show_spinner=False)
def quality_report(data: bytes, add_key: tuple, remove_key: tuple):
    """Metrics and segmental-SNR tracks of the noisy input and of both removal methods, vs the clean speech.

    Both methods use the same slider settings (strength, floor, noise estimate), whichever one is in use.
    """
    _, fs = load(data)
    ref, _ = noisy_parts(data, add_key)
    signals = {"noisy": noisy_signal(data, add_key)}
    for m in REMOVE_METHODS:
        signals[m] = cleaned_signal(data, add_key, replace(RemoveNoise(*remove_key), enabled=True, method=m).key())
    return {name: (quality(ref, sig, fs), *segmental_snr_track(ref, sig)) for name, sig in signals.items()}


def wav(sig: np.ndarray, fs: int) -> bytes:
    """WAV bytes at the signal's true level (st.audio would rescale a NumPy array to full volume)."""
    return to_wav_bytes(sig, fs)


# ---------- state: defaults, presets, settings from the web address ----------

def apply_preset():
    s = PRESETS.get(st.session_state.preset)
    if s is None:          # "Custom" keeps the current slider values
        return
    for k in SLIDERS:
        st.session_state[k] = getattr(s, k)


def mark_custom():
    st.session_state.preset = "Custom"


def apply_query_params():
    """Read settings from the web address once per browser session."""
    q = st.query_params
    if q.get("view", "").lower() == "learn":
        st.session_state.view = "Learn"
    if q.get("lesson", "").isdigit():
        i = int(q["lesson"]) - 1
        if 0 <= i < len(lessons.LESSONS):
            st.session_state.lesson = lessons.LESSONS[i][0]
    if (SAMPLES / q.get("sample", "")).is_file():
        st.session_state.source = "Sample"
        st.session_state.sample_name = q["sample"]
    if q.get("preset") in PRESETS:
        st.session_state.preset = q["preset"]
        apply_preset()
    if q.get("add") in NOISE_TYPES:
        st.session_state.add_on, st.session_state.add_kind = True, q["add"]
    if q.get("snr", "").lstrip("-").isdigit():
        st.session_state.add_snr = int(q["snr"])
    if q.get("remove") in REMOVE_METHODS:
        st.session_state.rem_on, st.session_state.rem_method = True, q["remove"]
    if "click" in q:
        try:
            t, f = (float(v) for v in q["click"].split(","))
            st.session_state.cursor = plots.Click("spec", t, f)
            st.session_state.open_inspect = True
        except ValueError:
            pass
    if q.get("layout") == "stacked":
        st.session_state.layout = "Stacked"


if "_started" not in st.session_state:
    st.session_state._started = True
    for k in SLIDERS:
        st.session_state[k] = 1.0
    st.session_state.update(preset="Neutral (original)", cursor=None, view="Editor", source="File",
                            add_on=False, add_kind="White", add_snr=15, rem_on=False,
                            rem_method=REMOVE_METHODS[0], layout="Side by side")
    apply_query_params()


def section(text: str) -> None:
    st.markdown(f'<p class="ve-sec">{text}</p>', unsafe_allow_html=True)


# ---------- sidebar: input, voice, noise ----------

with st.sidebar:
    section("1 · Input")
    source = st.radio("Source", ["File", "Record", "Sample"], horizontal=True, key="source",
                      label_visibility="collapsed")
    data = None
    if source == "File":
        up = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "mp3"], label_visibility="collapsed")
        data = up.getvalue() if up else None
    elif source == "Record":
        rec = st.audio_input("Record from microphone", label_visibility="collapsed")
        data = rec.getvalue() if rec else None
    else:
        names = [p.name for p in sorted(SAMPLES.glob("*.wav"))]
        if names:
            st.session_state.setdefault("sample_name", names[0])
            name = st.selectbox("Sample", names, key="sample_name", label_visibility="collapsed")
            data = (SAMPLES / name).read_bytes()
        else:
            st.info("No samples in the samples folder yet.")

    section("2 · Voice")
    st.selectbox("Preset", list(PRESETS) + ["Custom"], key="preset", on_change=apply_preset)
    for k, (label, lo, hi, step) in SLIDERS.items():
        st.slider(label, lo, hi, step=step, key=k, on_change=mark_custom)
    st.caption("Engine: WORLD vocoder · Klatt formant engine planned")

    section("3 · Noise")
    with st.expander("Add noise", expanded=st.session_state.add_on):
        add_on = st.checkbox("Add noise", key="add_on")
        add_kind = st.selectbox("Noise type", NOISE_TYPES, key="add_kind", disabled=not add_on,
                                help="White: flat hiss. Pink and brown: more low-frequency rumble. "
                                     "Hum: mains electricity buzz and its harmonics.")
        add_snr = st.slider("SNR (dB) · higher = quieter noise", -10, 40, key="add_snr", disabled=not add_on,
                            help="Signal-to-noise ratio: how much louder speech is than the noise. "
                                 "30 dB: faint. 15 dB: clearly audible. 5 dB: heavy. "
                                 "0 dB: noise as loud as speech.")
    with st.expander("Remove noise", expanded=st.session_state.rem_on):
        rem_on = st.checkbox("Remove noise", key="rem_on")
        rem_method = st.selectbox("Method", REMOVE_METHODS, key="rem_method", disabled=not rem_on,
                                  help="Spectral subtraction subtracts the noise spectrum. "
                                       "Wiener filter scales each frequency by its estimated SNR: "
                                       "smoother, fewer leftover beeps.")
        rem_strength = st.slider("Strength ×", 0.5, 4.0, 1.0, step=0.1, disabled=not rem_on,
                                 help="How many times the estimated noise to remove. "
                                      "Higher removes more noise but damages speech.")
        rem_floor = st.slider("Spectral floor %", 0, 30, 5, disabled=not rem_on,
                              help="Always keep this share of each frequency. "
                                   "Higher sounds more natural but leaves more noise.")
        rem_quiet = st.slider("Noise estimate from quietest frames %", 5, 30, 10, disabled=not rem_on,
                              help="The quietest frames are assumed to be noise only.")
    st.caption("Processing order: add noise → remove noise → voice edits")

voice = EditSettings(**{k: float(st.session_state[k]) for k in SLIDERS})
add_s = AddNoise(add_on, add_kind, float(add_snr))
rem_s = RemoveNoise(rem_on, rem_method, float(rem_strength), rem_floor / 100.0, rem_quiet / 100.0)

# ---------- top bar ----------

title_col, view_col = st.columns([3, 1], vertical_alignment="center")
title_col.markdown(
    '<div class="ve-brand"><div class="ve-logo">'
    '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" stroke="white" stroke-width="2.4" '
    'stroke-linecap="round"><path d="M3 14h2M8 9v10M12 5v18M16 9v10M20 11v6M24 14h1"/></svg></div>'
    '<div><div class="ve-title">VOICE <span>EDITOR</span></div>'
    '<div class="ve-sub">Classical speech processing · No AI</div></div></div>',
    unsafe_allow_html=True)
view = view_col.segmented_control("View", ["Editor", "Learn"], key="view", label_visibility="collapsed") or "Editor"

if view == "Learn":
    lessons.render()
    st.stop()

# ---------- editor ----------

if data is None:
    st.info("Load a file, record your voice, or pick a sample in the sidebar to start.")
    st.stop()

data_id = hashlib.md5(data).hexdigest()
if st.session_state.get("data_id") not in (None, data_id):     # a different recording: forget the old click
    st.session_state.cursor = None
st.session_state.data_id = data_id

try:
    x, fs = load(data)
except Exception:
    st.error("That file couldn't be read as audio. Try a WAV file.")
    st.stop()

with st.spinner("Processing…"):
    f0_o, sp_o, _ = world(x, fs)
    ceiling = formant_ceiling(f0_o)
    panels = [{"name": "original", "title": "Original", "detail": "", "x": x, "f0": f0_o, "sp": sp_o,
               "ft": formants(x, fs, ceiling), "extras": []}]
    if add_s.enabled:
        xn = noisy_signal(data, add_s.key())
        f0_n, sp_n, _ = world(xn, fs)
        panels.append({"name": "noisy", "title": "With added noise",
                       "detail": f"{add_s.kind.lower()}, {add_s.snr_db:.0f} dB SNR",
                       "x": xn, "f0": f0_n, "sp": sp_n, "ft": formants(xn, fs, ceiling),
                       "extras": [("Noise alone", noisy_parts(data, add_s.key())[1])]})
    y, f0_a, sp_a = render(data, add_s.key(), rem_s.key(), voice.key())
    changes = [n for n, v in [("pitch", voice.pitch), ("range", voice.pitch_range),
                              ("formants", voice.formant), ("speed", voice.speed)] if v != 1.0]
    extras = []
    if rem_s.enabled:     # what the noise reduction took away (ideally only noise)
        changes.insert(0, f"denoised ({rem_s.method.lower()})")
        removed = noisy_signal(data, add_s.key()) - cleaned_signal(data, add_s.key(), rem_s.key())
        extras.append(("Removed part", removed))
    elif add_s.enabled:
        changes.insert(0, "noise added")
    panels.append({"name": "after", "title": "After", "detail": " + ".join(changes) if changes else "no edits",
                   "x": y, "f0": f0_a, "sp": sp_a, "ft": formants(y, fs, ceiling * voice.formant),
                   "extras": extras})


def panel_time(cursor: plots.Click, name: str) -> float:
    """A click on "after" is on the stretched clock; convert so every panel marks the same sound."""
    on_after = cursor.area.startswith("after")
    if name == "after":
        return cursor.time if on_after else cursor.time / voice.speed
    return cursor.time * voice.speed if on_after else cursor.time


@st.dialog("Inspect a moment", width="large")
def inspect(cursor: plots.Click) -> None:
    where = f"t = {cursor.time:.3f} s" + (f" · f = {cursor.freq:.0f} Hz" if cursor.freq is not None else "")
    st.markdown(f"**{where}**")
    if voice.speed != 1.0:
        st.caption(f"Speed is ×{voice.speed:.2f}, so the same sound sits at a different time in the "
                   "After panel. Each row uses that panel's own time.")
    rows, curves, snippets = [], [], []
    for p in panels:
        t = panel_time(cursor, p["name"])
        i = int(np.clip(t * fs, 0, p["x"].size - 1))
        k = int(np.clip(round(t / 0.005), 0, len(p["f0"]) - 1))
        f0 = p["f0"][k]
        j = int(np.clip(np.searchsorted(p["ft"].times, t), 0, p["ft"].times.size - 1))
        fm = p["ft"].values[:, j] if f0 > 0 else np.full(3, np.nan)
        level = None
        if cursor.freq is not None:
            f, ts, db = plots.spectrogram_db(p["x"], fs)
            level = db[np.argmin(np.abs(f - cursor.freq)), np.argmin(np.abs(ts - t))]
        rows.append({
            "Panel": p["title"], "Time (s)": f"{t:.3f}", "Amplitude": f"{p['x'][i]:+.3f}",
            "Level at clicked Hz (dB)": f"{level:.1f}" if level is not None else "–",
            "F0 (Hz)": f"{f0:.0f}" if f0 > 0 else "unvoiced",
            "F1 (Hz)": f"{fm[0]:.0f}" if np.isfinite(fm[0]) else "–",
            "F2 (Hz)": f"{fm[1]:.0f}" if np.isfinite(fm[1]) else "–",
            "F3 (Hz)": f"{fm[2]:.0f}" if np.isfinite(fm[2]) else "–",
        })
        freqs, sdb = plots.frame_spectrum(p["x"], fs, t)
        if freqs is not None:
            env = 10 * np.log10(np.maximum(p["sp"][k], 1e-16))
            env = env - env.max() + sdb.max()       # align envelope with the harmonics for display
            curves.append({"label": p["title"], "freqs": freqs, "db": sdb, "color": PANEL_COLORS[p["name"]],
                           "env_freqs": np.linspace(0, fs / 2, env.size),
                           "env_db": env if f0 > 0 else None, "formants": list(fm)})
        half = int(0.25 * fs)
        snippets.append((p["title"], p["x"][max(0, i - half): i + half]))
    st.dataframe(rows, hide_index=True, width="stretch")
    if curves:
        st.pyplot(plots.spectrum_slice_figure(curves, cursor.freq))
        st.caption("Coloured: harmonics (FFT of a 40 ms frame) · white: WORLD spectral envelope · "
                   "green dotted: formants F1–F3 · red: clicked frequency")
    for col, (label, seg) in zip(st.columns(len(snippets)), snippets):
        col.caption(f"{label} · 0.5 s")
        if seg.size:
            col.audio(wav(seg, fs), format="audio/wav")


# Toolbar: overlays, layout, marker actions, summary chips
original, after = panels[0], panels[-1]
sb = summary(WorldParams(f0=original["f0"], sp=sp_o[:1], ap=sp_o[:1], fs=fs), original["ft"], x.size)
sa = summary(WorldParams(f0=after["f0"], sp=sp_o[:1], ap=sp_o[:1], fs=fs), after["ft"], y.size)


def chip(label: str, key: str, unit: str, fmt: str = "{:.0f}") -> str:
    a, b = sa[key], sb[key]
    if a is None or b is None:
        return f'<span class="ve-chip"><i>{label}</i> –</span>'
    return (f'<span class="ve-chip"><i>{label}</i> {fmt.format(b)} → <b>{fmt.format(a)}</b> {unit}</span>')


cursor = st.session_state.cursor
bar = st.columns([0.9, 0.9, 1.3, 3.6], vertical_alignment="center")
show_f0 = bar[0].toggle("F0 curve", value=True, help="Pitch track (orange), its scale on the right")
show_fm = bar[1].toggle("Formants", value=True, help="F1–F3 tracks from Praat (green dots)")
layout = bar[2].segmented_control("Layout", ["Side by side", "Stacked"], key="layout",
                                  label_visibility="collapsed") or "Side by side"
bar[3].markdown('<div class="ve-chips">' + chip("F0", "mean_f0", "Hz") + chip("range", "f0_range", "Hz")
                + chip("F1", "median_f1", "Hz") + chip("duration", "duration", "s", "{:.2f}") + "</div>",
                unsafe_allow_html=True)

hint, act1, act2 = st.columns([4, 1, 1], vertical_alignment="center")
if cursor is None:
    hint.caption("Click any waveform or spectrogram to inspect that moment.")
else:
    hint.caption(f"Marker at {cursor.time:.3f} s" + (f", {cursor.freq:.0f} Hz" if cursor.freq else ""))
    if act1.button("Inspect", icon=":material/search:", width="stretch"):
        st.session_state.open_inspect = True
    if act2.button("Clear", icon=":material/close:", width="stretch"):
        st.session_state.cursor = None
        st.rerun()

# Panels
x_max = max(p["x"].size for p in panels) / fs
side = layout == "Side by side"
size = (6.2, 3.9) if side and len(panels) == 2 else (5.2, 4.2) if side else (10, 2.7)
cols = st.columns(len(panels), gap="medium") if side else [st.container() for _ in panels]
for col, p in zip(cols, panels):
    with col:
        detail = f' <span>· {p["detail"]}</span>' if p["detail"] else ""
        full = p["title"] + (f' · {p["detail"]}' if p["detail"] else "")
        c = PANEL_COLORS[p["name"]]
        st.markdown(f'<p class="ve-panel" title="{full}" style="color:{c}; border-color:{c}55; '
                    f'border-bottom-color:{c}; background:{c}1f">{p["title"]}{detail}</p>', unsafe_allow_html=True)
        players = [("Listen", p["x"])] + p["extras"]
        for pc, (label, sig) in zip(st.columns(len(players)), players):
            pc.caption(label)       # every player labelled, so panels side by side line up
            pc.audio(wav(sig, fs), format="audio/wav")
        times = np.arange(len(p["f0"])) * 0.005
        mark = plots.Click(cursor.area, panel_time(cursor, p["name"]), cursor.freq) if cursor else None
        img = plots.voice_panel(p["x"], fs, p["f0"], times, p["ft"], show_f0, show_fm, x_max, mark, size=size,
                                wave_color=PANEL_COLORS[p["name"]])
        value = streamlit_image_coordinates(img.pixels, width="stretch", key=f"img_{p['name']}",
                                            png_compression_level=6, cursor="crosshair")
        if value and value.get("unix_time") != st.session_state.get(f"seen_{p['name']}"):
            st.session_state[f"seen_{p['name']}"] = value.get("unix_time")
            click = plots.click_to_data(img, value)
            if click is not None:
                click.area = f"{p['name']}:{click.area}" if p["name"] == "after" else click.area
                st.session_state.cursor = click
                st.session_state.open_inspect = True
                st.rerun()



# ---------- quality metrics: every noise method, in numbers ----------

SHORT = {"seg_snr": "Seg. SNR"}     # card labels must fit narrow columns


def fmt_db(v: float) -> str:
    return "–" if not np.isfinite(v) else "> 99" if v > 99 else f"{v:.1f}"


section("Quality metrics · compared with the clean original")
if not add_s.enabled:
    st.caption("Metrics need the clean speech as a reference. Turn on **Add noise**: the app then knows exactly "
               "what is speech and what is noise, and can score the noisy input and both removal methods. "
               "See the lesson *Measuring quality* in Learn.")
else:
    report = quality_report(data, add_s.key(), rem_s.key())
    shown = rem_s.method if rem_s.enabled else "noisy"
    cards = st.columns(len(METRICS))
    for col, (k, (label, unit, higher, help_text)) in zip(cards, METRICS.items()):
        value, base = report[shown][0][k], report["noisy"][0][k]
        delta = None if shown == "noisy" else f"{value - base:+.1f}"
        col.metric(f"{SHORT.get(k, label)} ({unit}) {'↑' if higher else '↓'}", fmt_db(value), delta,
                   delta_color="normal" if higher else "inverse", help=help_text, border=True)
    st.caption(("After " + rem_s.method + " · small numbers: change from the noisy input"
                if rem_s.enabled else "With added noise, before any removal")
               + (" · measured before the voice edits: after WORLD rebuilds the voice, the waveform no longer "
                  "lines up sample by sample with the original" if not voice.is_neutral() else ""))

    names = {"noisy": "With added noise"} | {m: m for m in REMOVE_METHODS}
    rows = []
    for name, (q, _, _) in report.items():
        row = {"Signal": names[name] + ("  ◀ in use" if rem_s.enabled and name == rem_s.method else "")}
        row |= {f"{lab} ({u}) {'↑' if hi else '↓'}": fmt_db(q[k]) for k, (lab, u, hi, _) in METRICS.items()}
        row["SI-SDR gain (dB)"] = "–" if name == "noisy" else f"{q['si_sdr'] - report['noisy'][0]['si_sdr']:+.1f}"
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")
    colors = {"noisy": PANEL_COLORS["noisy"]} | METHOD_COLORS
    tracks = [{"label": names[n], "db": db, "speech": sp, "color": colors[n], "mean": q["seg_snr"]}
              for n, (q, db, sp) in report.items()]
    st.pyplot(plots.quality_track_figure(tracks, (NPERSEG - NOVERLAP) / fs, SEG_MIN_DB, SEG_MAX_DB))
    st.caption("Frame-by-frame SNR over the speech (32 ms frames, clamped to −10…35 dB); its mean is the "
               "segmental SNR. Both methods use the same Strength, floor and noise-estimate sliders. "
               "↑ higher is better, ↓ lower is better. Numbers don't replace listening: musical noise can "
               "score well and still sound bad.")

if st.session_state.pop("open_inspect", False) and st.session_state.cursor is not None:
    inspect(st.session_state.cursor)

st.sidebar.download_button("Export result WAV", to_wav_bytes(y, fs), file_name="edited.wav",
                           mime="audio/wav", icon=":material/download:", width="stretch")
