"""Learn tab: lessons from the basics of sound to every algorithm used in this app.

Each lesson is a function that draws its text, formulas and small interactive
demos with Streamlit. Demos use the bundled sample voice or tiny synthesizers
written here, so every claim can be seen and heard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import freqz, lfilter, resample_poly, welch

from .analysis import world_analyze, world_synthesize
from .audio_io import WORK_FS, load_audio, to_wav_bytes
from .edits import shift_pitch, warp_formants
from .metrics import (
    METRICS, SEG_MAX_DB, SEG_MIN_DB, in_words, mess_breakdown, quality, segmental_snr_track, verdict,
)
from .noise import (
    NOISE_TYPES, NOVERLAP, NPERSEG, REMOVE_METHODS, AddNoise, RemoveNoise, make_noise, noise_parts,
    remove_noise, speech_power, split_result,
)
from .plots import quality_track_chart
from .theme import METHOD_COLORS, MUTED, PANEL_COLORS

FS = WORK_FS
SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_male_en.wav"

# Average formants of adult male speakers of American English (Peterson and Barney, 1952)
VOWELS = {
    "i (beet)": (270, 2290, 3010),
    "ɪ (bit)": (390, 1990, 2550),
    "ɛ (bet)": (530, 1840, 2480),
    "æ (bat)": (660, 1720, 2410),
    "ɑ (father)": (730, 1090, 2440),
    "ɔ (bought)": (570, 840, 2410),
    "ʊ (book)": (440, 1020, 2240),
    "u (boot)": (300, 870, 2240),
    "ʌ (but)": (640, 1190, 2390),
}
BANDWIDTHS = (90, 110, 170)


# ---------- small helpers ----------

def _fig(w: float = 10, h: float = 2.6) -> Figure:
    f = Figure(figsize=(w, h), dpi=100)
    FigureCanvasAgg(f)
    return f


def _show(f: Figure) -> None:
    f.tight_layout()
    st.pyplot(f)


def _play(x: np.ndarray, label: str | None = None, fs: int = FS, where=st) -> None:
    if label:
        where.caption(label)
    where.audio(to_wav_bytes(np.asarray(x, dtype=np.float64), fs), format="audio/wav")


def _fade(x: np.ndarray, ms: float = 20) -> np.ndarray:
    n = min(int(ms / 1000 * FS), x.size // 2)
    if n > 0:
        ramp = np.linspace(0, 1, n)
        x = x.copy()
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _peak(x: np.ndarray, p: float = 0.8) -> np.ndarray:
    m = np.max(np.abs(x))
    return x * (p / m) if m > 0 else x


@st.cache_data(show_spinner=False)
def _sample():
    return load_audio(SAMPLE.read_bytes())


@st.cache_data(show_spinner=False)
def _sample_world():
    x, fs = _sample()
    p = world_analyze(x, fs)
    return p.f0, p.sp, p.ap


def resonator(x: np.ndarray, freq: float, bw: float, fs: int = FS) -> np.ndarray:
    """Klatt's second-order resonator: one formant at `freq` Hz with bandwidth `bw` Hz."""
    t = 1.0 / fs
    c = -np.exp(-2 * np.pi * bw * t)
    b = 2 * np.exp(-np.pi * bw * t) * np.cos(2 * np.pi * freq * t)
    a = 1 - b - c
    return lfilter([a], [1, -b, -c], x)


def resonator_response(formants, bws, fs: int = FS):
    """Frequency response (dB) of resonators in cascade."""
    w = np.linspace(0, np.pi, 2048)
    h = np.ones_like(w, dtype=complex)
    for f, bw in zip(formants, bws):
        t = 1.0 / fs
        c = -np.exp(-2 * np.pi * bw * t)
        b = 2 * np.exp(-np.pi * bw * t) * np.cos(2 * np.pi * f * t)
        _, hi = freqz([1 - b - c], [1, -b, -c], worN=w)
        h *= hi
    return w * fs / (2 * np.pi), 20 * np.log10(np.abs(h) + 1e-12)


def glottal_source(f0: float, dur: float, fs: int = FS) -> np.ndarray:
    """Impulse train at F0, low-passed twice so its spectrum falls about 12 dB per octave."""
    n = int(dur * fs)
    src = np.zeros(n)
    src[np.arange(0, n, fs / f0).astype(int)] = 1.0
    a = np.exp(-2 * np.pi * 100 / fs)
    for _ in range(2):
        src = lfilter([1 - a], [1, -a], src)
    return src


def synth_vowel(f0: float, formants, bws=BANDWIDTHS, dur: float = 0.8) -> np.ndarray:
    """Source -> formant resonators in cascade -> lip radiation (first difference)."""
    y = glottal_source(f0, dur)
    for f, bw in zip(formants, bws):
        y = resonator(y, f, bw)
    y = np.diff(y, prepend=0.0)
    return _peak(_fade(y))


# ---------- lessons ----------

def lesson_sound():
    st.markdown("""
Sound is a **pressure wave**: a vibrating object (a loudspeaker cone, the vocal folds)
pushes on the air, creating regions of slightly higher and lower pressure that travel
outward at about **343 m/s** (in air at 20 °C). A microphone turns these pressure changes
into a voltage, and the computer stores that voltage as numbers. The plot of those numbers
over time is the **waveform**.

The simplest sound is a **sine wave**:
""")
    st.latex(r"x(t) = A \sin(2\pi f t + \varphi)")
    st.markdown("""
- **A, amplitude**: the size of the pressure change. Bigger amplitude sounds louder.
- **f, frequency** in hertz (Hz): how many cycles happen per second. Higher frequency sounds
  higher in pitch. Human hearing covers roughly **20 Hz to 20 kHz**, shrinking at the top with age.
- **T = 1/f, period**: the duration of one cycle. A 100 Hz tone repeats every 10 ms.
- **φ, phase**: where in its cycle the wave starts. Hearing is mostly insensitive to phase
  for steady sounds.

**Loudness in decibels (dB).** Hearing responds to *ratios*, not differences, so levels are
measured on a logarithmic scale:
""")
    st.latex(r"L = 20\log_{10}\frac{A}{A_\text{ref}} \;\text{dB (amplitudes)}"
             r"\qquad L = 10\log_{10}\frac{P}{P_\text{ref}} \;\text{dB (powers)}")
    st.markdown("""
Doubling the amplitude adds about 6 dB; ten times the power adds 10 dB. Power is
proportional to amplitude squared, which is why the two formulas give the same answer.
""")
    st.markdown("##### Try it")
    c1, c2 = st.columns(2)
    f = c1.slider("Frequency (Hz)", 50, 4000, 220, key="L1_f")
    db = c2.slider("Level (dB below full scale)", -40, 0, -6, key="L1_db")
    a = 10 ** (db / 20)
    t = np.arange(int(0.6 * FS)) / FS
    x = a * np.sin(2 * np.pi * f * t)
    fig = _fig(10, 2.2)
    ax = fig.add_subplot()
    show_n = int(min(5 / f, 0.05) * FS)
    ax.plot(t[:show_n] * 1000, x[:show_n])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"{f} Hz: period {1000 / f:.2f} ms, amplitude {a:.3f}", fontsize=9)
    _show(fig)
    _play(_fade(x), "Listen (lower your volume first)")


def lesson_digital():
    st.markdown("""
A computer can't store a continuous wave, so it measures it at regular instants. Two numbers
define digital audio:

**Sample rate (fs)**: measurements per second. This app works at **16,000 Hz**: each second of
speech becomes 16,000 numbers.

**Nyquist limit.** A sampled signal can only represent frequencies below **fs / 2**, the
Nyquist frequency (8 kHz here). This is enough for speech: telephone audio keeps only
300–3400 Hz and stays understandable, and the formants that identify vowels sit below about
4 kHz. Consonants like /s/ have energy above 8 kHz, which is lost at 16 kHz; that loss is a
small price for faster analysis.

**Aliasing.** A frequency above fs / 2 doesn't disappear when sampled: it shows up at a false,
lower frequency. A tone at frequency f appears at |f − k·fs| for the nearest integer k. That's
why audio is low-pass filtered *before* sampling or resampling (`scipy.signal.resample_poly`,
used in this app, does this automatically).

**Bit depth (quantization)**: how precisely each sample is stored. With b bits there are 2ᵇ
levels; every bit adds about 6 dB of dynamic range. 16-bit audio has 65,536 levels and
about 96 dB of range. Too few bits adds **quantization noise**: a hiss or crackle.
""")
    st.markdown("##### Try it: aliasing (sampling at 16 kHz)")
    f = st.slider("Tone frequency (Hz)", 500, 15500, 3000, step=250, key="L2_f")
    alias = abs(f - round(f / FS) * FS)
    st.write(f"A {f} Hz tone sampled at 16 kHz is heard at **{alias:.0f} Hz**"
             + (" (no aliasing, below 8 kHz)." if f < FS / 2 else " (aliased!)."))
    t = np.arange(int(0.6 * FS)) / FS
    _play(_fade(0.4 * np.sin(2 * np.pi * f * t)), "Samples of the tone, played at 16 kHz")

    st.markdown("##### Try it: bit depth")
    bits = st.slider("Bits per sample", 1, 16, 4, key="L2_bits")
    x, _ = _sample()
    levels = 2 ** bits
    q = np.round((x + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
    c1, c2 = st.columns(2)
    _play(x, "Original (16-bit)", where=c1)
    _play(q, f"Quantized to {bits} bits ({levels} levels)", where=c2)


def lesson_voice():
    st.markdown("""
Speech is made in three stages:

1. **Power: the lungs** push air upward.
2. **Source: the larynx.** The air passes the **vocal folds** (vocal cords). For **voiced**
   sounds (vowels, /m/, /z/) the folds open and close rapidly, chopping the airflow into
   puffs. The number of puffs per second is the **fundamental frequency, F0**, heard as pitch.
   For **unvoiced** sounds (/s/, /f/, /h/) the folds stay apart and the sound comes from
   turbulent air: noise, without a pitch.
3. **Filter: the vocal tract**, the tube of throat, mouth and nose above the larynx
   (about 17 cm long in adult men, 14 cm in adult women). Like any tube it **resonates**:
   it boosts some frequencies and weakens others. The resonance peaks are the **formants**.
   Moving the tongue, jaw and lips changes the tube's shape, moves the formants, and turns
   one vowel into another.

This is the **source–filter model** (Fant, 1960): speech = source × filter, where the source
decides pitch and voicing, and the filter decides which sound is spoken. In the frequency
domain:
""")
    st.latex(r"S(f) = G(f)\cdot H(f)\cdot R(f)")
    st.markdown("""
G(f) is the glottal source spectrum (harmonics at F0, 2·F0, 3·F0, … falling about 12 dB per
octave), H(f) the vocal-tract filter (the formant peaks), and R(f) the radiation at the
lips (rising about 6 dB per octave). The two are *independent*: you can sing many notes on
one vowel, or many vowels on one note. **Every edit in this app relies on that
independence**: WORLD separates source from filter so each can be changed alone.
""")
    st.markdown("##### Try it: build a vowel from a source and a filter")
    c1, c2 = st.columns(2)
    f0 = c1.slider("Source F0 (Hz)", 70, 300, 120, key="L3_f0")
    vowel = c2.selectbox("Filter (vowel)", list(VOWELS), index=4, key="L3_v")
    formants = VOWELS[vowel]
    src = glottal_source(f0, 0.8)
    y = synth_vowel(f0, formants)
    fig = _fig(10, 2.6)
    axes = fig.subplots(1, 3, sharey=False)
    for ax, sig, title in [(axes[0], src, "source: harmonics of F0"), (axes[2], y, "output: speech")]:
        spec = np.fft.rfft(sig[: 4096] * np.hanning(4096))
        fr = np.fft.rfftfreq(4096, 1 / FS)
        sdb = 20 * np.log10(np.abs(spec) + 1e-9)
        ax.plot(fr, sdb, linewidth=0.7)
        ax.set_ylim(sdb.max() - 70, sdb.max() + 5)      # top 70 dB: harmonics, not the dips between them
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 4000)
    fr, h = resonator_response(formants, BANDWIDTHS)
    axes[1].plot(fr, h, color="#2ee6a6")
    for f in formants:
        axes[1].axvline(f, color="#2ee6a6", linestyle=":", linewidth=0.8)
    axes[1].set_title(f"filter: formants {formants[0]}, {formants[1]}, {formants[2]} Hz", fontsize=9)
    axes[1].set_xlim(0, 4000)
    for ax in axes:
        ax.set_xlabel("Hz", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("dB")
    _show(fig)
    c1, c2 = st.columns(2)
    _play(_peak(_fade(np.diff(src, prepend=0.0))), "Source alone (a buzz)", where=c1)
    _play(y, f"Source through the {vowel} filter", where=c2)
    st.caption("This tiny synthesizer is a simplified Klatt formant synthesizer: an impulse train "
               "through three resonators in series.")


def lesson_spectrum():
    st.markdown("""
**Fourier's idea**: any signal can be written as a sum of sine waves of different
frequencies, amplitudes and phases. The **spectrum** shows how much of each frequency a
signal contains: frequency on the horizontal axis, strength (usually in dB) on the vertical.

**Harmonics.** A periodic signal with period T contains only frequencies at whole multiples of
F0 = 1/T: F0, 2·F0, 3·F0, … A voiced sound at F0 = 120 Hz has harmonics at 120, 240, 360 Hz
and so on. The vocal tract doesn't create new frequencies; it changes the *strength* of these
harmonics, and the formants are where harmonics come out strongest.

**DFT and FFT.** For N digital samples, the **Discrete Fourier Transform** computes N complex
numbers, one per frequency bin spaced fs/N apart:
""")
    st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}")
    st.markdown("""
|X[k]| is the amplitude of bin k, and its angle is the phase. The **Fast Fourier Transform
(FFT)** is an algorithm that computes the same result in N·log N steps instead of N², which
makes spectral analysis practical (`numpy.fft` in this app).

**Windowing.** The DFT treats the N samples as one repeating cycle. Cutting a signal abruptly
creates false frequencies (**spectral leakage**), so each piece is first multiplied by a
smooth **window** that fades to zero at both ends. This app uses the **Hann window**.
""")
    st.markdown("##### Try it: add harmonics one by one")
    c1, c2 = st.columns(2)
    f0 = c1.slider("F0 (Hz)", 80, 400, 150, key="L4_f0")
    n_h = c2.slider("Number of harmonics", 1, 30, 1, key="L4_n")
    t = np.arange(int(0.8 * FS)) / FS
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, n_h + 1) if k * f0 < FS / 2)
    x = _peak(x)
    fig = _fig(10, 2.4)
    a1, a2 = fig.subplots(1, 2)
    n = int(3 / f0 * FS)
    a1.plot(t[:n] * 1000, x[:n])
    a1.set_xlabel("time (ms)", fontsize=8)
    a1.set_title("waveform: 3 periods", fontsize=9)
    spec = np.fft.rfft(x[:4096] * np.hanning(4096))
    fr = np.fft.rfftfreq(4096, 1 / FS)
    a2.plot(fr, 20 * np.log10(np.abs(spec) + 1e-9), linewidth=0.8)
    a2.set_xlim(0, 5000)
    a2.set_ylim(np.max(20 * np.log10(np.abs(spec) + 1e-9)) - 60, None)
    a2.set_xlabel("frequency (Hz)", fontsize=8)
    a2.set_title("spectrum: peaks at k × F0 (amplitudes 1/k)", fontsize=9)
    _show(fig)
    _play(_fade(x))
    st.caption("With amplitudes 1/k the sum approaches a sawtooth wave: a buzzy sound similar "
               "to the glottal source.")


def lesson_spectrogram():
    st.markdown("""
A single spectrum describes a stretch of sound as if nothing changed inside it, but speech
changes every few milliseconds. The **spectrogram** solves this with the **Short-Time Fourier
Transform (STFT)**:

1. Cut the signal into short overlapping **frames** (here 25 ms, overlapping by 90 %).
2. Multiply each frame by a window (Hann).
3. Take the FFT of each frame.
4. Draw the spectra side by side: **time** across, **frequency** up, **level** as colour
   (bright = strong).

**Reading a spectrogram of speech**

- **Vowels**: dark-to-bright horizontal bands. The bright bands are the formants.
- **Voicing**: fine horizontal lines (harmonics) or vertical striations (one per glottal pulse).
- **Fricatives** (/s/, /ʃ/, /f/): fuzzy noise, often high up. /s/ is strongest above about 4 kHz.
- **Stops** (/p/, /t/, /k/): a silent gap, then a short burst of noise.
- **Silence**: dark.

**Time–frequency trade-off.** A window of length L can only separate frequencies about 1/L
apart.
- A **long window** (30 ms or more) gives a **narrowband** spectrogram: individual harmonics
  are resolved, but quick changes blur.
- A **short window** (around 5 ms) gives a **wideband** spectrogram: harmonics merge into
  formant bands and each glottal pulse shows as a vertical line, but frequency detail is lost.

This app uses 25 ms, a middle ground that shows both harmonics and formants.
""")
    st.markdown("##### Try it: change the window length")
    win_ms = st.slider("Window length (ms)", 3, 64, 25, key="L5_w")
    x, fs = _sample()
    seg = x[: int(2.0 * fs)]
    nper = max(16, int(win_ms / 1000 * fs))
    from scipy.signal import spectrogram as spg
    f, t, sxx = spg(seg, fs, window="hann", nperseg=nper, noverlap=int(nper * 0.9),
                    nfft=max(1024, 1 << (nper - 1).bit_length()))
    db = 10 * np.log10(sxx + 1e-12)
    fig = _fig(10, 3)
    ax = fig.add_subplot()
    keep = f <= 5000
    ax.pcolormesh(t, f[keep], db[keep], shading="auto", cmap="magma", vmin=db.max() - 70, vmax=db.max())
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    kind = "narrowband: harmonics visible" if win_ms >= 20 else "wideband: formants and pulses"
    ax.set_title(f"{win_ms} ms window, frequency resolution about {1000 / win_ms:.0f} Hz ({kind})", fontsize=9)
    _show(fig)


def lesson_pitch():
    st.markdown("""
**F0 (fundamental frequency)** is the vibration rate of the vocal folds; **pitch** is how high
we perceive it. Typical speaking F0 is roughly **85–180 Hz for adult men**, **165–255 Hz for
adult women**, and higher for children. Within one sentence F0 rises and falls: that melody
is **intonation**, and it carries questions, emphasis and emotion. Unvoiced sounds and silence
have no F0; trackers mark them as 0.

**The classic method: autocorrelation.** A voiced signal repeats itself every period T0. Compare
a frame with a copy of itself shifted by a lag τ:
""")
    st.latex(r"r(\tau) = \sum_n x[n]\,x[n+\tau]")
    st.markdown("""
r(τ) is large when the shifted copy lines up with the original, which first happens at
τ = T0. So the lag of the highest peak (within the plausible range 60–500 Hz) gives
F0 = fs / τ. Real trackers add safeguards against **octave errors** (picking 2·T0 or T0/2).

**Harvest**, the tracker used in this app (part of WORLD, Morise 2017), works differently and
is more robust:

1. It filters the signal with a set of band-pass filters, one tuned to each candidate F0.
2. When a filter is centred near the true F0, its output is close to a pure sine. The
   intervals between that sine's zero crossings and peaks give F0 estimates.
3. Candidates that agree with each other score as reliable.
4. Reliable candidates are connected into a smooth contour over time and refined.

Harvest is slower than simple methods but rarely misses voiced frames, which matters because
WORLD rebuilds speech using this contour. Pitch tracking fails more often on noisy, breathy,
very low (creaky) or very high voices.
""")
    st.markdown("##### Try it: autocorrelation on the sample voice")
    x, fs = _sample()
    f0_w, _, _ = _sample_world()
    voiced_t = np.flatnonzero(f0_w > 0) * 0.005
    t0 = st.slider("Frame position (s)", 0.0, float(x.size / fs - 0.05), float(voiced_t[len(voiced_t) // 3]),
                   step=0.005, key="L6_t")
    n = int(0.04 * fs)
    i = int(t0 * fs)
    frame = x[i: i + n] - np.mean(x[i: i + n])
    r = np.correlate(frame, frame, mode="full")[n - 1:]
    r = r / (r[0] + 1e-12)
    lo, hi = int(fs / 500), int(fs / 60)
    lag = lo + int(np.argmax(r[lo:hi]))
    harvest = f0_w[min(int(round(t0 / 0.005)), len(f0_w) - 1)]
    fig = _fig(10, 2.4)
    a1, a2 = fig.subplots(1, 2)
    a1.plot(np.arange(n) / fs * 1000, frame)
    a1.set_title("40 ms frame", fontsize=9)
    a1.set_xlabel("ms", fontsize=8)
    a2.plot(np.arange(r.size) / fs * 1000, r)
    a2.axvspan(lo / fs * 1000, hi / fs * 1000, color="#ff9f2e", alpha=0.1, label="search 60–500 Hz")
    a2.axvline(lag / fs * 1000, color="#ff5c5c", label=f"peak: {lag / fs * 1000:.2f} ms")
    a2.legend(fontsize=7)
    a2.set_title("autocorrelation r(τ)", fontsize=9)
    a2.set_xlabel("lag τ (ms)", fontsize=8)
    _show(fig)
    st.write(f"Autocorrelation estimate: **{fs / lag:.0f} Hz**. Harvest at this time: "
             + (f"**{harvest:.0f} Hz**." if harvest > 0 else "**unvoiced** (no F0)."))


def lesson_formants():
    st.markdown("""
**Formants** are the resonance frequencies of the vocal tract, numbered from the bottom:
F1, F2, F3, … They identify vowels:

- **F1** goes **up** when the mouth opens and the tongue drops: low in /i/ (beet), high in /ɑ/ (father).
- **F2** goes **up** when the tongue moves **forward**: high in /i/, low in /u/ (boot) and /ɔ/.
  Rounding the lips lowers it further.
- **F3** helps with r-colouring and voice quality; it falls in American /ɝ/ (bird).

**Why the tract resonates where it does.** A uniform tube closed at one end (the glottis) and
open at the other (the lips) resonates at odd multiples of a quarter wavelength:
""")
    st.latex(r"F_n = \frac{(2n-1)\,c}{4L}")
    st.markdown("""
With c = 350 m/s (warm, humid air in the mouth) and L = 17.5 cm this gives 500, 1500 and
2500 Hz, the formants of the neutral vowel schwa. **A shorter tract has proportionally higher
formants**: that's why women's formants are about 15–20 % higher than men's, children's
higher still, and why this app's *Formants ×* slider is the main "vocal-tract size" control.

**How Praat measures formants (used for the green dots).** Praat's Burg method uses
**Linear Prediction (LPC)**:

1. Resample the audio so its top frequency equals the **formant ceiling** (5000 Hz for
   typical male voices, 5500 Hz for female voices).
2. Apply pre-emphasis (boost the highs) to undo the source's downward tilt.
3. In each 25 ms frame, find coefficients aₖ so that each sample is predicted from the
   previous p samples (Burg's algorithm finds them; p = 2 × number of formants = 10):
""")
    st.latex(r"x[n] \approx \sum_{k=1}^{p} a_k\, x[n-k]")
    st.markdown("""
4. The prediction filter's inverse is an **all-pole filter**: a model of the vocal tract.
   Each pair of complex poles z = r·e^{jθ} is one resonance, with
   frequency F = θ·fs/(2π) and bandwidth B = −ln(r)·fs/π.
5. Sort the resonances by frequency: the lowest are F1, F2, F3.

**The ceiling matters.** LPC always finds exactly 5 resonances below the ceiling. If the
ceiling is too high for the voice, an extra peak gets counted and the labels shift; if too
low, a real formant is missed. After a formant edit by ×k, the ceiling must move by ×k too:
this app does that automatically. Formant tracks are also unreliable in consonants, silence,
noise, and for high voices where harmonics are spaced so widely that the envelope is poorly
sampled.
""")
    st.markdown("##### Try it: place a vowel on the F1–F2 chart")
    c1, c2, c3 = st.columns(3)
    f1 = c1.slider("F1 (Hz)", 200, 900, 500, step=10, key="L7_f1")
    f2 = c2.slider("F2 (Hz)", 600, 2600, 1500, step=10, key="L7_f2")
    f0 = c3.slider("F0 (Hz)", 80, 250, 120, key="L7_f0")
    fig = _fig(10, 3.2)
    a1, a2 = fig.subplots(1, 2)
    for name, (v1, v2, _) in VOWELS.items():
        a1.scatter(v2, v1, color=MUTED)
        a1.annotate(name.split()[0], (v2, v1), textcoords="offset points", xytext=(5, 3), fontsize=9)
    a1.scatter([f2], [f1], color="#ff5c5c", s=80, marker="x", label="your vowel")
    a1.set_xlim(2600, 600)       # phonetics convention: front vowels on the left, close vowels on top
    a1.set_ylim(900, 200)
    a1.set_xlabel("F2 (Hz)  ← front    back →", fontsize=8)
    a1.set_ylabel("F1 (Hz)  ← close    open →", fontsize=8)
    a1.set_title("vowel chart (Peterson and Barney averages)", fontsize=9)
    a1.legend(fontsize=7)
    length = st.slider("Vocal-tract length for the tube model (cm)", 10.0, 20.0, 17.5, step=0.5, key="L7_len")
    tube = [(2 * n - 1) * 350 / (4 * length / 100) for n in (1, 2, 3)]
    fr, h = resonator_response((f1, f2, 2500), BANDWIDTHS)
    a2.plot(fr, h, color="#2ee6a6", label="your vowel's filter")
    for i, f in enumerate(tube):
        a2.axvline(f, color="#8b6cff", linestyle="--", linewidth=0.8, label="tube formants" if i == 0 else None)
    a2.set_xlim(0, 4000)
    a2.set_xlabel("Hz", fontsize=8)
    a2.set_ylabel("dB", fontsize=8)
    a2.legend(fontsize=7)
    a2.set_title(f"uniform {length:.1f} cm tube: {tube[0]:.0f}, {tube[1]:.0f}, {tube[2]:.0f} Hz", fontsize=9)
    _show(fig)
    _play(synth_vowel(f0, (f1, f2, 2500)), f"Synthesized vowel with F1 = {f1} Hz, F2 = {f2} Hz, F3 = 2500 Hz")


def lesson_world():
    st.info("""
**What exactly is WORLD?** WORLD is a free, open-source **speech analysis and synthesis
library**: a vocoder written in C++ by Masanori Morise (then at the University of Yamanashi,
Japan), described in a 2016 journal paper. It is a set of signal-processing algorithms, not an AI model: it has no
training data and no learned weights, and gives the same output for the same input every time.

- **What it does:** takes a recording apart into three parameter tracks (pitch, spectral
  envelope, aperiodicity), and puts a waveform back together from those tracks, edited or not.
- **How this app uses it:** through **pyworld**, a Python wrapper around the C++ library
  (`pip install pyworld`). `world_analyze` and `world_synthesize` in `voiceeditor/analysis.py`
  call it.
- **Where else it's used:** WORLD is a standard tool in speech research. It extracts the
  acoustic features for statistical parametric speech synthesis (for example the Merlin
  toolkit) and for voice-conversion research, and serves as a baseline in vocoder comparisons.
- **What it isn't:** a neural vocoder (like WaveNet or HiFi-GAN, which learn to generate audio
  from data), or a complete voice changer. It provides the parameters; the edits are ours.
""", icon=":material/info:")
    st.markdown("""
A **vocoder** (voice coder) analyzes speech into a few parameters and synthesizes speech back
from them. **WORLD** (Morise, Yokomori and Ozawa, 2016) is a high-quality vocoder built purely
from signal processing, with no trained model. Every 5 ms it produces three parameters:

| Parameter | Meaning | Algorithm |
|---|---|---|
| **F0** | pitch, 0 when unvoiced | **Harvest** (see the pitch lesson) |
| **Spectral envelope** | the smooth outline of the spectrum: the vocal-tract filter | **CheapTrick** |
| **Aperiodicity** | per frequency, how noise-like the sound is: 0 = purely harmonic, 1 = pure noise | **D4C** |

**CheapTrick: removing the harmonics to get the envelope.** A voiced spectrum is a comb of
harmonics; the envelope is the curve along their tops. CheapTrick:

1. Cuts each frame with a Hann window **three pitch periods long**, so the window always
   fits the voice (a pitch-adaptive window).
2. Smooths the power spectrum with a moving average **one F0 wide**, which removes the
   harmonic ripple.
3. Smooths further with **cepstral liftering** (low-pass filtering the log spectrum) and
   corrects for the smoothing's side effects.

Because the window follows F0, the result barely depends on pitch: the property that lets
pitch be changed without changing the envelope.

**D4C: measuring breathiness.** For each frequency band, D4C estimates how much energy is
noise rather than harmonics, using a group-delay analysis made independent of where the frame
falls in the pitch cycle. Breathy voices, fricatives and the high frequencies of most voices
are strongly aperiodic.

**Synthesis.** For voiced frames, WORLD places one pulse per pitch period. Each pulse's
spectrum is shaped by envelope × (1 − aperiodicity) and made into a short minimum-phase
impulse response. Noise shaped by envelope × aperiodicity is added, and everything is
overlap-added into a waveform. Because the three parameters are independent, editing one
(for example F0) and resynthesizing changes only that aspect of the voice.
""")
    st.markdown("##### Try it: the sample voice, decomposed")
    x, fs = _sample()
    f0, sp, ap = _sample_world()
    t = np.arange(len(f0)) * 0.005
    freqs = np.linspace(0, fs / 2, sp.shape[1])
    fig = _fig(10, 5.4)
    a1, a2, a3 = fig.subplots(3, 1, sharex=True)
    a1.plot(t, np.where(f0 > 0, f0, np.nan), color="#ff9f2e")
    a1.set_ylabel("F0 (Hz)", fontsize=8)
    a1.set_title("Harvest: F0", fontsize=9)
    keep = freqs <= 5000
    a2.pcolormesh(t, freqs[keep], 10 * np.log10(sp[:, keep].T + 1e-16), shading="auto", cmap="magma")
    a2.set_ylabel("Hz", fontsize=8)
    a2.set_title("CheapTrick: spectral envelope (dB): no harmonics, only formants", fontsize=9)
    a3.pcolormesh(t, freqs[keep], ap[:, keep].T, shading="auto", cmap="viridis", vmin=0, vmax=1)
    a3.set_ylabel("Hz", fontsize=8)
    a3.set_xlabel("time (s)", fontsize=8)
    a3.set_title("D4C: aperiodicity (yellow = noise-like, dark = harmonic)", fontsize=9)
    for ax in (a1, a2, a3):
        ax.tick_params(labelsize=7)
    _show(fig)
    c1, c2 = st.columns(2)
    _play(x, "Original", where=c1)
    _play(world_synthesize_cached(), "Rebuilt by WORLD from the three parameters", where=c2)


@st.cache_data(show_spinner=False)
def world_synthesize_cached():
    from .analysis import WorldParams
    f0, sp, ap = _sample_world()
    return world_synthesize(WorldParams(f0=f0, sp=sp, ap=ap, fs=FS))


def lesson_edits():
    st.markdown("""
Each slider in the editor changes exactly one WORLD parameter.

**Pitch ×k** multiplies every voiced F0 value by k. Pitch is perceived roughly on a
logarithmic scale (an octave is ×2 at any height), so scaling is the natural operation:
×1.06 is one semitone, ×2 an octave.

**Pitch range ×r** stretches the intonation around the mean, in the log domain:
""")
    st.latex(r"\log F_0' = \mu + r\,(\log F_0 - \mu) + \log k,\qquad \mu = \text{mean of }\log F_0")
    st.markdown("""
r < 1 flattens the melody (monotone, sad-sounding); r > 1 exaggerates it (lively, excited).

**Formants ×α** stretches the spectral envelope along the frequency axis: a peak at f moves to
α·f.
""")
    st.latex(r"S'(f) = S(f/\alpha)")
    st.markdown("""
In code, each output bin reads the input envelope at bin/α, interpolating on **log** power so
peak shapes are kept. This imitates a shorter (α > 1) or longer (α < 1) vocal tract.

**Speed ×s** resamples the sequence of 5 ms frames: speed 2 keeps every other frame, 0.5
repeats each frame twice. Envelope and aperiodicity are interpolated between neighbouring
frames; F0 takes the nearest frame so voiced and unvoiced frames are never blended. Pitch
doesn't change, because F0 values are kept, only their timing changes.

**Why pitch alone doesn't make a male voice female.** Women's voices differ from men's in
*both* F0 (about 1.7× higher) and formants (about 15–20 % higher, from a shorter vocal tract).
Raising only F0 sounds like a man speaking in a high register. Playing a recording faster
(as with a sped-up tape) raises pitch *and* formants by the same factor, often too much for
the formants, giving the "chipmunk" sound. Convincing changes move F0 and formants separately,
each by its own factor: the *More female* preset uses F0 ×1.6 and formants ×1.17.

**Emotion presets** follow findings from emotion research (for example Murray and Arnott,
1993): happy speech tends to have higher, more variable pitch and a faster rate; sad speech
lower, flatter pitch and a slower rate. Real emotion also changes loudness, voice quality
(breathy, tense), timing of individual words, and word choice, which these presets don't
model. Expect a hint of the emotion, not a convincing actor.
""")
    st.markdown("##### Try it: three ways to raise a voice")
    x, fs = _sample()
    sounds = _three_ways()
    cols = st.columns(4)
    _play(x, "Original", where=cols[0])
    _play(sounds["pitch"], "Pitch ×1.6 only", where=cols[1])
    _play(sounds["both"], "Pitch ×1.6 + formants ×1.17", where=cols[2])
    _play(sounds["tape"], "Tape-style ×1.6 (all scaled)", where=cols[3])


@st.cache_data(show_spinner=False)
def _three_ways():
    from dataclasses import replace

    from .analysis import WorldParams
    x, fs = _sample()
    f0, sp, ap = _sample_world()
    p = WorldParams(f0=f0, sp=sp, ap=ap, fs=fs)
    pitch = world_synthesize(replace(p, f0=shift_pitch(f0, 1.6, 1.0)))
    both = world_synthesize(replace(p, f0=shift_pitch(f0, 1.6, 1.0), sp=warp_formants(sp, 1.17)))
    tape = resample_poly(x, 5, 8)         # 1/1.6 of the samples, played at the same rate
    return {"pitch": _peak(pitch, 0.9), "both": _peak(both, 0.9), "tape": _peak(tape, 0.9)}


def lesson_noise():
    st.markdown("""
**Noise** is any unwanted sound mixed with the speech. This app can add four kinds:

| Type | Spectrum | Sounds like |
|---|---|---|
| **White** | equal power at every frequency | hiss, radio static |
| **Pink** | power falls 3 dB per octave (∝ 1/f) | rain, waterfall |
| **Brown** | power falls 6 dB per octave (∝ 1/f²) | rumble, distant traffic |
| **Hum** | 50 or 60 Hz plus its harmonics | mains electricity (50 Hz in Egypt and Europe, 60 Hz in the Americas) |

**Signal-to-noise ratio (SNR)** compares speech power Pₛ with noise power Pₙ:
""")
    st.latex(r"\text{SNR} = 10\log_{10}\frac{P_s}{P_n}\ \text{dB}")
    st.markdown("""
20 dB is mildly noisy, 5 dB clearly noisy, 0 dB means equal power, negative means the noise is
stronger. To add noise at a chosen SNR, the app measures Pₛ **on active speech only** (20 ms
frames louder than 10 % of the 95th-percentile frame, so one click can't set the scale), so pauses don't make the SNR look better, then
scales unit-power noise by √(Pₛ / 10^{SNR/10}).

**Removing noise.** Both methods work frame by frame in the STFT domain (32 ms frames, 75 %
overlap):

1. **Estimate the noise.** The quietest 10 % of frames are assumed to contain only noise;
   their average power per frequency is the noise spectrum N(f).
2. **Compute a gain** between 0 and 1 for every frequency of every frame, from the
   ratio γ = |X|² / |N|² (the a-posteriori SNR).
3. Multiply, keep the noisy phase, and rebuild the waveform with the inverse STFT.

**Spectral subtraction** (Boll, 1979) subtracts the noise magnitude: |Ŝ| = |X| − α|N|.
As a gain, G = 1 − α/√γ. α is the *Strength* slider.

**Wiener filter** sets G = ξ / (1 + ξ), where ξ estimates the clean-speech SNR, here
ξ = max(γ − 1, 0) with power smoothed over 5 frames (40 ms). It's the gain that minimises
the mean squared error for Gaussian signals. The smoothing gives fewer random leftovers at
the cost of slightly less removal.

**Spectral floor.** Both gains are kept above a floor β (the *Spectral floor* slider).
Without it, random noise peaks that survive in isolated bins become short tones:
**musical noise**, the typical "underwater beeps" of these methods.

**Limits.** Both methods assume the noise is **stationary**: the same spectrum all the time.
They work on fans, hiss and hum, and poorly on babble, music or traffic that changes. They
also remove some speech wherever speech is weaker than the noise. Removing more noise always
damages more speech: listen to *Removed part* in the editor to hear it,
and see the next lesson to measure it.
""")
    st.markdown("##### Try it")
    c1, c2 = st.columns(2)
    kind = c1.selectbox("Noise type", NOISE_TYPES, key="L9_kind")
    snr = c2.slider("SNR (dB)", -10, 30, 5, key="L9_snr")
    x, fs = _sample()
    noise = make_noise(kind, x.size, fs) * np.sqrt(speech_power(x, fs) / 10 ** (snr / 10))
    fig = _fig(10, 2.6)
    a1, a2 = fig.subplots(1, 2)
    shown = NOISE_TYPES[:3] + ([kind] if kind.startswith("Hum") else [])
    for k in shown:
        f, pxx = welch(make_noise(k, FS * 4, FS), FS, nperseg=2048)
        a1.semilogx(f[1:], 10 * np.log10(pxx[1:]), label=k, linewidth=1)
    a1.set_xlabel("frequency (Hz, log scale)", fontsize=8)
    a1.set_ylabel("power (dB)", fontsize=8)
    a1.set_title("noise spectra (equal total power)", fontsize=9)
    a1.legend(fontsize=7)
    g_db = np.linspace(-5, 30, 300)
    gamma = 10 ** (g_db / 10)
    a2.plot(g_db, 20 * np.log10(np.maximum(1 - 1 / np.sqrt(gamma), 0.05)), label="spectral subtraction (α = 1)")
    xi = np.maximum(gamma - 1, 0)
    a2.plot(g_db, 20 * np.log10(np.maximum(xi / (1 + xi), 0.05)), label="Wiener")
    a2.axhline(20 * np.log10(0.05), color=MUTED, linestyle=":", label="floor 5 %")
    a2.set_xlabel("γ = noisy power / noise power (dB)", fontsize=8)
    a2.set_ylabel("gain (dB)", fontsize=8)
    a2.set_title("how much each method keeps", fontsize=9)
    a2.legend(fontsize=7)
    for ax in (a1, a2):
        ax.tick_params(labelsize=7)
    _show(fig)
    cols = st.columns(3)
    _play(x, "Speech", where=cols[0])
    _play(noise, "Noise alone", where=cols[1])
    mix = x + noise
    _play(mix / max(1.0, np.max(np.abs(mix)) / 0.95), f"Mixed at {snr} dB SNR", where=cols[2])


def lesson_quality():
    st.info("""
**In plain words.** Take the clean voice. Everything else in the result is *mess*: noise that is
still there, plus any damage the cleaning did to the voice. Each SNR answers one question:
**how many times stronger is the voice than the mess?** Bigger is better.

| Number | Means | Sounds like |
|---|---|---|
| below 0 dB | the mess is stronger than the voice | noise wins |
| 0 to 10 dB | voice 1× to 10× stronger | heavy noise |
| 10 to 20 dB | voice 10× to 100× stronger | noticeable noise |
| 20 to 30 dB | voice 100× to 1000× stronger | slight noise |
| above 30 dB | voice over 1000× stronger | almost clean |

Reading dB: **+3 dB ≈ 2×**, **+10 dB = 10×**, **+20 dB = 100×** (in power). So "+5 dB after
cleaning" means the mess got about 3× weaker.

**LSD** is the odd one out: it compares the *sound colour* (which frequencies are present), and
**smaller is better**. Below 2 dB the colour is almost identical, 2–5 dB close, 5–10 dB clearly
different, above 10 dB very different. These LSD bands are a rough guide, not a standard.
""", icon=":material/lightbulb:")
    st.markdown("""
Listening is the final test, but ears disagree and can't be averaged. An **objective quality
metric** turns "how good is this?" into a number, so methods and settings can be compared fairly.
The four metrics here are **intrusive** (full-reference): they compare the processed signal ŝ
with the **clean original** s, sample by sample. That's why the editor shows them only when it
adds the noise itself: then it knows exactly which part of the mix is speech and which is noise.

**1. SNR: the whole signal at once.** Everything that differs from the clean speech counts as error:
""")
    st.latex(r"\text{SNR} = 10\log_{10}\frac{\sum_n s[n]^2}{\sum_n \big(s[n]-\hat s[n]\big)^2}\ \text{dB}")
    st.markdown("""
Simple, but two weaknesses. Loud vowels dominate the sums, so errors in quiet consonants barely
count. And it isn't **scale-invariant**: the clean speech at half volume is "wrong" by s/2 and scores
only 6 dB, although nothing is lost.

Note: the *Add noise* slider sets the SNR on **active speech** only, while this metric uses the whole
signal including pauses, so it reads about 2 dB lower for the samples.

**2. Segmental SNR: one SNR per frame, then the average.** Cut both signals into frames
(32 ms here, the same grid as the noise remover), compute the SNR of each speech frame, clamp it, and average:
""")
    st.latex(r"\text{SNR}_\text{seg} = \frac{1}{|\mathcal{M}|}\sum_{m\in\mathcal{M}}"
             r"\operatorname{clip}_{[-10,\,35]}\!\left(10\log_{10}\frac{\sum_{n} s_m[n]^2}"
             r"{\sum_{n}\big(s_m[n]-\hat s_m[n]\big)^2}\right)")
    st.markdown("""
𝓜 is the set of frames where someone is talking (louder than −20 dB from the 95th-percentile frame,
the same rule used to set the added-noise level). Every syllable now counts equally, which matches
hearing better than one global SNR (Quackenbush, Barnwell and Clements, 1988). The clamp stops
silent frames from going to −∞ and near-perfect frames from going to +∞ and dominating the mean
(Hansen and Pellom, 1998).

**3. SI-SDR: scale-invariant signal-to-distortion ratio.** First find how much of the reference is in
the estimate, by projecting ŝ onto s (both with their mean removed). That part is the *target*;
everything else is distortion:
""")
    st.latex(r"\alpha = \frac{\langle \hat s, s\rangle}{\lVert s\rVert^2},\qquad "
             r"\text{SI-SDR} = 10\log_{10}\frac{\lVert \alpha s\rVert^2}{\lVert \hat s-\alpha s\rVert^2}\ \text{dB}")
    st.markdown("""
A change of volume is no longer an error: s/2 scores as perfect. Le Roux, Wisdom, Erdogan and Hershey
(2019) proposed it to fix the older **BSS Eval SDR** (Vincent, Gribonval and Févotte, 2006), whose
filtering allowance could be gamed. It's the standard score in speech enhancement and source
separation papers, usually reported as the **SI-SDR improvement** (SI-SDRi): output minus input.

**4. Log-spectral distance (LSD): compare spectra, not waveforms.** For every speech frame, take the
power spectra P (clean) and P̂ (processed) in dB and measure their RMS difference over frequency:
""")
    st.latex(r"\text{LSD} = \frac{1}{|\mathcal{M}|}\sum_{m\in\mathcal{M}}"
             r"\sqrt{\frac{1}{K}\sum_{k=1}^{K}\Big(10\log_{10}P_m[k]-10\log_{10}\hat P_m[k]\Big)^2}\ \text{dB}")
    st.markdown("""
Lower is better; 0 dB means identical spectra. LSD ignores phase, and because it works in dB, a
quiet frequency band counts as much as a loud one: it sees what a noise remover does to the
weak high frequencies that the three SNRs hardly notice (Gray and Markel, 1976).
Powers are floored 80 dB below the loudest bin so empty bins stay finite.

| Metric | Unit | Better | Scale-invariant | Sensitive to |
|---|---|---|---|---|
| SNR | dB | higher | no | loud parts |
| Segmental SNR | dB | higher | no | every speech frame equally |
| SI-SDR | dB | higher | **yes** | waveform shape |
| LSD | dB | **lower** | no | spectral shape, every band equally (phase ignored) |

**What the mess is made of.** After noise removal the mess has two sources: **leftover noise** (noise
that wasn't removed) and **damage to the voice** (speech removed or changed with it). Both methods
multiply each STFT bin of the noisy mix X = S + N by a gain G between 0 and 1, and a fixed gain is
linear, so the result splits exactly:
""")
    st.latex(r"\hat S = G\,(S + N) = \underbrace{G S}_{\text{voice after removal}} + "
             r"\underbrace{G N}_{\text{leftover noise}},\qquad "
             r"\text{damage} = G S - S")
    st.markdown("""
The gains still come from the noisy mix, as in the app; they are just applied to the speech and the
noise separately. The two parts are nearly independent, so their powers add up to the mess power.
This turns the strength slider into a visible trade-off: gentle removal leaves noise and spares
the voice, hard removal does the opposite. The best setting is usually near where the two are equal
(Loizou's book discusses this speech-distortion versus noise-reduction trade-off at length).

**When the numbers disagree**, each disagreement tells you something:

- **SI-SDR above SNR**: part of the mess SNR sees is only a change of loudness (noise removal also
  turns the voice down a little).
- **Segmental SNR below SNR**: the quiet parts of the speech are messier than the loud ones.
- **SNRs up, LSD down… or LSD up**: LSD weighs every frequency equally, so it notices a remover that
  turns many weak frequencies down slightly, which the SNRs (dominated by strong frequencies) miss.
  With hum this is typical.
- **Result worse than the noisy input**: the removal damages more voice than the noise it removes.

**Limits.** None of these models hearing. Musical noise is a few isolated tones: it costs little SNR
but sounds annoying. Standardised perceptual scores exist: **PESQ** (ITU-T P.862, 2001) and its
successor **POLQA** (P.863) predict listener opinion scores, and **STOI** (Taal et al., 2011)
predicts intelligibility; they need more code or licensed implementations. For this app's
methods, read the four numbers together, then listen.
""")
    st.markdown("##### Try it: score the two noise removers")
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Noise type", NOISE_TYPES, key="L11_kind")
    snr_db = c2.slider("Added noise SNR (dB)", -5, 30, 5, key="L11_snr")
    strength = c3.slider("Removal strength ×", 0.5, 4.0, 1.0, step=0.1, key="L11_strength")
    rows, tracks, x_clean, sounds, si_values = _quality_demo(kind, snr_db, strength)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.markdown("\n".join(f"- **{row['Signal']}** ({verdict('si_sdr', v)[0]}): {in_words('si_sdr', v)}"
                          for row, v in zip(rows, si_values)))
    st.caption("Segmental SNR, frame by frame (speech frames only). Hover for values, drag to zoom, "
               "click the legend to highlight one.")
    st.altair_chart(quality_track_chart([{"label": label, "db": db, "speech": speech, "color": color}
                                         for label, (db, speech, color) in tracks.items()],
                                        (NPERSEG - NOVERLAP) / FS, SEG_MIN_DB, SEG_MAX_DB), width="stretch")
    st.markdown("**The trade-off: leftover noise against damage, for every strength.** Each line says how "
                "many dB stronger the voice is than one part of the mess (higher is better): **dashed** "
                "against the leftover noise, **solid** against the damage. Raising the "
                "strength lifts the noise lines and lowers the damage lines; where they cross, both parts are "
                "equal. For white noise at 5 dB they cross near ×1 for spectral subtraction and ×2 for the "
                "Wiener filter: the same strengths that give each method its best scores.")
    st.altair_chart(_tradeoff_chart(kind, snr_db), width="stretch")
    cols = st.columns(4)
    _play(x_clean, "Clean reference", where=cols[0])
    for col, (label, y) in zip(cols[1:], sounds.items()):
        _play(y, label, where=col)
    st.markdown("""
Things to notice: the last row, the clean speech at half volume, loses about 6 dB of SNR and
segmental SNR while SI-SDR calls it perfect. With **hum**, both removers raise every SNR, yet LSD gets
worse: hum sits in a few low bins, so the noisy spectrum was already close, while the removal
lowers speech bins all over the spectrum. Raise the strength to see the trade-off: SNRs climb at
first, then speech damage takes over. For this voice, spectral subtraction scores best near ×1 and
drops quickly after; the Wiener filter peaks near ×2. The two Strength values don't mean the same
thing: spectral subtraction multiplies the noise *magnitude*, the Wiener filter the noise *power*,
so Wiener ×2 treats the noise as only √2 ≈ 1.4 times louder in magnitude terms.
""")


@st.cache_data(show_spinner=False)
def _tradeoff_data(kind: str, snr_db: float):
    x, fs = _sample()
    ref, noise = noise_parts(x, fs, AddNoise(True, kind, float(snr_db)))
    rows = []
    for m in REMOVE_METHODS:
        for strength in np.arange(0.5, 4.01, 0.25):
            parts = mess_breakdown(ref, *split_result(ref, noise, fs, RemoveNoise(True, m, float(strength))))
            rows += [{"strength": strength, "method": m, "part": "vs leftover noise",
                      "dB": parts["voice_vs_noise_db"]},
                     {"strength": strength, "method": m, "part": "vs damage", "dB": parts["voice_vs_damage_db"]}]
    return rows


def _tradeoff_chart(kind: str, snr_db: float):
    import altair as alt
    import pandas as pd

    return (
        alt.Chart(pd.DataFrame(_tradeoff_data(kind, snr_db)))
        .mark_line(point=True)
        .encode(
            x=alt.X("strength:Q", title="Strength ×"),
            y=alt.Y("dB:Q", title="voice stronger by (dB)"),
            color=alt.Color("method:N", scale=alt.Scale(domain=REMOVE_METHODS,
                                                        range=[METHOD_COLORS[m] for m in REMOVE_METHODS]),
                            legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.StrokeDash("part:N", legend=None,      # explained in the text: dashed = noise
                                      scale=alt.Scale(domain=["vs leftover noise", "vs damage"],
                                                      range=[[5, 3], [1, 0]])),
            tooltip=["method:N", "part:N", alt.Tooltip("strength:Q", format=".2f"), alt.Tooltip("dB:Q", format=".1f")],
        )
        .properties(height=300)
    )


@st.cache_data(show_spinner=False)
def _quality_demo(kind: str, snr_db: float, strength: float):
    """Noisy and denoised versions of the sample, with the four metrics of each."""
    x, fs = _sample()
    ref, noise = noise_parts(x, fs, AddNoise(True, kind, float(snr_db)))
    signals = {"With added noise": ref + noise}
    for m in REMOVE_METHODS:
        signals[m] = remove_noise(ref + noise, fs, RemoveNoise(True, m, strength))
    rows, si_values = [], []
    for label, y in list(signals.items()) + [("Clean speech at half volume", 0.5 * ref)]:
        q = quality(ref, y, fs)
        si_values.append(q["si_sdr"])
        rows.append({"Signal": label} | {f"{lab} ({u}) {'↑' if hi else '↓'}": "> 99" if q[k] > 99 else f"{q[k]:.1f}"
                                         for k, (lab, u, hi, _) in METRICS.items()})
    colors = [PANEL_COLORS["noisy"], *(METHOD_COLORS[m] for m in REMOVE_METHODS)]
    tracks = {label: (*segmental_snr_track(ref, y), c) for (label, y), c in zip(signals.items(), colors)}
    return rows, tracks, ref, signals, si_values


def lesson_arabic():
    st.markdown("""
Arabic has **emphatic consonants**: ص (ṣ, /sˤ/), ض (ḍ, /dˤ/), ط (ṭ, /tˤ/) and ظ (ẓ, /ðˤ/),
the emphatic counterparts of س, د, ت and ذ. They're articulated with a secondary constriction
in the pharynx: the tongue root pulls back toward the back wall of the throat
(**pharyngealization**).

**What it does acoustically.** Narrowing the pharynx and widening the mouth cavity
**lowers F2** of nearby vowels, often by several hundred hertz, and raises F1 slightly. The
effect spreads beyond the consonant into neighbouring vowels and sometimes across a whole
syllable or word. Listeners hear the vowel as "darker" or "heavier"; for example the /a/ in
صار sounds backer than in سار.

**Exercise in the editor**

1. Choose *Record* in the sidebar and say a minimal pair, pausing between words, for
   example **سَد** (sadd) then **صَد** (ṣadd), or **تين** then **طين**.
2. Keep *Show formants* on. Click the middle of each vowel on the spectrogram.
3. Compare F2 in the inspection table: it should be clearly lower after the emphatic
   consonant.
4. Look at the green dots: the second band from the bottom sits lower in the emphatic word.

The formant ceiling is chosen from your average pitch (5000 Hz for low voices, 5500 Hz for
higher ones). If the F2 dots jump around, the recording may be too quiet or noisy.
""")


def lesson_tools():
    st.markdown("""
**How a recording flows through the app**
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style=rounded, fontsize=11];
  in [label="Input\\nfile, microphone\\nor sample"];
  load [label="Load\\nmono, 16 kHz\\npeak-normalize"];
  add [label="Add noise\\n(optional)"];
  rem [label="Remove noise\\n(optional)"];
  ana [label="WORLD analysis\\nHarvest, CheapTrick, D4C"];
  edit [label="Edits\\npitch, range,\\nformants, speed"];
  syn [label="WORLD\\nsynthesis"];
  out [label="Result\\nplay, plot, export"];
  met [label="Quality metrics\\nSNR, segSNR,\\nSI-SDR, LSD"];
  in -> load -> add -> rem -> ana -> edit -> syn -> out;
  rem -> out [style=dashed, label="no voice edits"];
  add -> met [style=dashed, label="clean reference"];
  rem -> met [style=dashed];
}
""")
    st.markdown("""
**Libraries**

| Tool | What it is | Used for |
|---|---|---|
| **NumPy** | arrays and maths in Python | every computation on samples and frames |
| **SciPy** | scientific algorithms | resampling, STFT and inverse STFT, spectrograms, filters |
| **soundfile** | reads and writes audio files (libsndfile) | WAV, FLAC, OGG, MP3 input; WAV export |
| **pyworld** | Python wrapper for **WORLD**, a free C++ speech analysis/synthesis library (a signal-processing vocoder, no AI) by M. Morise | Harvest, CheapTrick, D4C, synthesis |
| **parselmouth** | Python interface to **Praat**, the standard phonetics program | formant tracking (Burg LPC) |
| **matplotlib** | plotting library | waveforms, spectrograms, lesson charts |
| **Streamlit** | turns a Python script into a web page | the whole user interface |
| **streamlit-image-coordinates** | reports where an image was clicked | clicking on waveforms and spectrograms |

**Code map**

| File | Contents |
|---|---|
| `app.py` | the user interface: sidebar, panels, inspection, metrics |
| `voiceeditor/audio_io.py` | loading, resampling, normalizing, WAV export |
| `voiceeditor/noise.py` | adding noise at an SNR; spectral subtraction and Wiener filter |
| `voiceeditor/analysis.py` | WORLD analysis and synthesis; Praat formants; summary numbers |
| `voiceeditor/metrics.py` | quality metrics: SNR, segmental SNR, SI-SDR, log-spectral distance |
| `voiceeditor/edits.py` | pitch, pitch range, formant warp, time stretch; presets |
| `voiceeditor/plots.py` | panels, click-to-data conversion, spectrum at a moment |
| `voiceeditor/lessons.py` | these lessons and their demos |
| `check_pipeline.py` | measures that each edit and noise tool does what it claims |

**References**

- G. Fant, *Acoustic Theory of Speech Production*, 1960 (source–filter model).
- G. E. Peterson and H. L. Barney, "Control methods used in a study of the vowels", 1952.
- D. H. Klatt, "Software for a cascade/parallel formant synthesizer", 1980.
- S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction", 1979.
- M. Morise, F. Yokomori and K. Ozawa, "WORLD: a vocoder-based high-quality speech synthesis
  system for real-time applications", 2016; M. Morise, "Harvest", 2017; "CheapTrick", 2015;
  "D4C", 2016.
- S. R. Quackenbush, T. P. Barnwell and M. A. Clements, *Objective Measures of Speech Quality*,
  Prentice Hall, 1988 (segmental SNR).
- J. H. L. Hansen and B. L. Pellom, "An effective quality evaluation protocol for speech
  enhancement algorithms", ICSLP 1998.
- A. H. Gray and J. D. Markel, "Distance measures for speech processing", IEEE Trans. ASSP, 1976
  (log-spectral distance).
- E. Vincent, R. Gribonval and C. Févotte, "Performance measurement in blind audio source
  separation", IEEE Trans. ASLP, 2006 (BSS Eval SDR).
- J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR – half-baked or well done?",
  ICASSP 2019 (SI-SDR).
- P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013 (objective
  quality measures for speech enhancement).
- I. R. Murray and J. L. Arnott, "Toward the simulation of emotion in synthetic speech", 1993.
- P. Boersma and D. Weenink, *Praat: doing phonetics by computer* (praat.org).
""")


def lesson_glossary():
    terms = {
        "Amplitude": "Size of the pressure (or sample value) change; related to loudness.",
        "Aperiodicity": "How noise-like a frequency band is, from 0 (harmonic) to 1 (noise). Measured by D4C.",
        "Aliasing": "A frequency above fs/2 appearing at a false lower frequency after sampling.",
        "Autocorrelation": "A signal compared with shifted copies of itself; peaks at the period.",
        "Bandwidth (of a formant)": "Width of a resonance peak; narrow = sharp, ringing resonance.",
        "Cepstrum": "Spectrum of the log spectrum; separates smooth envelope from harmonic ripple.",
        "dB (decibel)": "Logarithmic level unit: 20·log10 of an amplitude ratio, 10·log10 of a power ratio.",
        "DFT / FFT": "Transform from samples to frequencies; the FFT is a fast algorithm for the DFT.",
        "Envelope (spectral)": "Smooth curve along the tops of the harmonics: the vocal-tract filter.",
        "F0": "Fundamental frequency: vocal-fold vibration rate; perceived as pitch.",
        "Formant": "Resonance of the vocal tract: F1, F2, F3 from the bottom.",
        "Formant ceiling": "Highest frequency Praat searches for formants; must suit the voice.",
        "Frame": "Short piece of signal (5–40 ms) analyzed as if steady.",
        "Fricative": "Consonant made by turbulent air through a narrow gap: /s/, /f/, /ʃ/.",
        "Harmonic": "Component at a whole multiple of F0.",
        "Hann window": "Smooth bell-shaped window that reduces spectral leakage.",
        "Intonation": "The melody of speech: F0 rising and falling over a phrase.",
        "LPC": "Linear prediction: models each sample from past samples; gives an all-pole vocal-tract filter.",
        "LSD": "Log-spectral distance: RMS difference in dB between two spectra, averaged over frames.",
        "Musical noise": "Short random tones left by spectral subtraction or Wiener filtering.",
        "Nyquist frequency": "Half the sample rate: the highest frequency a digital signal can hold.",
        "Pharyngealization": "Secondary narrowing of the pharynx; makes Arabic emphatics; lowers F2.",
        "Prosody": "Pitch, loudness and timing patterns of speech beyond the individual sounds.",
        "Reference (clean)": "The original speech a processed signal is compared with by intrusive quality metrics.",
        "Sample rate": "Samples per second (Hz). This app uses 16,000.",
        "Segmental SNR": "SNR computed per short frame, clamped (here to −10…35 dB) and averaged over speech frames.",
        "SI-SDR": "Scale-invariant signal-to-distortion ratio: SNR after rescaling the estimate to fit the reference.",
        "SNR": "Signal-to-noise ratio in dB: 10·log10(speech power / noise power).",
        "Source–filter model": "Speech = glottal source shaped by the vocal-tract filter.",
        "Spectrogram": "Picture of spectra over time: time across, frequency up, level as colour.",
        "Spectrum": "Strength of each frequency in a signal.",
        "STFT": "Short-Time Fourier Transform: FFTs of overlapping windowed frames.",
        "Stationary noise": "Noise whose spectrum doesn't change over time.",
        "Vocoder": "System that analyzes speech into parameters and synthesizes it back.",
        "Voiced / unvoiced": "With / without vocal-fold vibration.",
        "Wiener filter": "Noise-reduction gain ξ/(1+ξ) that minimises mean squared error.",
        "WORLD": "Free C++ library by M. Morise (2016) that splits speech into F0 (Harvest), spectral envelope "
                 "(CheapTrick) and aperiodicity (D4C), and rebuilds speech from them. Signal processing, no AI.",
    }
    st.dataframe([{"Term": k, "Meaning": v} for k, v in terms.items()], hide_index=True,
                 width="stretch", height=36 * (len(terms) + 1))


LESSONS = [
    ("1. What is sound?", lesson_sound),
    ("2. Digital audio", lesson_digital),
    ("3. How the voice is made", lesson_voice),
    ("4. Frequency and spectrum", lesson_spectrum),
    ("5. The spectrogram", lesson_spectrogram),
    ("6. Pitch (F0) and how it's tracked", lesson_pitch),
    ("7. Formants and how they're measured", lesson_formants),
    ("8. The WORLD vocoder", lesson_world),
    ("9. How the voice edits work", lesson_edits),
    ("10. Noise: adding and removing it", lesson_noise),
    ("11. Measuring quality", lesson_quality),
    ("12. Arabic emphatic consonants", lesson_arabic),
    ("13. Tools, code and references", lesson_tools),
    ("14. Glossary", lesson_glossary),
]


def render() -> None:
    """Draw the Learn tab: lesson list on the left, the chosen lesson on the right."""
    nav, body = st.columns([1, 4])
    with nav:
        choice = st.radio("Lessons", [t for t, _ in LESSONS], key="lesson", label_visibility="collapsed")
    with body:
        title, fn = next((t, f) for t, f in LESSONS if t == choice)
        st.markdown(f"### {title.split('. ', 1)[1]}")
        fn()
