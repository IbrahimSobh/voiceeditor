# Voice editor

A personal and teaching tool that edits speech with **classic signal processing, no AI**.
Load a file, record your voice or pick a sample, then change pitch, pitch range, formants
and speed, or add and remove background noise. See every change on waveforms and
spectrograms, click any moment to inspect it, and learn how it all works in 13 built-in
lessons.

![Editor: original, noisy and denoised + "More female" side by side](docs/img/editor_noise.png)

## Quick start

You need **Python 3.11 or newer** ([python.org](https://www.python.org/downloads/)) and
**Git**. Tested on Windows 11 with Python 3.13.

**1. Get the code**

```
git clone https://github.com/IbrahimSobh/voiceeditor.git
cd voiceeditor
```

**2. Create a virtual environment and install the libraries** (a few minutes the first time)

Windows:

```
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

macOS / Linux:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**3. Start the app**

Windows:

```
.venv\Scripts\streamlit run app.py
```

macOS / Linux:

```
.venv/bin/streamlit run app.py
```

Your browser opens at **http://localhost:8501**. The first start takes about 30 seconds
while matplotlib builds its font cache.

**4. Try it**

1. In the sidebar, under **1 · Input**, choose **Sample** (or **Record** to use your microphone).
2. Under **2 · Voice**, pick the preset **More female** and press play on the **After** panel.
3. Under **3 · Noise**, open **Add noise**, tick it, then open **Remove noise** and tick it too.
4. Click anywhere on a spectrogram to inspect that moment.
5. Switch to **Learn** (top right) for the lessons.

**Troubleshooting**

- *Changes in the code don't show up:* stop the app with Ctrl+C and start it again.
- *`pip install` fails while building `pyworld`:* your Python version has no ready-made
  package for it. Use Python 3.11–3.13, or install a C++ compiler (on Windows, the free
  "Microsoft C++ Build Tools").
- *Port 8501 is busy:* add `--server.port 8502` to the run command.

## Features

### Edit a voice

| Control | What it changes |
|---|---|
| **Preset** | stored settings: *More female*, *More male*, *Happy*, *Sad* |
| **Pitch ×** | the F0 (vocal-fold rate), every voiced frame scaled |
| **Pitch range ×** | how lively the intonation is: flattens or widens the F0 curve around its mean |
| **Formants ×** | vocal-tract size: stretches the spectral envelope along frequency |
| **Speed ×** | duration, without changing pitch |
| **Add noise** | white, pink, brown, 50 Hz or 60 Hz hum, at a chosen SNR (measured on active speech) |
| **Remove noise** | spectral subtraction or Wiener filter, with strength, spectral floor and noise-estimate controls |

Each panel shows the waveform and the spectrogram, with the **F0 curve** (orange, scale on
the right) and the **formants F1–F3** (green dots) drawn on top. Players keep true
loudness, so *Noise alone* and *Removed part* play at their real level. Panels can sit
side by side or stacked:

![Stacked layout: the "Sad" preset makes the pitch flatter, lower and slower](docs/img/editor_stacked.png)

### Inspect any moment

Click a waveform or spectrogram. A window opens with amplitude, level at the clicked
frequency, F0 and F1–F3 for every panel, the spectrum of that 40 ms frame (harmonics and
the WORLD envelope), and a 0.5 s clip around the click from each panel.

![Inspection window](docs/img/editor_inspect.png)

### Learn

The **Learn** view has 13 lessons with interactive demos, from what sound is to every
algorithm used here: digital audio, the source–filter model, spectrum and FFT,
spectrogram, pitch tracking (autocorrelation and Harvest), formants and LPC, the WORLD
vocoder (Harvest, CheapTrick, D4C), how each edit works, noise and its removal, Arabic
emphatic consonants, tools and references, and a glossary.

| Build a vowel from source × filter | Formants and the vowel chart | WORLD, decomposed |
|---|---|---|
| ![](docs/img/learn_voice.png) | ![](docs/img/learn_formants.png) | ![](docs/img/learn_world.png) |

## Listen: before and after

All files were made by [make_examples.py](make_examples.py) with the same code as the
app, from the two sample sentences.

### Voice edits

| Processing | Before | After |
|---|---|---|
| Male → female (*More female*: F0 ×1.6, range ×1.1, formants ×1.17) | <audio controls src="docs/audio/original_male.wav"></audio><br>[original_male.wav](docs/audio/original_male.wav) | <audio controls src="docs/audio/male_to_female.wav"></audio><br>[male_to_female.wav](docs/audio/male_to_female.wav) |
| Female → male (*More male*: F0 ×0.65, range ×0.9, formants ×0.88) | <audio controls src="docs/audio/original_female.wav"></audio><br>[original_female.wav](docs/audio/original_female.wav) | <audio controls src="docs/audio/female_to_male.wav"></audio><br>[female_to_male.wav](docs/audio/female_to_male.wav) |
| *Happy*: higher, livelier pitch, 12 % faster | <audio controls src="docs/audio/original_female.wav"></audio><br>[original_female.wav](docs/audio/original_female.wav) | <audio controls src="docs/audio/happy.wav"></audio><br>[happy.wav](docs/audio/happy.wav) |
| *Sad*: lower, flatter pitch, 18 % slower | <audio controls src="docs/audio/original_female.wav"></audio><br>[original_female.wav](docs/audio/original_female.wav) | <audio controls src="docs/audio/sad.wav"></audio><br>[sad.wav](docs/audio/sad.wav) |
| Slower, speed ×0.75 (same pitch) | <audio controls src="docs/audio/original_male.wav"></audio><br>[original_male.wav](docs/audio/original_male.wav) | <audio controls src="docs/audio/slower_0.75x.wav"></audio><br>[slower_0.75x.wav](docs/audio/slower_0.75x.wav) |
| Faster, speed ×1.5 (same pitch) | <audio controls src="docs/audio/original_male.wav"></audio><br>[original_male.wav](docs/audio/original_male.wav) | <audio controls src="docs/audio/faster_1.5x.wav"></audio><br>[faster_1.5x.wav](docs/audio/faster_1.5x.wav) |

![Male → female: original (left) and edited (right). F0 (orange) and formants (green) both move up](docs/img/example_male_to_female.png)

### Noise

| Processing | Before | After |
|---|---|---|
| Add white noise at 10 dB SNR | <audio controls src="docs/audio/original_male.wav"></audio><br>[original_male.wav](docs/audio/original_male.wav) | <audio controls src="docs/audio/noisy_white_10dB.wav"></audio><br>[noisy_white_10dB.wav](docs/audio/noisy_white_10dB.wav) |
| Remove it: spectral subtraction | <audio controls src="docs/audio/noisy_white_10dB.wav"></audio><br>[noisy_white_10dB.wav](docs/audio/noisy_white_10dB.wav) | <audio controls src="docs/audio/denoised_spectral_subtraction.wav"></audio><br>[denoised_spectral_subtraction.wav](docs/audio/denoised_spectral_subtraction.wav) |
| Remove it: Wiener filter | <audio controls src="docs/audio/noisy_white_10dB.wav"></audio><br>[noisy_white_10dB.wav](docs/audio/noisy_white_10dB.wav) | <audio controls src="docs/audio/denoised_wiener.wav"></audio><br>[denoised_wiener.wav](docs/audio/denoised_wiener.wav) |
| Add 50 Hz mains hum at 10 dB SNR | <audio controls src="docs/audio/original_female.wav"></audio><br>[original_female.wav](docs/audio/original_female.wav) | <audio controls src="docs/audio/hum_50Hz_10dB.wav"></audio><br>[hum_50Hz_10dB.wav](docs/audio/hum_50Hz_10dB.wav) |
| Remove the hum: Wiener filter, strength ×1.5 | <audio controls src="docs/audio/hum_50Hz_10dB.wav"></audio><br>[hum_50Hz_10dB.wav](docs/audio/hum_50Hz_10dB.wav) | <audio controls src="docs/audio/hum_removed_wiener.wav"></audio><br>[hum_removed_wiener.wav](docs/audio/hum_removed_wiener.wav) |

![Noisy input (left) and Wiener-filtered result (right)](docs/img/example_denoised_wiener.png)

Listen for the trade-off: spectral subtraction removes more noise but leaves faint
"musical noise" (random beeps); the Wiener filter sounds smoother but keeps a little more
hiss.

## Settings in the web address

Settings can also be given in the address, which is handy for demos, for example:

- `http://localhost:8501/?sample=sample_male_en.wav&preset=More+female&add=White&snr=15&remove=Wiener+filter`
- `http://localhost:8501/?sample=sample_male_en.wav&click=1.62,1200` (opens the inspection window)
- `http://localhost:8501/?view=learn&lesson=7`

Parameters: `sample`, `preset`, `add` (noise type), `snr`, `remove` (method), `click`
(time in s, frequency in Hz), `layout=stacked`, `view=learn`, `lesson` (1–13).

## Checks and regenerating the docs

```
.venv\Scripts\python check_pipeline.py
```

Measures that each edit and noise tool does what it claims (pitch ratio, range ratio,
formant shift, duration, SNR accuracy, noise removal, click-to-data accuracy).

```
.venv\Scripts\python make_examples.py
```

Rebuilds `docs/audio` and the example spectrogram pictures.

```
.venv\Scripts\streamlit run app.py --server.port 8503
.venv\Scripts\python make_screenshots.py 8503
```

Rebuilds the screenshots in `docs/img` with a headless Chrome or Edge (a throwaway
profile, not yours).

## How it works

Processing order: input → add noise → remove noise → voice edits. When no voice edit is
active, WORLD is skipped, so the noise result is heard exactly.

| Step | Tool | File |
|---|---|---|
| Load, mono, resample to 16 kHz | soundfile, scipy | `voiceeditor/audio_io.py` |
| Add noise at a chosen SNR | numpy | `voiceeditor/noise.py` |
| Remove noise (spectral subtraction or Wiener filter) | scipy STFT | `voiceeditor/noise.py` |
| Split speech into F0, spectral envelope, aperiodicity | WORLD (pyworld) | `voiceeditor/analysis.py` |
| Formant tracks F1–F3 for display | Praat (parselmouth) | `voiceeditor/analysis.py` |
| Edits and presets | numpy | `voiceeditor/edits.py` |
| Plots, click-to-data conversion | matplotlib | `voiceeditor/plots.py` |
| Clickable images | streamlit-image-coordinates | `app.py` |
| User interface | Streamlit | `app.py` |
| Lessons and their demos | Streamlit, matplotlib | `voiceeditor/lessons.py` |

The Learn view explains every step in detail.

## Roadmap

- Version 2: Klatt formant engine, with a view comparing the same edit through WORLD and Klatt.
- Later: loudness and brightness controls for emotion.

The `samples` folder holds two English sentences made with Windows' built-in voices
(Microsoft David and Zira) and a short Arabic recording of a male voice.
