"""One place for the app's colours: a dark navy theme with a cyan accent.

Importing this module also styles every matplotlib figure (plots and lessons) to match.
The Streamlit side of the theme lives in .streamlit/config.toml and the CSS in app.py.
"""

from __future__ import annotations

from cycler import cycler
import matplotlib

BG = "#0b1220"          # page
PANEL = "#111a2e"       # cards, sidebar, figure background
PLOT = "#0d1526"        # inside the axes
GRID = "#26324d"
TEXT = "#dbe4f5"
MUTED = "#8a97b2"
ACCENT = "#22c7f0"      # cyan
ACCENT_2 = "#8b6cff"    # violet

PANEL_COLORS = {"original": "#3b9eff", "noisy": "#ff9f2e", "after": "#2ee6a6"}
F0_COLOR = "#ff8c1a"
FORMANT_COLOR = "#38e0c0"
ENVELOPE_COLOR = "#f2f5ff"

matplotlib.rcParams.update({
    "figure.facecolor": PANEL,
    "savefig.facecolor": PANEL,
    "axes.facecolor": PLOT,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.prop_cycle": cycler(color=[ACCENT, "#ff9f2e", "#2ee6a6", ACCENT_2, "#ff5c8a", "#ffd23f"]),
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
})
