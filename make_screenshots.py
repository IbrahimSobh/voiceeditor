"""Capture the README screenshots from a running copy of the app.

1. Start the app:   .venv\\Scripts\\streamlit run app.py --server.port 8503
2. Run:             .venv\\Scripts\\python make_screenshots.py [port]

Drives a headless Chrome (or Edge) through the Chrome DevTools Protocol, opens each
page with its settings in the web address, waits for it to finish, and saves a PNG
in docs/img. Uses a throwaway browser profile, never your own.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "img"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
SHOTS = [  # file name, address query, viewport height, seconds to wait
    ("editor_noise.png", "sample=sample_male_en.wav&preset=More+female&add=White&snr=15&remove=Wiener+filter", 900, 25),
    ("editor_inspect.png", "sample=sample_male_en.wav&preset=More+female&click=1.62,1200", 900, 20),
    ("editor_stacked.png", "sample=sample_female_en.wav&preset=Sad&layout=stacked", 1100, 20),
    ("learn_voice.png", "view=learn&lesson=3", 1450, 15),
    ("learn_formants.png", "view=learn&lesson=7", 1750, 15),
    ("learn_world.png", "view=learn&lesson=8", 1400, 20),
]
WIDTH, DEBUG_PORT = 1440, 9333


class Page:
    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8503
    exe = next(b for b in BROWSERS if Path(b).exists())
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        proc = subprocess.Popen([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                 f"--remote-debugging-port={DEBUG_PORT}", f"--user-data-dir={profile}",
                                 "--no-first-run", "about:blank"])
        try:
            for _ in range(50):
                try:
                    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/list"))
                    break
                except OSError:
                    time.sleep(0.2)
            ws_url = next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"]
            with connect(ws_url, max_size=None) as ws:
                shoot(Page(ws), port)
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def shoot(page: Page, port: int) -> None:
    page.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
    for name, query, height, wait in SHOTS:
        page.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=height,
                  deviceScaleFactor=1, mobile=False)
        page.call("Page.navigate", url=f"http://localhost:{port}/?{query}")
        time.sleep(wait)      # real time: the app computes, then its plot frames load
        png = page.call("Page.captureScreenshot", format="png")["data"]
        (OUT / name).write_bytes(base64.b64decode(png))
        print("saved", name)


if __name__ == "__main__":
    main()
