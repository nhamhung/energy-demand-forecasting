#!/usr/bin/env python
"""One-click launcher: open the notebooks (JupyterLab) or the Streamlit app in your browser.

You normally don't run this yourself. Double-click `start_windows.bat` (Windows)
or `start_mac.command` (macOS): they install uv if needed, let uv set up the
exact Python and packages from `uv.lock`, then run this script.

Usage (from the project folder):
    uv run python scripts/launch.py            # menu
    uv run python scripts/launch.py notebooks  # straight to JupyterLab
    uv run python scripts/launch.py app        # straight to the Streamlit app
    uv run python scripts/launch.py app-full   # the app on the full dataset instead of the one-year sample
    uv run python scripts/launch.py check      # only check the setup, then exit
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "Energy Demand Forecasting"
APP = PROJECT_ROOT / "app" / "streamlit_app.py"
NOTEBOOKS = PROJECT_ROOT / "notebooks"


# --- Project-specific: data the notebooks need ------------------------------

# Same paths as src/energy_demand_forecasting/config.py (not imported here, so the
# menu appears without loading pandas first). All three ship with the project.
FULL_CSV = PROJECT_ROOT / "data" / "raw" / "Romania_hourly.csv"  # 2019-2026 snapshot, 3.7 MB
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample" / "Romania_hourly_sample.csv.gz"  # last year only
MODEL = PROJECT_ROOT / "models" / "model.joblib"
NOTEBOOK_DATA = [FULL_CSV, MODEL]
KAGGLE_DIR = Path.home() / ".kaggle"
TOKEN_FILE = KAGGLE_DIR / "access_token"


def ensure_notebook_data() -> bool:
    """The notebook reads the saved model and the full dataset (it falls back
    to the one-year sample without it, but then the cells that look at specific
    2023-2025 dates fail). Both ship with the project; if the dataset was
    deleted (e.g. to get newer data), offer to download it from Kaggle.
    """
    if not MODEL.exists():
        print(f"\nThe notebook needs {MODEL.relative_to(PROJECT_ROOT)}, which is missing.")
        print("It ships with the project: download the project ZIP again for a complete copy.")
        return False
    if FULL_CSV.exists():
        return True
    print(f"\nThe full dataset ({FULL_CSV.relative_to(PROJECT_ROOT)}) is missing.")
    print("It can be downloaded from Kaggle (free account needed, about 4 MB).")
    if input("Download it now? [y/N] ").strip().lower() == "y" and ensure_full_data():
        return True
    print("Opening the notebook on the one-year sample: its numbers differ from the text, and the")
    print("cells that look at specific 2023-2025 dates show errors.")
    return True


def _kaggle_credentials_found() -> bool:
    """The places the `kaggle` package looks for credentials."""
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR") or KAGGLE_DIR)
    candidates = [TOKEN_FILE, KAGGLE_DIR / "access_token.txt", config_dir / "kaggle.json"]
    return any(p.is_file() and p.stat().st_size > 0 for p in candidates)


def _ask_for_kaggle_token() -> bool:
    print("\nDownloading the dataset from Kaggle needs a free Kaggle account and API token.")
    print('Step-by-step: START_HERE.md, section "Get the newest data from Kaggle".')
    print("  1. Sign in at https://www.kaggle.com")
    print('  2. Open https://www.kaggle.com/settings/api and click "Generate New Token"')
    print("  3. Copy the token it shows, paste it here and press Enter")
    token = input("Paste your Kaggle API token (or just press Enter to cancel): ").strip().strip("'\"")
    if not token:
        print("Cancelled.")
        return False
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
    try:
        TOKEN_FILE.chmod(0o600)
    except OSError:
        pass
    print(f"Saved the token to {TOKEN_FILE} (you won't be asked again).")
    return True


def ensure_full_data() -> bool:
    """Make sure data/raw/Romania_hourly.csv exists when full-data mode is requested;
    if it was deleted, download the current version of the live Kaggle dataset."""
    if FULL_CSV.exists():
        return True
    if not _kaggle_credentials_found() and not _ask_for_kaggle_token():
        return False
    print("\nDownloading the newest version of the dataset from Kaggle (about 4 MB)...")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from energy_demand_forecasting import data

        ok = data._download_from_kaggle()
    except (Exception, SystemExit):  # the kaggle package exits on bad credentials
        ok = False
    if ok:
        print(f"Done: saved {FULL_CSV.relative_to(PROJECT_ROOT)}. The notebook and menu choice 3 use it.")
        return True
    print("\nThe download didn't work. Usually the token is wrong or expired, or there is no internet.")
    print(f"Generate a new token on https://www.kaggle.com/settings/api, delete {TOKEN_FILE},")
    print("and try again. Menu choice 2 still runs the app on the one-year sample.")
    return False


# --- macOS: the OpenMP runtime LightGBM needs --------------------------------
# LightGBM's macOS wheels expect Homebrew's libomp (`brew install libomp`), which
# most Macs don't have. If it can't load, fetch libomp from conda-forge (~300 kB,
# no account, checksum-pinned) into .libomp/ and point the app and notebooks at it.

LIBOMP_DIR = PROJECT_ROOT / ".libomp"
LIBOMP_PACKAGES = {  # platform.machine() -> (conda-forge file, sha256)
    "arm64": ("osx-arm64/llvm-openmp-23.1.2-hdb3d66b_0.conda",
              "f9aea67117079c00b9663ee0bbca7e5409dbe94625c34e68881ce2c97c3d47c9"),
    "x86_64": ("osx-64/llvm-openmp-23.1.2-h8c1e6b9_0.conda",
               "3c17127e2a94d60a8babef57c66d643374c5b08c896ec5f0987a62ecfb81c8ee"),
}


def _openmp_loads() -> bool:
    probe = [sys.executable, "-c", "import lightgbm"]
    return subprocess.run(probe, env=_env(), capture_output=True).returncode == 0


def ensure_openmp(force: bool = False) -> bool:
    """True if LightGBM loads; on a Mac where it can't, download libomp once first."""
    if sys.platform != "darwin" or (not force and _openmp_loads()):
        return True
    import hashlib
    import io
    import platform
    import ssl
    import tarfile
    import zipfile

    import certifi
    from backports import zstd

    print("\nOne-time fix: downloading the OpenMP runtime that LightGBM needs on macOS (~300 kB)...")
    name, sha256 = LIBOMP_PACKAGES[platform.machine()]
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        url = f"https://conda.anaconda.org/conda-forge/{name}"
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            package = response.read()
        if hashlib.sha256(package).hexdigest() != sha256:
            raise OSError("checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(package)) as conda:
            inner = next(n for n in conda.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
            tar_bytes = zstd.decompress(conda.read(inner))
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            dylib = tar.extractfile("lib/libomp.dylib").read()
        LIBOMP_DIR.mkdir(exist_ok=True)
        (LIBOMP_DIR / "libomp.dylib").write_bytes(dylib)
    except (OSError, KeyError, StopIteration, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"That didn't work ({error}). The model needs it.")
        print("Fix: install Homebrew (https://brew.sh), run `brew install libomp`, then start again.")
        return False
    ok = _openmp_loads()
    print("Done." if ok else "LightGBM still can't load; see START_HERE.md, 'If something goes wrong'.")
    return ok


# --- Shared ------------------------------------------------------------------


def _free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found.")


def _open_when_ready(url: str, health: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(1)
    print(f"\nThe app is taking a while to start. Open {url} in your browser when it's ready.")


def _env() -> dict:
    env = os.environ.copy()
    # Keep Jupyter's kernels and settings inside the project, so an older
    # Jupyter install elsewhere on the computer can't take over the kernel.
    env["JUPYTER_DATA_DIR"] = str(PROJECT_ROOT / ".jupyter_data")
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    if sys.platform == "darwin" and (LIBOMP_DIR / "libomp.dylib").exists():
        # Used only if Homebrew's libomp isn't there (see ensure_openmp).
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{LIBOMP_DIR}:/usr/local/lib:/usr/lib"
    return env


def run_notebooks() -> None:
    ensure_openmp()
    if not ensure_notebook_data():
        return
    print("\nOpening JupyterLab in your browser...")
    print("Open a notebook from the list on the left, then use Run > Run All Cells.")
    print("To stop: close the browser tab, then press Ctrl+C in this window.\n")
    port = _free_port(8888)
    subprocess.run([sys.executable, "-m", "jupyter", "lab", f"--notebook-dir={NOTEBOOKS}", f"--port={port}"],
                   cwd=PROJECT_ROOT, env=_env())


def run_app(full_data: bool = False) -> None:
    ensure_openmp()
    env = _env()
    if full_data:
        if not ensure_full_data():
            return
        env["USE_FULL_KAGGLE_DATA"] = "true"  # read by src/energy_demand_forecasting/data.py
    port = _free_port(8501)
    url = f"http://localhost:{port}"
    print(f"\nStarting the app at {url} (your browser opens by itself)...")
    print("To stop: press Ctrl+C in this window.\n")
    threading.Thread(target=_open_when_ready, args=(url, f"{url}/_stcore/health"), daemon=True).start()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless=true",
                    f"--server.port={port}"], cwd=PROJECT_ROOT, env=env)


def check() -> None:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    for name in ("streamlit", "jupyterlab", "ipykernel"):
        __import__(name)
        print(f"  {name}: ok")
    if sys.platform == "darwin":
        print("OpenMP for LightGBM:", "ok" if ensure_openmp() else "missing")
    print("Model:", "ok" if MODEL.exists() else "missing (download the project again)")
    print("Full dataset:", "ok" if FULL_CSV.exists() else "missing (choice 1 or 3 can download it from Kaggle)")
    print("One-year sample:", "ok" if SAMPLE_CSV.exists() else "missing (download the project again)")
    print("Setup looks good.")


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    actions = {"notebooks": run_notebooks, "app": run_app, "app-full": lambda: run_app(full_data=True),
               "check": check}
    if choice:
        actions[choice]()
        return
    while True:
        print(f"\n=== {PROJECT_NAME} ===")
        print("  1  Open the notebooks (JupyterLab)")
        print("  2  Open the app (Streamlit)")
        print("  3  Open the app on the full dataset (2019-2026, slower to start)")
        print("  4  Quit")
        picked = input("Type 1, 2, 3 or 4 and press Enter: ").strip()
        try:
            if picked == "1":
                run_notebooks()
            elif picked == "2":
                run_app()
            elif picked == "3":
                run_app(full_data=True)
            elif picked == "4":
                return
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
