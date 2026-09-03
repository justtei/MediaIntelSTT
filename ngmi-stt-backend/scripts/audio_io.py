"""Shared audio loading: decode anything to float32 16 kHz mono via ffmpeg."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000


def find_ffmpeg() -> str:
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    hits = list((PROJECT_ROOT / "tools").glob("**/ffmpeg.exe"))
    if hits:
        return str(hits[0])
    sys.exit("ffmpeg not found (checked PATH and tools\\). Run setup again.")


def load_audio(path: str) -> np.ndarray:
    """Return float32 mono samples at 16 kHz, whatever the input container/rate."""
    if not os.path.exists(path):
        sys.exit(f"audio file not found: {path}")
    cmd = [
        find_ffmpeg(), "-v", "error", "-i", path,
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed on {path}:\n{proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)
