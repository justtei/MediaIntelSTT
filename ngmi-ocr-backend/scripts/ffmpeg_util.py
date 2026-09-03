"""Locate an ffmpeg binary: PATH first, then a project-local tools/ directory
(so this project can vendor a static ffmpeg build the same way
STTModelFramework does, without requiring a system install)."""
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_ffmpeg() -> str:
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    hits = list((PROJECT_ROOT / "tools").glob("**/ffmpeg.exe")) + list((PROJECT_ROOT / "tools").glob("**/ffmpeg"))
    if hits:
        return str(hits[0])
    sys.exit("ffmpeg not found (checked PATH and tools/). Install ffmpeg or vendor a static build under tools/.")
