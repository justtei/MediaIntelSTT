"""Pull real broadcast test audio as 16 kHz mono WAV via yt-dlp.

Usage:
  python scripts/fetch_audio.py "<url>" --name anchor_mono --section "00:10:00-00:20:00"
"""
import argparse
import subprocess
import sys
from pathlib import Path

from audio_io import find_ffmpeg, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--name", required=True, help="output name (audio/<name>.wav)")
    ap.add_argument("--section", default=None, help="e.g. 00:10:00-00:20:00")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "audio"
    out_dir.mkdir(exist_ok=True)
    ffmpeg_dir = str(Path(find_ffmpeg()).parent)

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg_dir,
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(out_dir / f"{args.name}.%(ext)s"),
    ]
    if args.section:
        cmd += ["--download-sections", f"*{args.section}", "--force-keyframes-at-cuts"]
    cmd.append(args.url)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
