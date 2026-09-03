"""Transcribe audio with Whisper via OpenVINO or faster-whisper.

Defaults to CPU on both backends — no GPU is assumed or required to run
this project. GPU (Intel iGPU via OpenVINO) and CUDA (via faster-whisper)
are fully implemented and one flag away whenever that hardware exists; see
scripts/backends.py for the shared device-fallback logic used by both this
CLI and webapp/server.py.

Examples:
  python scripts/transcribe.py audio/clip.wav                          # CPU, large-v3 int8, Urdu
  python scripts/transcribe.py audio/clip.wav --language auto          # test language ID
  python scripts/transcribe.py audio/clip.wav --model models/whisper-small-int8-ov
  python scripts/transcribe.py audio/clip.wav --backend faster-whisper # CTranslate2 CPU baseline w/ VAD
  python scripts/transcribe.py audio/clip.wav --initial-prompt "Imran Khan, Nawaz Sharif, Islamabad"

  # once GPU/CUDA hardware is available:
  python scripts/transcribe.py audio/clip.wav --device GPU                          # Intel iGPU (OpenVINO)
  python scripts/transcribe.py audio/clip.wav --backend faster-whisper --device cuda # NVIDIA (CTranslate2)
"""
import argparse
import sys
import time
from pathlib import Path

from audio_io import load_audio, SAMPLE_RATE, PROJECT_ROOT
from backends import BackendLoadError, load_backend
from webvtt import format_vtt

DEFAULT_OV_MODEL = PROJECT_ROOT / "models" / "whisper-large-v3-int8-ov"
# CPU by default on both backends — no GPU/CUDA hardware assumed. Override
# with --device GPU (openvino) or --device cuda (faster-whisper) once available.
DEFAULT_DEVICE = {"openvino": "CPU", "faster-whisper": "cpu"}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--backend", choices=["openvino", "faster-whisper"], default="openvino")
    ap.add_argument("--device", default=None,
                     help="openvino: GPU/CPU/NPU/auto (default CPU); "
                          "faster-whisper: cpu/cuda/auto (default cpu); "
                          "falls back to CPU on load failure")
    ap.add_argument("--model", default=None,
                     help="openvino: model dir; faster-whisper: size name or CT2 dir (default large-v3)")
    ap.add_argument("--language", default="ur", help="ISO code, or 'auto' for detection")
    ap.add_argument("--initial-prompt", default=None,
                     help="bias decoding toward custom vocabulary (proper nouns, places, orgs)")
    ap.add_argument("-o", "--output", default=None, help="write plain transcript here")
    args = ap.parse_args()

    samples = load_audio(args.audio)
    duration = len(samples) / SAMPLE_RATE
    print(f"{args.audio}: {duration/60:.1f} min of audio")

    device = args.device or DEFAULT_DEVICE[args.backend]
    t0 = time.perf_counter()
    try:
        if args.backend == "openvino":
            model_dir = Path(args.model) if args.model else DEFAULT_OV_MODEL
            if not model_dir.exists():
                sys.exit(f"model dir not found: {model_dir} — run scripts/download_models.py first")
            backend = load_backend("openvino", model_dir=model_dir, device=device)
        else:
            backend = load_backend("faster-whisper", model=args.model or "large-v3", device=device)
    except BackendLoadError as e:
        sys.exit(f"could not load {args.backend} backend: {e}")
    print(f"model loaded/compiled on {backend.device} in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    text, chunks = backend.transcribe(samples, args.language, initial_prompt=args.initial_prompt)
    elapsed = time.perf_counter() - t0

    print()
    for c in chunks:
        print(f"[{c.start:.1f}-{c.end:.1f}] {c.text}")
    print(f"\ntranscribed {duration:.0f}s in {elapsed:.1f}s  ->  {duration/elapsed:.2f}x realtime")

    out = Path(args.output) if args.output else (
        PROJECT_ROOT / "transcripts" / (Path(args.audio).stem + f".{args.backend}.txt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"transcript -> {out}")

    if chunks:
        vtt_out = out.with_suffix(".vtt")
        vtt_out.write_text(format_vtt(chunks), encoding="utf-8")
        print(f"captions   -> {vtt_out}")


if __name__ == "__main__":
    main()
