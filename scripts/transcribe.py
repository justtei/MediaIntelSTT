"""Transcribe audio with Whisper on Intel Iris Xe (OpenVINO GPU) or CPU (faster-whisper).

Examples:
  python scripts/transcribe.py audio/clip.wav                          # iGPU, large-v3 int8, Urdu
  python scripts/transcribe.py audio/clip.wav --language auto          # test language ID
  python scripts/transcribe.py audio/clip.wav --model models/whisper-small-int8-ov
  python scripts/transcribe.py audio/clip.wav --backend faster-whisper # CPU baseline w/ VAD
"""
import argparse
import sys
import time
from pathlib import Path

from audio_io import load_audio, SAMPLE_RATE, PROJECT_ROOT

DEFAULT_OV_MODEL = PROJECT_ROOT / "models" / "whisper-large-v3-int8-ov"


def run_openvino(args, samples):
    import openvino_genai

    model_dir = Path(args.model) if args.model else DEFAULT_OV_MODEL
    if not model_dir.exists():
        sys.exit(f"model dir not found: {model_dir} — run scripts/download_models.py first")

    print(f"loading {model_dir.name} on {args.device} ...", flush=True)
    t0 = time.perf_counter()
    pipe = openvino_genai.WhisperPipeline(str(model_dir), device=args.device)
    print(f"model loaded/compiled in {time.perf_counter() - t0:.1f}s", flush=True)

    config = pipe.get_generation_config()
    config.task = "transcribe"
    config.return_timestamps = True
    if args.language != "auto":
        config.language = f"<|{args.language}|>"

    t0 = time.perf_counter()
    result = pipe.generate(samples, config)
    elapsed = time.perf_counter() - t0

    lines = []
    for chunk in result.chunks or []:
        lines.append(f"[{chunk.start_ts:.1f}-{chunk.end_ts:.1f}] {chunk.text.strip()}")
    text = str(result)
    return text, lines, elapsed


def run_faster_whisper(args, samples):
    from faster_whisper import WhisperModel

    size = args.model or "large-v3"
    print(f"loading faster-whisper {size} (CPU int8) ...", flush=True)
    model = WhisperModel(size, device="cpu", compute_type="int8")

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        samples,
        language=None if args.language == "auto" else args.language,
        vad_filter=True,
        beam_size=5,
    )
    lines, parts = [], []
    for s in segments:  # generator — consumes here, so time the loop too
        lines.append(f"[{s.start:.1f}-{s.end:.1f}] {s.text.strip()}")
        parts.append(s.text)
    elapsed = time.perf_counter() - t0
    print(f"detected language: {info.language} (p={info.language_probability:.2f})")
    return " ".join(parts).strip(), lines, elapsed


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--backend", choices=["openvino", "faster-whisper"], default="openvino")
    ap.add_argument("--device", default="GPU", help="OpenVINO device: GPU (Iris Xe), CPU, NPU")
    ap.add_argument("--model", default=None,
                    help="openvino: model dir; faster-whisper: size name (default large-v3)")
    ap.add_argument("--language", default="ur", help="ISO code, or 'auto' for detection")
    ap.add_argument("-o", "--output", default=None, help="write plain transcript here")
    args = ap.parse_args()

    samples = load_audio(args.audio)
    duration = len(samples) / SAMPLE_RATE
    print(f"{args.audio}: {duration/60:.1f} min of audio")

    if args.backend == "openvino":
        text, lines, elapsed = run_openvino(args, samples)
    else:
        text, lines, elapsed = run_faster_whisper(args, samples)

    print()
    for line in lines:
        print(line)
    print(f"\ntranscribed {duration:.0f}s in {elapsed:.1f}s  ->  {duration/elapsed:.2f}x realtime")

    out = Path(args.output) if args.output else (
        PROJECT_ROOT / "transcripts" / (Path(args.audio).stem + f".{args.backend}.txt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"transcript -> {out}")


if __name__ == "__main__":
    main()
