"""End-to-end sanity check: OpenVINO sees the Iris Xe iGPU and the Whisper
pipeline compiles and runs on it. Uses 3s of generated audio — the point is
proving the GPU path works, not transcript quality.

Usage: python scripts/smoke_test.py [model_dir]
"""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    import openvino

    core = openvino.Core()
    devices = core.available_devices
    print("OpenVINO devices:", devices)
    for d in devices:
        print(f"  {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
    if not any(d.startswith("GPU") for d in devices):
        sys.exit("FAIL: no GPU device — check Intel graphics driver")

    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PROJECT_ROOT / "models" / "whisper-large-v3-int8-ov")
    if not model_dir.exists():
        sys.exit(f"model dir missing: {model_dir} — run scripts/download_models.py")

    import openvino_genai
    print(f"\ncompiling {model_dir.name} for GPU (first time takes a few minutes)...")
    t0 = time.perf_counter()
    pipe = openvino_genai.WhisperPipeline(str(model_dir), device="GPU")
    print(f"compiled in {time.perf_counter() - t0:.1f}s")

    # 3 seconds of quiet noise — enough to exercise encoder+decoder on the iGPU.
    audio = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.005).astype(np.float32)
    config = pipe.get_generation_config()
    config.language = "<|ur|>"
    config.task = "transcribe"
    t0 = time.perf_counter()
    result = pipe.generate(audio, config)
    print(f"generate ran in {time.perf_counter() - t0:.1f}s, output: {str(result)!r}")
    print("\nSMOKE TEST PASS — Whisper is running on the Iris Xe iGPU")


if __name__ == "__main__":
    main()
