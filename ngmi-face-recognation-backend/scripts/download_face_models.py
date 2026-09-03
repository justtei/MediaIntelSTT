"""Download the YuNet (detection) + SFace (recognition) ONNX models from
OpenCV Zoo — Apache-2.0, actively maintained, the reference pair for
scripts/face_backend.py. Run once; re-run is a harmless no-op if the files
already exist.

Usage: python scripts/download_face_models.py
"""
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = PROJECT_ROOT / "models" / "face"

FILES = {
    "yunet.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "sface.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = DEST_DIR / name
        if dest.exists():
            print(f"{dest} already present, skipping")
            continue
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        if size < 10_000:  # a real model is hundreds of KB+; this small means an error page, not a model
            dest.unlink()
            sys.exit(f"download of {name} looks wrong (only {size} bytes) — aborting")
        print(f"  -> {size:,} bytes")
    print("done")


if __name__ == "__main__":
    main()
