"""Face detection + open-set recognition, shared by the webapp's
live-channel/video-upload pipeline and the enrollment/test scripts.

  - Detector: YuNet (OpenCV Zoo, ONNX, Apache-2.0)
  - Embedder:  SFace (OpenCV Zoo, ONNX, Apache-2.0)

Both run via OpenCV's own `cv2.dnn`-based APIs (`FaceDetectorYN`,
`FaceRecognizerSF`) rather than loaded as raw ONNX through a generic
inference runtime. These two models need non-trivial, model-specific
post-processing to use correctly (YuNet's multi-scale anchor decoding +
NMS, SFace's 5-point-landmark face alignment before embedding), and OpenCV
ships the tested reference implementation for exactly that pair —
hand-decoding it against the raw ONNX graph would risk subtle correctness
bugs for no real benefit.

Recognition is few-shot / open-set, not a trained classifier: a person is
"enrolled" by storing their face embedding(s) (see enroll_faces.py), and a
detected face is identified by nearest-neighbor cosine similarity against
the gallery. Below `match_threshold`, the answer is "Unknown" — there is no
retraining step to add a new person, just re-running enrollment.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTOR_MODEL = PROJECT_ROOT / "models" / "face" / "yunet.onnx"
RECOGNIZER_MODEL = PROJECT_ROOT / "models" / "face" / "sface.onnx"
GALLERY_DIR = PROJECT_ROOT / "faces" / "gallery"
GALLERY_INDEX = PROJECT_ROOT / "faces" / "gallery_index.npz"

# SFace compares embeddings with cosine similarity; higher = more alike.
# Calibrated empirically against the enrolled gallery (40 photos, 4 people)
# via `python scripts/enroll_faces.py --calibrate`, rather than taken from a
# README default: every same-person pair scored >= 0.508, every
# different-person pair scored <= 0.434 — clean separation, no overlap — so
# 0.47 sits in the gap. Re-run --calibrate and adjust this after adding more
# people; a bigger gallery is more likely to produce look-alike pairs that
# need a different cutoff.
DEFAULT_MATCH_THRESHOLD = 0.47

DETECT_INPUT_SIZE = (320, 320)
DETECT_SCORE_THRESHOLD = 0.7
DETECT_NMS_THRESHOLD = 0.3
DETECT_TOP_K = 5000


class FaceBackendError(RuntimeError):
    """Raised when the face models or gallery can't be loaded."""


@dataclass
class FaceMatch:
    name: str                    # enrolled person's name, or "Unknown"
    confidence: float            # cosine similarity to the best gallery match
    bbox: tuple                  # (x, y, w, h) in the source frame, ints
    detect_confidence: float     # YuNet's own detection confidence (not gallery similarity) —
                                  # a proxy for "clear, frontal, well-lit face", used to gate
                                  # which frame is good enough to save for an unknown person
    embedding: np.ndarray = field(repr=False)  # for unknown-face dedup upstream


class FaceBackend:
    """Wraps YuNet + SFace. Create one instance per worker thread — cv2.dnn
    Net objects aren't documented as safe for concurrent inference calls
    across threads."""

    def __init__(self, detector_model=DETECTOR_MODEL, recognizer_model=RECOGNIZER_MODEL,
                 match_threshold: float = DEFAULT_MATCH_THRESHOLD):
        detector_model, recognizer_model = Path(detector_model), Path(recognizer_model)
        if not detector_model.exists() or not recognizer_model.exists():
            raise FaceBackendError(
                f"face models not found at {detector_model} / {recognizer_model} — "
                f"run 'python scripts/download_face_models.py' first")
        self.detector = cv2.FaceDetectorYN_create(
            str(detector_model), "", DETECT_INPUT_SIZE,
            DETECT_SCORE_THRESHOLD, DETECT_NMS_THRESHOLD, DETECT_TOP_K)
        self.recognizer = cv2.FaceRecognizerSF_create(str(recognizer_model), "")
        self.match_threshold = match_threshold
        self.gallery: dict[str, list[np.ndarray]] = {}

    def load_gallery(self, index_path=GALLERY_INDEX) -> None:
        index_path = Path(index_path)
        if not index_path.exists():
            raise FaceBackendError(f"{index_path} not found — run scripts/enroll_faces.py first")
        data = np.load(index_path, allow_pickle=True)
        self.gallery = {name: list(vectors) for name, vectors in data.items()}
        logger.info("loaded face gallery: %s", {k: len(v) for k, v in self.gallery.items()})

    def embed(self, frame_bgr: np.ndarray, face) -> np.ndarray:
        """`face` is one row from detect_raw()'s output (bbox + 5 landmarks)."""
        aligned = self.recognizer.alignCrop(frame_bgr, face)
        return self.recognizer.feature(aligned)

    def detect_raw(self, frame_bgr: np.ndarray):
        """Low-level: just detection, no recognition. Returns YuNet's raw
        Nx15 array (bbox, 5 landmarks, confidence) or None."""
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        return faces

    def detect(self, frame_bgr: np.ndarray) -> list[FaceMatch]:
        """Detect every face in a frame and identify each against the loaded
        gallery. Returns [] if no faces found. Call load_gallery() first."""
        faces = self.detect_raw(frame_bgr)
        if faces is None:
            return []
        results = []
        for f in faces:
            bbox = tuple(int(v) for v in f[:4])
            embedding = self.embed(frame_bgr, f)
            name, sim = self.identify(embedding)
            results.append(FaceMatch(name=name, confidence=sim, bbox=bbox,
                                      detect_confidence=float(f[-1]), embedding=embedding))
        return results

    def identify(self, embedding: np.ndarray) -> tuple[str, float]:
        """Match one embedding against the gallery. Returns (name, similarity),
        name == "Unknown" if nothing clears match_threshold."""
        best_name, best_sim = "Unknown", 0.0
        for name, vectors in self.gallery.items():
            for gv in vectors:
                sim = float(self.recognizer.match(embedding, gv, cv2.FaceRecognizerSF_FR_COSINE))
                if sim > best_sim:
                    best_name, best_sim = name, sim
        if best_sim < self.match_threshold:
            return "Unknown", best_sim
        return best_name, best_sim


def load_image_bgr(path) -> Optional[np.ndarray]:
    """cv2.imread returns None (not an exception) on a bad/unreadable file —
    callers must check, this just centralizes that footgun in one place."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("could not read image: %s", path)
    return img
