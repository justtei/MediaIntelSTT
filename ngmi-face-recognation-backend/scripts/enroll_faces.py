"""Build the face-recognition gallery from faces/gallery/<person>/*.{jpg,png}.

This is the "training" step for face recognition in this project — except
there's no actual model training involved: YuNet/SFace are pretrained,
frozen networks. "Training on these images" means computing each enrolled
photo's face embedding once and saving it to a small gallery index. Add a
new person or more photos later by dropping images into a new/existing
faces/gallery/<name>/ folder and re-running this script — no retraining,
no GPU, finishes in seconds.

If a gallery photo has more than one detected face (a crowd shot, or a
framed photo-of-a-photo in the background), the enrolled subject is assumed
to be whichever face scores highest on bbox-area * confidence combined —
neither signal alone proved reliable against this project's own gallery
(see the comment at the sort call below for the two real cases that
falsified each one individually).

Usage:
  python scripts/enroll_faces.py                 # build faces/gallery_index.npz
  python scripts/enroll_faces.py --calibrate      # also print similarity stats
                                                   # to sanity-check the match
                                                   # threshold in face_backend.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_backend import (  # noqa: E402
    FaceBackend, GALLERY_DIR, GALLERY_INDEX, load_image_bgr,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def enroll(backend: FaceBackend, gallery_dir: Path = GALLERY_DIR) -> dict[str, list[np.ndarray]]:
    gallery: dict[str, list[np.ndarray]] = {}
    if not gallery_dir.is_dir():
        sys.exit(f"{gallery_dir} not found")

    for person_dir in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
        name = person_dir.name
        images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"  {name}: no images found, skipping")
            continue
        vectors = []
        for img_path in images:
            img = load_image_bgr(img_path)
            if img is None:
                print(f"  {name}/{img_path.name}: unreadable, skipped")
                continue
            faces = backend.detect_raw(img)
            if faces is None or len(faces) == 0:
                print(f"  {name}/{img_path.name}: no face detected, skipped")
                continue
            if len(faces) > 1:
                # area * confidence: neither signal alone is reliable on its
                # own (verified against two real cases in this project's
                # gallery) — area alone picked a blurry framed photo in the
                # background over the real subject once; confidence alone
                # picked a small bystander over the real (larger, barely
                # lower-confidence) subject once. The product correctly
                # resolved both, plus two other multi-face photos where
                # either signal alone would also have worked.
                faces = sorted(faces, key=lambda f: f[2] * f[3] * f[-1], reverse=True)
                print(f"  {name}/{img_path.name}: {len(faces)} faces detected, using area*confidence")
            vectors.append(backend.embed(img, faces[0]))
        if not vectors:
            print(f"  {name}: 0/{len(images)} images usable — not enrolled")
            continue
        print(f"  {name}: {len(vectors)}/{len(images)} images enrolled")
        gallery[name] = vectors
    return gallery


def calibrate(gallery: dict[str, list[np.ndarray]], backend: FaceBackend) -> None:
    """Print same-person vs. different-person similarity distributions from
    the actual enrolled data, to sanity-check DEFAULT_MATCH_THRESHOLD in
    face_backend.py against real numbers instead of a memorized default."""
    import cv2

    def sim(a, b):
        return float(backend.recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))

    same, diff = [], []
    names = list(gallery.keys())
    for name in names:
        vecs = gallery[name]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                same.append(sim(vecs[i], vecs[j]))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for a in gallery[names[i]]:
                for b in gallery[names[j]]:
                    diff.append(sim(a, b))

    def stats(label, vals):
        if not vals:
            print(f"  {label}: (no pairs)")
            return
        arr = np.array(vals)
        print(f"  {label}: n={len(arr)} min={arr.min():.3f} mean={arr.mean():.3f} "
              f"max={arr.max():.3f} p5={np.percentile(arr, 5):.3f} p95={np.percentile(arr, 95):.3f}")

    print("\n--- calibration (same-person vs. different-person cosine similarity) ---")
    stats("same-person pairs", same)
    stats("different-person pairs", diff)
    if same and diff:
        # A reasonable threshold sits between the highest different-person
        # similarity and the lowest same-person similarity, if they don't overlap.
        diff_max, same_min = max(diff), min(same)
        if diff_max < same_min:
            print(f"  clean separation: different-person max={diff_max:.3f} < same-person min={same_min:.3f}")
            suggestion = (diff_max + same_min) / 2
        else:
            print(f"  overlap: different-person max={diff_max:.3f} >= same-person min={same_min:.3f} "
                  "(some pair is ambiguous — expected with only a few people enrolled)")
            suggestion = diff_max + 0.02
        print(f"  suggested threshold: ~{suggestion:.3f} "
              f"(current DEFAULT_MATCH_THRESHOLD in face_backend.py: see that file)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="also print similarity stats for threshold tuning")
    ap.add_argument("--gallery-dir", default=str(GALLERY_DIR))
    ap.add_argument("--out", default=str(GALLERY_INDEX))
    args = ap.parse_args()

    backend = FaceBackend()
    print(f"enrolling from {args.gallery_dir} ...")
    gallery = enroll(backend, Path(args.gallery_dir))

    if not gallery:
        sys.exit("nothing enrolled — check faces/gallery/<person>/ has readable photos with detectable faces")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{name: np.array(vecs) for name, vecs in gallery.items()})
    total = sum(len(v) for v in gallery.values())
    print(f"\nsaved {len(gallery)} people, {total} embeddings -> {out}")

    if args.calibrate:
        calibrate(gallery, backend)


if __name__ == "__main__":
    main()
