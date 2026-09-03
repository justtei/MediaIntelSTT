# NGMI Face Recognition

Standalone open-set face recognition for live video — no speech-to-text,
no audio pipeline, just: detect faces in a live YouTube channel, any
YouTube video link, or an uploaded video file, and identify enrolled
people against a small photo gallery. Extracted from a sibling
speech-to-text project's face-recognition feature into its own project.

## Approach

Not a trained classifier — a few-shot, open-set embedding match:

- **Detector:** YuNet (OpenCV Zoo, ONNX, Apache-2.0)
- **Embedder:** SFace (OpenCV Zoo, ONNX, Apache-2.0)

Both are pretrained, frozen networks, run via OpenCV's own `cv2.dnn` APIs
(`FaceDetectorYN`, `FaceRecognizerSF`) rather than a hand-rolled ONNX
decode — these two models need non-trivial, model-specific alignment/
post-processing, and OpenCV ships the tested reference implementation for
exactly this pair.

"Training" = enrollment: a person's photos get embedded once and stored in
a small gallery index (`faces/gallery_index.npz`). A detected face is
identified by nearest-neighbor cosine similarity against that gallery;
below a calibrated threshold, the answer is "Unknown" — there's no
retraining step to add a new person, just dropping in a few more photos
and re-running one script.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_face_models.py
```

Then populate `faces/gallery/<person-name>/*.jpg` (3-10 varied photos per
person; kebab-case folder names) and enroll:

```powershell
.venv\Scripts\python scripts\enroll_faces.py --calibrate
```

`--calibrate` prints same-person vs. different-person similarity stats
from your actual data — use it to sanity-check (or override via
`FACE_MATCH_THRESHOLD`) the default threshold in `scripts/face_backend.py`
any time the gallery changes meaningfully.

ffmpeg (and, for reliable YouTube extraction, `deno`) are expected under
`tools/` or on `PATH` — see `scripts/ffmpeg_util.py`.

## Run

```powershell
.venv\Scripts\python webapp\server.py
```

Then open http://127.0.0.1:8010. Add a live YouTube channel or any
YouTube video link, or upload a local video file — the dashboard shows a
live "who's on screen" chip list per channel. Faces that don't match the
gallery get one representative snapshot saved to
`faces/unknown_faces/unknown_N.jpg` (deduplicated by embedding similarity,
so the same unrecognized person isn't saved repeatedly).

Key env vars (all optional): `FACE_HOST`/`FACE_PORT` (default
127.0.0.1:8010), `FACE_SAMPLE_FPS` (default 1.0), `FACE_MATCH_THRESHOLD`,
`FACE_UNKNOWN_DEDUP_THRESHOLD`, `FACE_UNKNOWN_MIN_CONFIDENCE`,
`FACE_MAX_UPLOAD_BYTES`, `FACE_ALLOWED_STREAM_HOSTS` /
`FACE_ALLOW_ANY_STREAM_HOST`, `FACE_STREAM_STALL_TIMEOUT_S` /
`FACE_STREAM_MAX_RESTARTS` — see the module docstring in
`webapp/server.py` for the full list.

## Layout

```
scripts/
  face_backend.py          detection + recognition + gallery matching
  enroll_faces.py           builds faces/gallery_index.npz from faces/gallery/
  download_face_models.py   fetches yunet.onnx / sface.onnx
  ffmpeg_util.py             locates ffmpeg (PATH or tools/)
  url_safety.py              SSRF-safe allow-list for channel URLs
webapp/
  server.py                  FastAPI + WebSocket server, FramePuller
  static/index.html          dashboard UI
faces/
  gallery/<person>/*.jpg     enrollment photos (not tracked in git — real
                              people's photos; supply your own or copy from
                              the sibling STT project's faces/gallery/)
  gallery_index.npz          built by enroll_faces.py
  unknown_faces/              auto-captured, deduplicated unknown sightings
models/face/                 yunet.onnx, sface.onnx (not tracked in git —
                              fetched by download_face_models.py)
```

## Known limitations

- Broadcast/live video is a harder domain than clean reference photos —
  compression, motion blur, small/angled faces all reduce confidence.
  Mixing real video screen-grabs into the enrollment gallery (not just
  posed photos) narrows this gap.
- Look-alikes and relatives can be confused at the embedding-similarity
  level — this is an assistive signal, not a legal-grade biometric system.
- `FramePuller` has stall-detection + auto-reconnect for live channels
  (mirroring the same hardening done in the sibling STT project), but no
  authentication exists yet — do not bind `FACE_HOST` beyond localhost on
  an untrusted network.
