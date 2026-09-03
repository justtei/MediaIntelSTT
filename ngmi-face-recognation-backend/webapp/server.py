"""Multi-channel face recognition server — live YouTube channels, any
YouTube video link, or uploaded video files. Detects and identifies
enrolled people (see faces/gallery/) in the video, broadcasting live
"who's on screen" updates to every connected dashboard, and capturing one
representative snapshot of every distinct unrecognized person.

This is a standalone extraction of the face-recognition half of a design
originally built inside a sibling speech-to-text project — no audio/speech
transcription exists in this project at all, just detection + open-set
recognition against a small enrolled photo gallery.

  1. YouTube channels — POST /channels or WS {"type":"channel_start","url":...}.
     Runs server-side, keeps recognizing even if no dashboard is open. URL is
     allow-list validated (SSRF safety, scripts/url_safety.py) before
     yt-dlp/ffmpeg ever sees it.
  2. Uploaded video — POST /upload_video (multipart file). Same pipeline,
     reading a local file instead of a live stream.
  3. Uploaded image — POST /recognize_image (multipart file). One-shot
     recognition against a static photo, no channel/stream involved.
  4. New person enrollment — POST /gallery/enroll (multipart, name + one or
     more files). Adds photos to faces/gallery/<slug>/ and merges their
     embeddings into the live gallery immediately, no restart needed.

Run:  .venv/Scripts/python webapp/server.py   then open http://127.0.0.1:8010

Key env vars (all optional):
  FACE_HOST / FACE_PORT            bind address (default 127.0.0.1:8010)
  FACE_SAMPLE_FPS                  frames/sec pulled from video for recognition (default 1.0)
  FACE_MATCH_THRESHOLD             override the calibrated default in scripts/face_backend.py
  FACE_ALLOWED_STREAM_HOSTS        comma-separated domain allow-list for channel URLs
  FACE_ALLOW_ANY_STREAM_HOST       1 to disable the allow-list entirely (dangerous — SSRF risk)
  FACE_UNKNOWN_DEDUP_THRESHOLD     cosine similarity above which an unknown face is
                                    considered "already captured" (default 0.5)
  FACE_UNKNOWN_MIN_CONFIDENCE      minimum YuNet detection confidence to bother
                                    capturing an unknown face (default 0.85)
  FACE_MAX_UPLOAD_BYTES            upload size limit (default 500 MB)
  FACE_STREAM_STALL_TIMEOUT_S      seconds of silence before a stream is considered
                                    stalled and reconnected (default 20)
  FACE_STREAM_MAX_RESTARTS         consecutive stall-reconnect attempts before giving
                                    up (default 0 = unlimited)
  FACE_LOG_LEVEL                   default INFO

Security note: the stream URL allow-list exists for the moment this stops
being localhost-only. Binding FACE_HOST to 0.0.0.0/LAN is a foot-gun without
further hardening (no auth exists here yet) — do not do it on an untrusted
network.
"""
import asyncio
import functools
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

from scripts.face_backend import FaceBackend, GALLERY_DIR, GALLERY_INDEX, load_image_bgr  # noqa: E402
from scripts.ffmpeg_util import find_ffmpeg  # noqa: E402
from scripts.url_safety import DEFAULT_ALLOWED_HOSTS, validate_stream_url  # noqa: E402

logging.basicConfig(
    level=os.environ.get("FACE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("face.server")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v else default
    except ValueError:
        log.warning("invalid %s=%r, using default %s", name, v, default)
        return default


# --- config -----------------------------------------------------------------
HOST = os.environ.get("FACE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FACE_PORT", "8010"))

FACE_SAMPLE_FPS = _env_float("FACE_SAMPLE_FPS", 1.0)
FACE_MATCH_THRESHOLD_OVERRIDE = os.environ.get("FACE_MATCH_THRESHOLD")

ALLOW_ANY_STREAM_HOST = os.environ.get("FACE_ALLOW_ANY_STREAM_HOST") == "1"
_hosts_env = os.environ.get("FACE_ALLOWED_STREAM_HOSTS")
ALLOWED_STREAM_HOSTS = (
    frozenset(h.strip().lower() for h in _hosts_env.split(",") if h.strip())
    if _hosts_env else DEFAULT_ALLOWED_HOSTS
)

UNKNOWN_FACES_DIR = PROJECT / "faces" / "unknown_faces"
UNKNOWN_FACE_DEDUP_THRESHOLD = _env_float("FACE_UNKNOWN_DEDUP_THRESHOLD", 0.5)
UNKNOWN_FACE_MIN_CAPTURE_CONFIDENCE = _env_float("FACE_UNKNOWN_MIN_CONFIDENCE", 0.85)

UPLOADS_DIR = PROJECT / "webapp" / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("FACE_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
MAX_CHANNEL_NAME_CHARS = 80
MAX_PERSON_NAME_CHARS = 80
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STREAM_RESOLVE_TIMEOUT_S = _env_float("FACE_STREAM_RESOLVE_TIMEOUT_S", 30.0)
STREAM_STALL_TIMEOUT_S = _env_float("FACE_STREAM_STALL_TIMEOUT_S", 20.0)
STREAM_MAX_RESTARTS = int(os.environ.get("FACE_STREAM_MAX_RESTARTS", "0"))  # 0 = unlimited retries

# --- face recognition backend ------------------------------------------
# Its own single-thread executor so CPU-bound detection/embedding never
# blocks the async event loop (accepting new connections, broadcasting to
# dashboards, etc).
_face_executor = ThreadPoolExecutor(max_workers=1)
_face_backend: Optional[FaceBackend] = None
_face_backend_lock = asyncio.Lock()

_unknown_faces_lock = asyncio.Lock()
_unknown_face_embeddings: list = []  # np.ndarray per already-captured unknown person
_unknown_face_counter = 0


async def _get_face_backend() -> FaceBackend:
    global _face_backend
    async with _face_backend_lock:
        if _face_backend is None:
            loop = asyncio.get_running_loop()
            log.info("loading face recognition backend ...")
            kwargs = {}
            if FACE_MATCH_THRESHOLD_OVERRIDE:
                kwargs["match_threshold"] = float(FACE_MATCH_THRESHOLD_OVERRIDE)
            backend = await loop.run_in_executor(_face_executor, functools.partial(FaceBackend, **kwargs))
            await loop.run_in_executor(_face_executor, backend.load_gallery)
            _face_backend = backend
            await loop.run_in_executor(_face_executor, _load_existing_unknown_faces, backend)
        return _face_backend


def _load_existing_unknown_faces(backend: FaceBackend) -> None:
    """Re-embed any unknown_faces/*.jpg already on disk so a restart doesn't
    re-capture duplicates of people already saved, and numbering continues
    instead of restarting at unknown_1.jpg."""
    global _unknown_face_counter
    if not UNKNOWN_FACES_DIR.is_dir():
        return
    for path in sorted(UNKNOWN_FACES_DIR.glob("unknown_*.jpg")):
        try:
            n = int(path.stem.split("_")[-1])
            _unknown_face_counter = max(_unknown_face_counter, n)
        except ValueError:
            continue
        img = load_image_bgr(path)
        if img is None:
            continue
        faces = backend.detect_raw(img)
        if faces is not None and len(faces) > 0:
            _unknown_face_embeddings.append(backend.embed(img, faces[0]))
    if _unknown_face_embeddings:
        log.info("loaded %d previously-captured unknown face(s) from %s",
                  len(_unknown_face_embeddings), UNKNOWN_FACES_DIR)


async def _capture_unknown_face(frame_bgr: np.ndarray, match, backend: FaceBackend) -> None:
    """Save one representative snapshot for a face that doesn't match the
    enrolled gallery — but only one per apparently-distinct unknown person
    (deduped against every previously-captured unknown face's embedding),
    and only from a reasonably clear/frontal detection."""
    global _unknown_face_counter
    if match.detect_confidence < UNKNOWN_FACE_MIN_CAPTURE_CONFIDENCE:
        return
    async with _unknown_faces_lock:
        for existing in _unknown_face_embeddings:
            sim = float(backend.recognizer.match(match.embedding, existing, cv2.FaceRecognizerSF_FR_COSINE))
            if sim >= UNKNOWN_FACE_DEDUP_THRESHOLD:
                return  # already have a snapshot of this person
        x, y, w, h = match.bbox
        x, y = max(0, x), max(0, y)
        crop = frame_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            return
        UNKNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        _unknown_face_counter += 1
        path = UNKNOWN_FACES_DIR / f"unknown_{_unknown_face_counter}.jpg"
        cv2.imwrite(str(path), crop)
        _unknown_face_embeddings.append(match.embedding)
        log.info("captured new unknown face -> %s", path)


async def _process_face_frame(channel_id: str, frame_bgr: np.ndarray, send) -> None:
    try:
        backend = await _get_face_backend()
    except Exception:
        log.exception("face backend unavailable, skipping frame for channel %s", channel_id)
        return
    loop = asyncio.get_running_loop()
    try:
        matches = await loop.run_in_executor(_face_executor, backend.detect, frame_bgr)
    except Exception:
        log.exception("face detection failed for channel %s", channel_id)
        return
    if not matches:
        await send({"type": "faces_seen", "faces": [],
                     "frame_w": frame_bgr.shape[1], "frame_h": frame_bgr.shape[0]})
        return
    payload = []
    for m in matches:
        payload.append({"name": m.name, "confidence": round(m.confidence, 3), "bbox": list(m.bbox)})
        if m.name == "Unknown":
            await _capture_unknown_face(frame_bgr, m, backend)
    # frame_w/h let the frontend scale bbox pixel coordinates (given in the
    # analyzed frame's own resolution) to whatever size the video is
    # actually displayed at.
    await send({"type": "faces_seen", "faces": payload,
                 "frame_w": frame_bgr.shape[1], "frame_h": frame_bgr.shape[0]})


def _slugify_person_name(name: str) -> str:
    """Turns a typed display name into the same kebab-case folder-name
    convention already used by every enrolled person under faces/gallery/
    (e.g. "Waseem Badami" -> "waseem-badami")."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:MAX_PERSON_NAME_CHARS]


def _enroll_person_images(backend: FaceBackend, slug: str, image_paths: list) -> tuple:
    """Embeds each newly-uploaded photo and merges the result into the
    live backend's gallery + persists it to disk. Must run on the face
    executor thread (the same one every detect()/identify() call runs on)
    so a concurrent recognition can never see a half-updated gallery dict.

    Reuses the exact multi-face selection heuristic from enroll_faces.py:
    when a photo has more than one detected face, keep whichever face
    scores highest on bbox-area * detection-confidence combined."""
    vectors = []
    skipped = 0
    for path in image_paths:
        img = load_image_bgr(path)
        if img is None:
            skipped += 1
            continue
        faces = backend.detect_raw(img)
        if faces is None or len(faces) == 0:
            skipped += 1
            continue
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3] * f[-1], reverse=True)
        vectors.append(backend.embed(img, faces[0]))
    if vectors:
        backend.gallery.setdefault(slug, [])
        backend.gallery[slug].extend(vectors)
        GALLERY_INDEX.parent.mkdir(parents=True, exist_ok=True)
        np.savez(GALLERY_INDEX, **{n: np.array(v) for n, v in backend.gallery.items()})
    return len(vectors), skipped, len(backend.gallery.get(slug, []))


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    try:
        await _get_face_backend()
        log.info("ready in %.1fs", time.perf_counter() - t0)
    except Exception:
        log.exception("failed to load face recognition backend")
        raise
    yield
    for channel_id in list(CHANNELS):
        await stop_channel(channel_id, reason="server_shutdown")


app = FastAPI(lifespan=lifespan)


def _deno_path() -> Optional[str]:
    import shutil
    on_path = shutil.which("deno")
    if on_path:
        return on_path
    for p in (PROJECT / "tools" / "deno").glob("**/deno*"):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


@dataclass
class ResolvedStream:
    media_url: str
    video_id: Optional[str] = None
    title: Optional[str] = None


def _resolve_media_url(url: str) -> ResolvedStream:
    """Ask yt-dlp for the direct, playable media URL plus metadata — one
    call, one network round-trip (blocking; always call via run_in_executor).
    Prefers a video-only stream (no audio needed here) to save bandwidth."""
    cmd = [sys.executable, "-m", "yt_dlp", "-j", "-f", "bestvideo/best", "--no-warnings"]
    deno = _deno_path()
    if deno:  # lets yt-dlp solve YouTube's player-JS challenges
        cmd += ["--js-runtimes", f"deno:{deno}"]
    else:
        log.warning("no deno runtime found — YouTube extraction is degraded/deprecated without "
                    "one (yt-dlp's own warning). Install: curl -fsSL https://deno.land/install.sh | sh")
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=STREAM_RESOLVE_TIMEOUT_S)
    if result.returncode != 0 or not result.stdout.strip():
        errors = [line for line in result.stderr.splitlines() if line.startswith("ERROR")]
        raise RuntimeError(errors[-1] if errors else "yt-dlp could not resolve a playable URL")
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise RuntimeError(f"yt-dlp returned unparseable output: {e}")
    media_url = info.get("url")
    if not media_url:
        raise RuntimeError("yt-dlp resolved no playable media URL")
    return ResolvedStream(media_url=media_url, video_id=info.get("id"), title=info.get("title"))


class FramePuller:
    """Pulls low-rate JPEG frames (FACE_SAMPLE_FPS, default 1/s) from a
    channel's video via ffmpeg, decodes each, and runs it through the face
    pipeline. One independent ffmpeg process per channel — a stuck/stalled
    stream for one channel can never affect another.

    `source` is either a YouTube URL (resolved here via yt-dlp) or, for an
    uploaded file, an already-usable local path (`is_local=True` skips
    resolution — ffmpeg reads a local path the same way it reads a URL).

    A resolved live-stream media URL can silently stall mid-stream (YouTube
    CDN behavior, not something this project controls). Every read is
    bounded by STREAM_STALL_TIMEOUT_S; a stall triggers an automatic
    re-resolve + reconnect (bounded by STREAM_MAX_RESTARTS, default
    unlimited) instead of leaving the channel silently dead.
    """

    def __init__(self, source: str, channel_id: str, send, is_local: bool = False):
        self.source = source
        self.channel_id = channel_id
        self.send = send
        self.is_local = is_local
        self.proc = None
        self.task = None
        self._stopped = False
        self._last_stderr_line = ""

    async def start(self):
        self.task = asyncio.ensure_future(self._run())

    async def _run(self):
        if self.is_local:
            if not Path(self.source).resolve().is_relative_to(UPLOADS_DIR.resolve()):
                await self.send({"type": "stream_status", "state": "error",
                                  "detail": "refusing to read a path outside webapp/uploads"})
                return
        else:
            rejection = validate_stream_url(self.source, ALLOWED_STREAM_HOSTS, ALLOW_ANY_STREAM_HOST)
            if rejection:
                log.warning("rejected stream URL %r: %s", self.source, rejection)
                await self.send({"type": "stream_status", "state": "error", "detail": rejection})
                return

        attempt = 0
        while True:
            if self.is_local:
                resolved = ResolvedStream(media_url=self.source)
            else:
                loop = asyncio.get_running_loop()
                try:
                    resolved = await loop.run_in_executor(None, _resolve_media_url, self.source)
                except subprocess.TimeoutExpired:
                    await self.send({"type": "stream_status", "state": "error",
                                      "detail": f"yt-dlp did not resolve a playable URL within "
                                                f"{STREAM_RESOLVE_TIMEOUT_S:.0f}s"})
                    return
                except Exception as e:
                    await self.send({"type": "stream_status", "state": "error", "detail": str(e)})
                    return

            if self._stopped:  # channel was stopped while the (unbounded-by-us) resolve was in flight
                return

            ffmpeg_cmd = [
                find_ffmpeg(), "-v", "error", "-re", "-i", resolved.media_url,
                "-vf", f"fps={FACE_SAMPLE_FPS}",
                "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5",
                "pipe:1",
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if self._stopped:  # stop() raced us between resolve and spawn
                proc.kill()
                return
            self.proc = proc
            self._last_stderr_line = ""
            stderr_task = asyncio.ensure_future(self._drain_stderr(proc))
            await self.send({"type": "stream_status", "state": "started",
                              "video_id": resolved.video_id, "title": resolved.title})

            stalled = await self._pump(proc)
            stderr_task.cancel()

            if self._stopped or not stalled:
                return

            attempt += 1
            if STREAM_MAX_RESTARTS and attempt > STREAM_MAX_RESTARTS:
                await self.send({"type": "stream_status", "state": "error",
                                  "detail": f"stream stalled repeatedly ({attempt} attempts), giving up"})
                return
            budget = str(STREAM_MAX_RESTARTS) if STREAM_MAX_RESTARTS else "unlimited"
            log.warning("stream %r stalled (no data for %.0fs), reconnecting (attempt %d/%s)",
                        self.source, STREAM_STALL_TIMEOUT_S, attempt, budget)
            await asyncio.sleep(min(2 * attempt, 10))

    async def _drain_stderr(self, proc):
        """Continuously consume ffmpeg's stderr — without this a chatty
        ffmpeg can fill the OS pipe buffer and block on the write, which
        would silently stall stdout too."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, proc.stderr.readline)
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._last_stderr_line = text
                    log.debug("ffmpeg stderr (%s): %s", self.source, text)
        except Exception:
            pass

    async def _pump(self, proc) -> bool:
        """Demux ffmpeg's raw MJPEG byte stream into individual JPEG frames
        by scanning for JPEG start/end-of-image markers, decode each, and
        run it through the face pipeline. Returns True if the stream
        stalled and the caller should reconnect, False for a clean end/error
        (a terminal stream_status has already been sent in that case)."""
        loop = asyncio.get_running_loop()
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        buf = b""
        got_data = False
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(None, proc.stdout.read, 65536),
                        timeout=STREAM_STALL_TIMEOUT_S)
                except asyncio.TimeoutError:
                    if proc.poll() is None:
                        proc.kill()
                    return True  # stalled — _run() decides whether to reconnect
                if not chunk:
                    break
                got_data = True
                buf += chunk
                while True:
                    start = buf.find(SOI)
                    if start == -1:
                        buf = b""
                        break
                    end = buf.find(EOI, start + 2)
                    if end == -1:
                        buf = buf[start:]  # incomplete frame — keep, wait for more data
                        break
                    jpg_bytes = buf[start:end + 2]
                    buf = buf[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        await _process_face_frame(self.channel_id, frame, self.send)
            detail = "" if got_data else (self._last_stderr_line or "no video received from ffmpeg")
            await self.send({"type": "stream_status",
                             "state": "ended" if got_data else "error", "detail": detail})
            return False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("frame pump failed for channel %s", self.channel_id)
            await self.send({"type": "stream_status", "state": "error", "detail": str(e)})
            return False

    def stop(self):
        self._stopped = True
        if self.task:
            self.task.cancel()
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


# --- channel registry ---------------------------------------------------
@dataclass
class Channel:
    id: str
    name: str
    source_type: str  # "youtube" | "upload"
    url: Optional[str]
    status: str = "starting"
    puller: Optional[FramePuller] = None
    created_at: float = field(default_factory=time.time)
    video_id: Optional[str] = None
    upload_path: Optional[Path] = None  # deleted on stop_channel() for "upload" channels

    def public(self) -> dict:
        return {
            "id": self.id, "name": self.name, "source_type": self.source_type,
            "url": self.url, "status": self.status, "created_at": self.created_at,
            "video_id": self.video_id,
            "preview_url": f"/uploads/{self.upload_path.name}" if self.upload_path else None,
        }


CHANNELS: dict[str, Channel] = {}
SUBSCRIBERS: set[WebSocket] = set()


async def broadcast(event: dict):
    dead = []
    for ws in SUBSCRIBERS:
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        SUBSCRIBERS.discard(ws)


_STREAM_STATE_TO_STATUS = {"started": "live", "error": "error", "ended": "stopped", "stopped": "stopped"}


def _channel_send(channel_id: str):
    """The `send` callback given to a channel's FramePuller: tags every
    event with channel_id, broadcasts to all dashboards, and keeps the
    channel registry's status in sync (so a dashboard that connects *after*
    a stream error still sees "error", not a stale "live")."""
    async def send(event: dict):
        await broadcast({**event, "channel_id": channel_id})
        channel = CHANNELS.get(channel_id)
        if channel and event.get("type") == "stream_status":
            new_status = _STREAM_STATE_TO_STATUS.get(event.get("state"))
            if new_status and new_status != channel.status:
                channel.status = new_status
            if event.get("video_id") and not channel.video_id:
                channel.video_id = event["video_id"]
    return send


async def start_channel(name: str, source_type: str, url: str,
                         upload_path: Optional[Path] = None) -> Channel:
    if source_type not in ("youtube", "upload"):
        raise ValueError(f"unknown source: {source_type!r} (expected 'youtube' or 'upload')")
    name = (name or "").strip()[:MAX_CHANNEL_NAME_CHARS] or "Channel"
    if source_type == "youtube":
        rejection = validate_stream_url(url or "", ALLOWED_STREAM_HOSTS, ALLOW_ANY_STREAM_HOST)
        if rejection:
            raise ValueError(rejection)

    channel_id = str(uuid.uuid4())
    send = _channel_send(channel_id)
    channel = Channel(id=channel_id, name=name, source_type=source_type, url=url, upload_path=upload_path)
    CHANNELS[channel_id] = channel

    # Broadcast immediately rather than after URL resolution (which can take
    # up to STREAM_RESOLVE_TIMEOUT_S) so the UI shows feedback right away.
    await broadcast({"type": "channel_added", "channel": channel.public()})

    puller = FramePuller(url, channel_id, send, is_local=(source_type == "upload"))
    channel.puller = puller
    asyncio.ensure_future(puller.start())  # status updates arrive via stream_status broadcasts

    return channel


async def stop_channel(channel_id: str, reason: str = "stopped"):
    channel = CHANNELS.pop(channel_id, None)
    if not channel:
        return
    if channel.puller:
        channel.puller.stop()
    channel.status = "stopped"
    if channel.upload_path and channel.upload_path.exists():
        try:
            channel.upload_path.unlink()
        except OSError:
            log.exception("failed to remove uploaded file %s", channel.upload_path)
    await broadcast({"type": "channel_removed", "channel_id": channel_id, "reason": reason})


# --- HTTP API -------------------------------------------------------------
class ChannelCreateRequest(BaseModel):
    name: str
    url: str


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    ready = _face_backend is not None
    body = {
        "status": "ok" if ready else "loading",
        "active_channels": len(CHANNELS),
        "gallery": {} if _face_backend is None else {k: len(v) for k, v in _face_backend.gallery.items()},
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.get("/uploads/{filename}")
def serve_upload(filename: str):
    """Serves an uploaded video back to the browser for local preview
    playback. `filename` is resolved strictly under UPLOADS_DIR and checked
    against path traversal — it's the uuid-based name we generated on
    upload, never the client's original filename."""
    path = (UPLOADS_DIR / filename).resolve()
    if not path.is_relative_to(UPLOADS_DIR.resolve()) or not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path)


@app.get("/channels")
def list_channels_endpoint():
    return [c.public() for c in CHANNELS.values()]


@app.post("/channels")
async def create_channel_endpoint(req: ChannelCreateRequest):
    try:
        channel = await start_channel(req.name, "youtube", req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return channel.public()


@app.delete("/channels/{channel_id}")
async def delete_channel_endpoint(channel_id: str):
    if channel_id not in CHANNELS:
        raise HTTPException(404, "channel not found")
    await stop_channel(channel_id, reason="stopped_by_user")
    return {"status": "stopped"}


ALLOWED_UPLOAD_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


@app.post("/upload_video")
async def upload_video_endpoint(file: UploadFile = File(...), name: str = Form("Uploaded video")):
    """Accepts a local video file, saves it under webapp/uploads/, and
    starts it as a channel exactly like a live YouTube URL would be — same
    FramePuller pipeline, just reading a local path (is_local=True skips
    yt-dlp resolution — ffmpeg reads a local path the same way it reads a URL)."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(400, f"unsupported file type {ext!r} (allowed: {sorted(ALLOWED_UPLOAD_EXTS)})")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / f"{uuid.uuid4()}{ext}"
    written = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"upload failed: {e}")

    try:
        channel = await start_channel(name, "upload", str(dest), upload_path=dest)
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    return channel.public()


@app.post("/gallery/enroll")
async def enroll_person_endpoint(name: str = Form(...), files: list[UploadFile] = File(...)):
    """Dynamically adds a new person to the recognition gallery — no server
    restart, no re-running enroll_faces.py by hand. Saves the uploaded
    photos under faces/gallery/<slug>/, embeds them, and merges the result
    into the live in-memory gallery + faces/gallery_index.npz so the next
    detect() call already recognizes them."""
    slug = _slugify_person_name(name)
    if not slug:
        raise HTTPException(400, "name must contain at least one letter or digit")
    if not files:
        raise HTTPException(400, "at least one image is required")

    backend = await _get_face_backend()
    person_dir = GALLERY_DIR / slug
    person_dir.mkdir(parents=True, exist_ok=True)
    start_index = len(list(person_dir.iterdir()))

    saved_paths = []
    try:
        for i, f in enumerate(files):
            ext = Path(f.filename or "").suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            data = await f.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
            dest = person_dir / f"{start_index + i + 1}{ext}"
            dest.write_bytes(data)
            saved_paths.append(dest)
    except HTTPException:
        for p in saved_paths:
            p.unlink(missing_ok=True)
        raise

    if not saved_paths:
        raise HTTPException(400, f"no usable images (allowed: {sorted(IMAGE_EXTS)})")

    loop = asyncio.get_running_loop()
    enrolled, skipped, total = await loop.run_in_executor(
        _face_executor, _enroll_person_images, backend, slug, saved_paths)

    if enrolled == 0:
        for p in saved_paths:
            p.unlink(missing_ok=True)
        raise HTTPException(400, "no face detected in any uploaded image")

    log.info("enrolled %d/%d new photo(s) for %r (%d total)", enrolled, len(saved_paths), slug, total)
    return {
        "name": slug,
        "images_uploaded": len(saved_paths),
        "images_enrolled": enrolled,
        "images_skipped": skipped,
        "total_enrolled_photos": total,
    }


@app.post("/recognize_image")
async def recognize_image_endpoint(file: UploadFile = File(...)):
    """Recognizes every enrolled person in a single uploaded image — the
    same detect()+identify() pipeline the video pipeline uses, just run
    once against a static image instead of a stream of frames."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"unsupported image type {ext!r} (allowed: {sorted(IMAGE_EXTS)})")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "could not decode image")

    backend = await _get_face_backend()
    loop = asyncio.get_running_loop()
    matches = await loop.run_in_executor(_face_executor, backend.detect, frame)

    faces = [{"name": m.name, "confidence": round(m.confidence, 3), "bbox": list(m.bbox)} for m in matches]
    return {"faces": faces, "frame_w": frame.shape[1], "frame_h": frame.shape[0]}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    SUBSCRIBERS.add(ws)

    async def send(obj):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            pass  # client went away mid-send; the receive loop will close us

    await send({"type": "channels_snapshot", "channels": [c.public() for c in CHANNELS.values()]})

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await send({"type": "error", "detail": "malformed JSON message"})
                    continue
                if not isinstance(data, dict):
                    await send({"type": "error", "detail": "message must be a JSON object"})
                    continue
                kind = data.get("type")

                if kind == "channel_start":
                    try:
                        await start_channel(data.get("name", "Channel"), "youtube", data.get("url", ""))
                    except ValueError as e:
                        await send({"type": "error", "detail": str(e)})

                elif kind == "channel_stop":
                    cid = data.get("channel_id")
                    if cid:
                        await stop_channel(cid, reason="stopped_by_user")
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        log.exception("unhandled error in /ws connection")
    finally:
        SUBSCRIBERS.discard(ws)


if __name__ == "__main__":
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        log.warning("FACE_HOST=%s exposes this server beyond localhost with no auth in place — "
                     "anyone who can reach it can pull arbitrary allow-listed URLs through your "
                     "server and upload files to it. Don't do this on an untrusted network.", HOST)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
