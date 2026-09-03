"""Multi-channel Urdu OCR server -- live YouTube channels, any YouTube
video link, or uploaded video files. Periodically crops each channel's
video to its news-ticker regions (bottom crawl + side sticker) and reads
the Urdu text with a small vision-language model fine-tuned specifically
on real Nastaliq broadcast fonts, broadcasting live updates to every
connected dashboard.

This is a standalone sibling of a face-recognition project built the same
way -- no face detection/recognition exists in this project at all, just
detecting-by-cropping and reading Urdu ticker text.

  1. YouTube channels -- POST /channels or WS {"type":"channel_start","url":...}.
     Runs server-side, keeps reading even if no dashboard is open. URL is
     allow-list validated (SSRF safety, scripts/url_safety.py) before
     yt-dlp/ffmpeg ever sees it.
  2. Uploaded video -- POST /upload_video (multipart file). Same pipeline,
     reading a local file instead of a live stream.

Run:  .venv/Scripts/python webapp/server.py   then open http://127.0.0.1:8020

IMPORTANT -- this is slow by nature, not by bug: the OCR model needs real
per-image inference time (seconds on a real GPU, tens of seconds per crop
on CPU -- see scripts/ocr_backend.py's docstring for why a lighter/faster
model isn't a simple swap). OCR_SAMPLE_INTERVAL_S is deliberately coarse
(news tickers don't change every second), and every channel's OCR calls
share one single-worker queue (the model isn't safe for concurrent calls),
so running several channels at once multiplies end-to-end latency rather
than parallelizing it. Treat this as a periodic ticker-headline digest, not
a real-time feed.

Key env vars (all optional):
  OCR_HOST / OCR_PORT              bind address (default 127.0.0.1:8020)
  OCR_SAMPLE_INTERVAL_S             seconds between ticker reads per channel (default 45)
  OCR_DEVICE                        "cpu" or "cuda:0" (default cpu)
  OCR_REGIONS                       comma-separated subset of "bottom,side" (default both)
  OCR_BOTTOM_Y0 / OCR_BOTTOM_Y1     bottom-ticker crop, as a fraction of frame height (default 0.82-1.0)
  OCR_SIDE_X0 / OCR_SIDE_X1         side-ticker crop, as a fraction of frame width (default 0.78-1.0)
  OCR_ALLOWED_STREAM_HOSTS          comma-separated domain allow-list for channel URLs
  OCR_ALLOW_ANY_STREAM_HOST         1 to disable the allow-list entirely (dangerous -- SSRF risk)
  OCR_MAX_UPLOAD_BYTES              upload size limit (default 500 MB)
  OCR_STREAM_STALL_TIMEOUT_S        seconds of silence before a stream is considered
                                      stalled and reconnected (default 20)
  OCR_STREAM_MAX_RESTARTS           consecutive stall-reconnect attempts before giving
                                      up (default 0 = unlimited)
  OCR_LOG_LEVEL                     default INFO

Security note: the stream URL allow-list exists for the moment this stops
being localhost-only. Binding OCR_HOST to 0.0.0.0/LAN is a foot-gun without
further hardening (no auth exists here yet) -- do not do it on an
untrusted network.
"""
import asyncio
import functools
import json
import logging
import os
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

from scripts.ocr_backend import OcrBackend  # noqa: E402
from scripts.ffmpeg_util import find_ffmpeg  # noqa: E402
from scripts.url_safety import DEFAULT_ALLOWED_HOSTS, validate_stream_url  # noqa: E402

logging.basicConfig(
    level=os.environ.get("OCR_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ocr.server")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v else default
    except ValueError:
        log.warning("invalid %s=%r, using default %s", name, v, default)
        return default


# --- config -----------------------------------------------------------------
HOST = os.environ.get("OCR_HOST", "127.0.0.1")
PORT = int(os.environ.get("OCR_PORT", "8020"))

OCR_SAMPLE_INTERVAL_S = _env_float("OCR_SAMPLE_INTERVAL_S", 45.0)
OCR_DEVICE = os.environ.get("OCR_DEVICE", "cpu")

_regions_env = os.environ.get("OCR_REGIONS", "bottom,side")
OCR_REGIONS = [r.strip() for r in _regions_env.split(",") if r.strip() in ("bottom", "side")]

OCR_BOTTOM_Y0 = _env_float("OCR_BOTTOM_Y0", 0.82)
OCR_BOTTOM_Y1 = _env_float("OCR_BOTTOM_Y1", 1.0)
OCR_SIDE_X0 = _env_float("OCR_SIDE_X0", 0.78)
OCR_SIDE_X1 = _env_float("OCR_SIDE_X1", 1.0)

ALLOW_ANY_STREAM_HOST = os.environ.get("OCR_ALLOW_ANY_STREAM_HOST") == "1"
_hosts_env = os.environ.get("OCR_ALLOWED_STREAM_HOSTS")
ALLOWED_STREAM_HOSTS = (
    frozenset(h.strip().lower() for h in _hosts_env.split(",") if h.strip())
    if _hosts_env else DEFAULT_ALLOWED_HOSTS
)

UPLOADS_DIR = PROJECT / "webapp" / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("OCR_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
MAX_CHANNEL_NAME_CHARS = 80

STREAM_RESOLVE_TIMEOUT_S = _env_float("OCR_STREAM_RESOLVE_TIMEOUT_S", 30.0)
# Must comfortably exceed the gap between frames or every stream looks
# "stalled" before its first sample ever arrives -- at a 45s default
# sample interval, a 20s timeout (fine for the sibling face-recognition
# project's 1s interval) would reconnect-loop forever without ever
# receiving a frame.
STREAM_STALL_TIMEOUT_S = _env_float("OCR_STREAM_STALL_TIMEOUT_S", max(60.0, OCR_SAMPLE_INTERVAL_S * 2))
STREAM_MAX_RESTARTS = int(os.environ.get("OCR_STREAM_MAX_RESTARTS", "0"))  # 0 = unlimited retries

# --- OCR backend --------------------------------------------------------
# Its own single-thread executor so a slow OCR call never blocks the async
# event loop (accepting new connections, broadcasting to dashboards, etc) --
# and, just as importantly, so two channels can never call the model
# concurrently (generate() isn't documented thread-safe).
_ocr_executor = ThreadPoolExecutor(max_workers=1)
_ocr_backend: Optional[OcrBackend] = None
_ocr_backend_lock = asyncio.Lock()


async def _get_ocr_backend() -> OcrBackend:
    global _ocr_backend
    async with _ocr_backend_lock:
        if _ocr_backend is None:
            loop = asyncio.get_running_loop()
            log.info("loading OCR backend (this downloads ~4.5GB on first run) ...")
            _ocr_backend = await loop.run_in_executor(
                _ocr_executor, functools.partial(OcrBackend, device=OCR_DEVICE))
        return _ocr_backend


def _crop_region(frame_bgr: np.ndarray, region: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if region == "bottom":
        y0, y1 = int(h * OCR_BOTTOM_Y0), int(h * OCR_BOTTOM_Y1)
        return frame_bgr[y0:y1, 0:w]
    if region == "side":
        x0, x1 = int(w * OCR_SIDE_X0), int(w * OCR_SIDE_X1)
        return frame_bgr[0:h, x0:x1]
    raise ValueError(f"unknown region {region!r}")


async def _process_ocr_frame(channel_id: str, frame_bgr: np.ndarray, send) -> None:
    try:
        backend = await _get_ocr_backend()
    except Exception:
        log.exception("OCR backend unavailable, skipping frame for channel %s", channel_id)
        return
    loop = asyncio.get_running_loop()
    for region in OCR_REGIONS:
        crop = _crop_region(frame_bgr, region)
        if crop.size == 0:
            continue
        await send({"type": "ocr_reading", "region": region})
        try:
            text = await loop.run_in_executor(_ocr_executor, backend.read_text, crop)
        except Exception:
            log.exception("OCR failed for channel %s region %s", channel_id, region)
            continue
        await send({
            "type": "ocr_text", "region": region, "text": text,
            "read_at": time.time(),
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    try:
        await _get_ocr_backend()
        log.info("ready in %.1fs", time.perf_counter() - t0)
    except Exception:
        log.exception("failed to load OCR backend")
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
    """Ask yt-dlp for the direct, playable media URL plus metadata -- one
    call, one network round-trip (blocking; always call via run_in_executor).
    Prefers a video-only stream (no audio needed here) to save bandwidth."""
    cmd = [sys.executable, "-m", "yt_dlp", "-j", "-f", "bestvideo/best", "--no-warnings"]
    deno = _deno_path()
    if deno:  # lets yt-dlp solve YouTube's player-JS challenges
        cmd += ["--js-runtimes", f"deno:{deno}"]
    else:
        log.warning("no deno runtime found -- YouTube extraction is degraded/deprecated without "
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
    """Pulls one JPEG frame every OCR_SAMPLE_INTERVAL_S from a channel's
    video via ffmpeg, decodes it, and runs it through the OCR pipeline. One
    independent ffmpeg process per channel -- a stuck/stalled stream for one
    channel can never affect another.

    `source` is either a YouTube URL (resolved here via yt-dlp) or, for an
    uploaded file, an already-usable local path (`is_local=True` skips
    resolution -- ffmpeg reads a local path the same way it reads a URL).

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

            fps = 1.0 / OCR_SAMPLE_INTERVAL_S
            # -re paces ffmpeg's *reads* to wall-clock real time -- correct
            # for a live network stream (which is already rate-limited by
            # the encoder on the other end regardless), but actively wrong
            # for a local upload: with fps=1/45, it would force a ~45s
            # real-time wait before the very first sampled frame is ready,
            # instead of just decoding through the file as fast as the disk
            # and CPU allow.
            ffmpeg_cmd = [find_ffmpeg(), "-v", "error"]
            if not self.is_local:
                ffmpeg_cmd.append("-re")
            ffmpeg_cmd += [
                "-i", resolved.media_url,
                "-vf", f"fps={fps}",
                "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3",
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
        """Continuously consume ffmpeg's stderr -- without this a chatty
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
        run it through the OCR pipeline (which is slow -- see module
        docstring -- so frames from a fast-reconnecting or long-stalled
        stream will simply queue up and be processed with growing lag
        rather than dropped). Returns True if the stream stalled and the
        caller should reconnect, False for a clean end/error (a terminal
        stream_status has already been sent in that case)."""
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
                    return True  # stalled -- _run() decides whether to reconnect
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
                        buf = buf[start:]  # incomplete frame -- keep, wait for more data
                        break
                    jpg_bytes = buf[start:end + 2]
                    buf = buf[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        await _process_ocr_frame(self.channel_id, frame, self.send)
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
    ready = _ocr_backend is not None
    body = {
        "status": "ok" if ready else "loading",
        "active_channels": len(CHANNELS),
        "regions": OCR_REGIONS,
        "sample_interval_s": OCR_SAMPLE_INTERVAL_S,
        "device": OCR_DEVICE,
        "region_boxes": {
            "bottom": {"x0": 0.0, "y0": OCR_BOTTOM_Y0, "x1": 1.0, "y1": OCR_BOTTOM_Y1},
            "side": {"x0": OCR_SIDE_X0, "y0": 0.0, "x1": OCR_SIDE_X1, "y1": 1.0},
        },
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.get("/uploads/{filename}")
def serve_upload(filename: str):
    """Serves an uploaded video back to the browser for local preview
    playback. `filename` is resolved strictly under UPLOADS_DIR and checked
    against path traversal -- it's the uuid-based name we generated on
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
    starts it as a channel exactly like a live YouTube URL would be -- same
    FramePuller pipeline, just reading a local path (is_local=True skips
    yt-dlp resolution -- ffmpeg reads a local path the same way it reads a URL)."""
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
        log.warning("OCR_HOST=%s exposes this server beyond localhost with no auth in place -- "
                     "anyone who can reach it can pull arbitrary allow-listed URLs through your "
                     "server and upload files to it. Don't do this on an untrusted network.", HOST)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
