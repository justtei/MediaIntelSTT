"""Multi-channel live Whisper transcription server.

Runs on CPU by default — no GPU is assumed or required. Server-managed
"channels" are the core concept: each is an independent audio source (a
YouTube/live-stream URL pulled server-side, or a browser microphone) with
its own rolling transcript. Multiple channels — e.g. several news channels'
YouTube live streams plus an operator's mic — run concurrently, each
segmented and transcribed independently, broadcasting live captions to every
connected dashboard and persisting every finalized segment to Postgres as a
growing corpus for future fine-tuning.

  1. YouTube/stream channels — POST /channels or WS {"type":"channel_start",
     "source":"youtube", "url":...}. Runs server-side; keeps transcribing
     even if no dashboard is open. URL is allow-list validated (SSRF safety,
     scripts/url_safety.py) before yt-dlp/ffmpeg ever sees it.
  2. Mic channels — WS {"type":"channel_start","source":"mic"} then binary
     PCM frames from that same connection. Tied to the owning connection
     (only that browser can supply mic audio) and stopped when it disconnects.

Backend/device/model, concurrency, and every tuning knob are environment-
configurable — see the STT_* variables below. GPU (Intel iGPU via OpenVINO)
and CUDA (via faster-whisper) are fully implemented and are one env var away
whenever that hardware is available — set STT_DEVICE=GPU or STT_DEVICE=cuda;
no code changes needed (requirements.md §2.2).

Run:  .venv/bin/python webapp/server.py   then open http://127.0.0.1:8000
Needs Postgres: docker compose up -d   (server still runs without it —
transcription works, segments just aren't persisted; see db.is_connected())

Key env vars (all optional, sane defaults for local CPU-only dev):
  STT_BACKEND               openvino | faster-whisper       (default: openvino)
  STT_DEVICE                openvino: GPU/CPU/NPU/auto (default CPU) ;
                             faster-whisper: cpu/cuda/auto (default cpu)
  STT_MODEL                 openvino: model dir (default small-int8 — see note below) ;
                             faster-whisper: size name or CT2 dir (default small)
  STT_WORKER_THREADS        concurrent transcriptions across all channels (default 2).
                             Each thread loads its OWN backend instance (pipelines
                             aren't assumed thread-safe for concurrent generate() calls)
                             — RAM cost is N x model size.
  STT_INITIAL_PROMPT        default custom-vocabulary prompt (FR7), overridable per channel
  STT_HOST / STT_PORT       bind address (default 127.0.0.1:8000 — see security note below)
  STT_WS_TOKEN              if set, required as /ws?token=... before a connection is accepted
  STT_ALLOWED_STREAM_HOSTS  comma-separated domain allow-list for channel_start URLs
  STT_ALLOW_ANY_STREAM_HOST 1 to disable the allow-list entirely (dangerous — SSRF risk)
  DATABASE_URL              postgres DSN (default matches docker-compose.yml's defaults)
  STT_MAX_SEG_S / STT_MIN_SEG_S / STT_SIL_WIN_S / STT_SIL_RMS / STT_SKIP_RMS /
  STT_PARTIAL_MIN_S / STT_PARTIAL_GAP_S   Segmenter tuning (defaults tuned for slow
                                          CPU inference — retune once running on GPU/CUDA)

Security note: the stream URL allow-list and the optional WS token both exist
for the moment this stops being localhost-only. Binding STT_HOST to 0.0.0.0/
LAN without setting STT_WS_TOKEN is a foot-gun — do not do it.

Latency note: model choice matters as much as channel count. Measured on
this project's own CPU dev machine: large-v3-int8 ran at 0.2x realtime even
for a SINGLE channel (too slow for live use, matching the README's own
"well below realtime on CPU" caveat) — small-int8 ran at 1.2-2.6x realtime.
That's why small-int8 is the default here, not large-v3-int8's better
accuracy. Separately, CPU inference is fundamentally serial per core:
STT_WORKER_THREADS lets N channels transcribe truly in parallel (bounded by
CPU cores and RAM for N model copies), but beyond that, more concurrent
channels means more queueing delay per channel — a hardware limit, not a
code one. For many simultaneous news channels at low latency with
large-v3-int8's quality, GPU/CUDA is what makes that realistic; CPU is for
development, evaluation, and modest channel counts with a fast model.
"""
import asyncio
import hmac
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

from scripts import db  # noqa: E402
from scripts.audio_io import find_ffmpeg  # noqa: E402
from scripts.backends import Backend, BackendLoadError, load_backend  # noqa: E402
from scripts.url_safety import DEFAULT_ALLOWED_HOSTS, validate_stream_url  # noqa: E402

logging.basicConfig(
    level=os.environ.get("STT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("stt.server")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v else default
    except ValueError:
        log.warning("invalid %s=%r, using default %s", name, v, default)
        return default


# --- config -----------------------------------------------------------------
BACKEND_NAME = os.environ.get("STT_BACKEND", "openvino")
if BACKEND_NAME not in ("openvino", "faster-whisper"):
    sys.exit(f"STT_BACKEND must be 'openvino' or 'faster-whisper', got {BACKEND_NAME!r}")

# CPU by default on both backends — no GPU/CUDA hardware assumed. Set
# STT_DEVICE=GPU (openvino) or STT_DEVICE=cuda (faster-whisper) once available.
DEFAULT_DEVICE = {"openvino": "CPU", "faster-whisper": "cpu"}
DEVICE = os.environ.get("STT_DEVICE") or DEFAULT_DEVICE[BACKEND_NAME]

def _get_default_model(backend: str) -> str:
    if backend == "openvino":
        small_ov = PROJECT / "models" / "whisper-small-int8-ov"
        if small_ov.is_dir():
            return str(small_ov)
        medium_ov = PROJECT / "models" / "whisper-medium-int8-ov"
        if medium_ov.is_dir():
            return str(medium_ov)
        large_ov = PROJECT / "models" / "whisper-large-v3-int8-ov"
        if large_ov.is_dir():
            return str(large_ov)
        return str(small_ov)
    return "small"


_env_model = os.environ.get("STT_MODEL")
if _env_model in ("medium", "medium-int8"):
    MODEL = str(PROJECT / "models" / "whisper-medium-int8-ov")
elif _env_model in ("small", "small-int8"):
    MODEL = str(PROJECT / "models" / "whisper-small-int8-ov")
elif _env_model in ("large", "large-v3", "large-v3-int8"):
    MODEL = str(PROJECT / "models" / "whisper-large-v3-int8-ov")
else:
    MODEL = _env_model or _get_default_model(BACKEND_NAME)

_default_workers = 1
WORKER_THREADS = int(os.environ.get("STT_WORKER_THREADS", str(_default_workers)))
INITIAL_PROMPT = os.environ.get("STT_INITIAL_PROMPT") or None
HOST = os.environ.get("STT_HOST", "127.0.0.1")
PORT = int(os.environ.get("STT_PORT", "8000"))
WS_TOKEN = os.environ.get("STT_WS_TOKEN") or None

ALLOW_ANY_STREAM_HOST = os.environ.get("STT_ALLOW_ANY_STREAM_HOST") == "1"
_hosts_env = os.environ.get("STT_ALLOWED_STREAM_HOSTS")
ALLOWED_STREAM_HOSTS = (
    frozenset(h.strip().lower() for h in _hosts_env.split(",") if h.strip())
    if _hosts_env else DEFAULT_ALLOWED_HOSTS
)

SR = 16000
MIN_SEG_S = _env_float("STT_MIN_SEG_S", 1.5)      # don't transcribe fragments shorter than this on silence
MAX_SEG_S = _env_float("STT_MAX_SEG_S", 3.5)       # hard flush — bounds latency
SIL_WIN_S = _env_float("STT_SIL_WIN_S", 0.35)      # trailing window checked for silence
SIL_RMS = _env_float("STT_SIL_RMS", 0.010)
SKIP_RMS = _env_float("STT_SKIP_RMS", 0.004)       # whole-segment RMS below this = no speech, drop it
PARTIAL_MIN_S = _env_float("STT_PARTIAL_MIN_S", 1.0)  # start showing live partials once buffer has this much
PARTIAL_GAP_S = _env_float("STT_PARTIAL_GAP_S", 0.8)  # min wall-clock between partial decodes

MAX_WS_BINARY_BYTES = int(os.environ.get("STT_MAX_WS_BINARY_BYTES", str(4 * 1024 * 1024)))  # 4 MB/frame
STREAM_FIRST_DATA_TIMEOUT_S = _env_float("STT_STREAM_FIRST_DATA_TIMEOUT_S", 20.0)
MAX_INITIAL_PROMPT_CHARS = int(os.environ.get("STT_MAX_INITIAL_PROMPT_CHARS", "500"))
MAX_CHANNEL_NAME_CHARS = 80

worker = ThreadPoolExecutor(max_workers=WORKER_THREADS)
# One backend instance per worker thread — pipelines aren't assumed safe for
# concurrent generate()/transcribe() calls from multiple threads, so instead
# of sharing one we pool N independently-loaded instances (thread-safe by
# construction: queue.Queue.get()/put() serialize access to each instance).
_backend_pool: "queue.Queue[Backend]" = queue.Queue()


def _load_backend_pool(n: int) -> str:
    device = None
    for i in range(n):
        log.info("loading backend instance %d/%d (backend=%s model=%s device=%s) ...",
                 i + 1, n, BACKEND_NAME, MODEL, DEVICE)
        if BACKEND_NAME == "openvino":
            b = load_backend("openvino", model_dir=MODEL, device=DEVICE)
        else:
            b = load_backend("faster-whisper", model=MODEL, device=DEVICE)
        device = b.device
        _backend_pool.put(b)
    return device


def _transcribe(samples: np.ndarray, language: str, initial_prompt: Optional[str], is_partial: bool = False):
    b = _backend_pool.get()
    try:
        t0 = time.perf_counter()
        text, _chunks = b.transcribe(
            samples, language, initial_prompt=initial_prompt, is_partial=is_partial, return_timestamps=False
        )
        return text, time.perf_counter() - t0
    finally:
        _backend_pool.put(b)


ACTIVE_DEVICE: Optional[str] = None  # set once the pool is loaded, for /health


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ACTIVE_DEVICE
    t0 = time.perf_counter()
    try:
        ACTIVE_DEVICE = _load_backend_pool(WORKER_THREADS)
        log.info("ready on %s (%d instance(s)) in %.1fs", ACTIVE_DEVICE, WORKER_THREADS,
                  time.perf_counter() - t0)
    except BackendLoadError:
        log.exception("backend failed to load on every candidate device")
        raise

    try:
        await db.connect()
        log.info("connected to database")
    except Exception as e:
        log.warning("database unavailable (%s) — running without persistence (run `docker compose up -d` if Postgres is desired)", e)

    yield

    for channel_id in list(CHANNELS):
        await stop_channel(channel_id, reason="server_shutdown")
    await db.disconnect()


app = FastAPI(lifespan=lifespan)


class Segmenter:
    """Buffers a PCM stream, cuts on silence (hard cap MAX_SEG_S), transcribes
    finals and rolling partials on the worker pool, and reports via `send`
    (async). Backend errors are caught per-segment so one bad decode doesn't
    take down the whole channel."""

    def __init__(self, send, get_language, get_prompt, source: str):
        self.send = send
        self.get_language = get_language
        self.get_prompt = get_prompt
        self.source = source
        self.loop = asyncio.get_running_loop()
        self.buf: list[np.ndarray] = []
        self.buf_len = 0
        self.clock = 0.0
        self.gen = 0
        self.partial_busy = False
        self.last_partial = 0.0

    async def feed(self, chunk: np.ndarray):
        self.buf.append(chunk)
        self.buf_len += len(chunk)
        total_s = self.buf_len / SR
        if total_s >= MAX_SEG_S:
            await self.flush("max")
            return
        if total_s >= MIN_SEG_S:
            tail = np.concatenate(self.buf)[-int(SIL_WIN_S * SR):]
            if float(np.sqrt(np.mean(tail ** 2))) < SIL_RMS:
                await self.flush("silence")
                return
        if (total_s >= PARTIAL_MIN_S and not self.partial_busy
                and self.loop.time() - self.last_partial >= PARTIAL_GAP_S):
            asyncio.ensure_future(self.partial())

    async def flush(self, reason: str):
        self.gen += 1
        if not self.buf_len:
            return
        self.partial_busy = True
        try:
            samples = np.concatenate(self.buf)
            self.buf, self.buf_len = [], 0
            seg_dur = len(samples) / SR
            t_start = self.clock
            self.clock += seg_dur
            if float(np.sqrt(np.mean(samples ** 2))) < SKIP_RMS:
                await self.send({"type": "partial", "text": ""})
                return  # silence-only segment — transcribing it just invites hallucination
            lang = self.get_language()
            try:
                text, dt = await self.loop.run_in_executor(
                    worker, _transcribe, samples, lang, self.get_prompt(), False)
            except Exception:
                log.exception("transcribe failed on a %s segment (source=%s)", self.source, self.source)
                await self.send({"type": "error", "detail": "transcription failed on this segment",
                                  "source": self.source})
                return
            if text:
                await self.send({
                    "type": "segment", "text": text, "t0": round(t_start, 1),
                    "t1": round(self.clock, 1), "infer_s": round(dt, 2),
                    "rtf": round(seg_dur / dt, 2) if dt > 0 else None, "lang": lang,
                    "reason": reason, "source": self.source,
                })
        finally:
            self.partial_busy = False
            self.last_partial = self.loop.time()

    async def partial(self):
        self.partial_busy = True
        try:
            if not self.buf:
                return
            my_gen = self.gen
            samples = np.concatenate(self.buf)
            if float(np.sqrt(np.mean(samples ** 2))) < SKIP_RMS:
                return
            try:
                text, _ = await self.loop.run_in_executor(
                    worker, _transcribe, samples, self.get_language(), self.get_prompt(), True)
            except Exception:
                log.exception("partial transcribe failed (source=%s)", self.source)
                return
            if self.gen == my_gen and text:  # buffer wasn't flushed mid-decode
                await self.send({"type": "partial", "text": text})
        finally:
            self.partial_busy = False
            self.last_partial = self.loop.time()


def _deno_path() -> Optional[str]:
    return shutil.which("deno") or next(
        (str(p) for p in (PROJECT / "tools" / "deno").glob("**/deno*")
         if p.is_file() and os.access(p, os.X_OK)), None)


STREAM_RESOLVE_TIMEOUT_S = _env_float("STT_STREAM_RESOLVE_TIMEOUT_S", 30.0)


@dataclass
class ResolvedStream:
    media_url: str
    video_id: Optional[str] = None
    title: Optional[str] = None


def _resolve_media_url(url: str) -> ResolvedStream:
    """Ask yt-dlp for the direct, playable media URL plus metadata (video id,
    title) — one call, one network round-trip (blocking; always call via
    run_in_executor). Reading straight from the media URL with our own
    ffmpeg avoids yt-dlp's internal ffmpeg-based downloader entirely:
    simpler, faster to first audio, and sidesteps a real crash hit in
    testing where that internal path segfaulted on live YouTube HLS with
    some ffmpeg builds (see PRODUCTION_READINESS.md). The video id feeds the
    frontend's embedded YouTube player for video/audio preview."""
    cmd = [sys.executable, "-m", "yt_dlp", "-j", "-f", "bestaudio/best", "--no-warnings"]
    deno = _deno_path()
    if deno:  # lets yt-dlp solve YouTube's player-JS challenges
        cmd += ["--js-runtimes", f"deno:{deno}"]
    else:
        log.warning("no deno runtime found — YouTube extraction is degraded/deprecated without "
                    "one (yt-dlp's own warning). Install: curl -fsSL https://deno.land/install.sh | sh")
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=STREAM_RESOLVE_TIMEOUT_S)
    if result.returncode != 0 or not result.stdout.strip():
        errors = [l for l in result.stderr.splitlines() if l.startswith("ERROR")]
        raise RuntimeError(errors[-1] if errors else "yt-dlp could not resolve a playable URL")
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise RuntimeError(f"yt-dlp returned unparseable output: {e}")
    media_url = info.get("url")
    if not media_url:
        raise RuntimeError("yt-dlp resolved no playable media URL")
    return ResolvedStream(media_url=media_url, video_id=info.get("id"), title=info.get("title"))


class StreamPuller:
    """yt-dlp resolves the direct media URL once; ffmpeg then reads it
    directly (low latency stream mode) -> float32 16 kHz PCM -> Segmenter. A
    single subprocess handles the actual media transport — see
    _resolve_media_url for why yt-dlp isn't kept running as a pipe source.

    The channel URL is validated against an allow-list (scripts/url_safety.py)
    before any subprocess is spawned — see issue #11 in requirements.md.
    """

    def __init__(self, url: str, segmenter: Segmenter, send):
        self.url = url
        self.seg = segmenter
        self.send = send
        self.procs = []
        self.task = None
        self._stopped = False  # guards against spawning ffmpeg after stop() raced a slow resolve

    async def start(self):
        rejection = validate_stream_url(self.url, ALLOWED_STREAM_HOSTS, ALLOW_ANY_STREAM_HOST)
        if rejection:
            log.warning("rejected stream URL %r: %s", self.url, rejection)
            await self.send({"type": "stream_status", "state": "error", "detail": rejection})
            return

        loop = asyncio.get_running_loop()
        try:
            resolved = await loop.run_in_executor(None, _resolve_media_url, self.url)
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
            find_ffmpeg(),
            "-v", "error",
            "-re",
            "-i", resolved.media_url,
            "-ac", "1",
            "-ar", str(SR),
            "-f", "f32le",
            "pipe:1",
        ]
        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if self._stopped:  # stop() raced us between resolve and spawn
            ffmpeg.kill()
            return
        self.procs = [ffmpeg]
        self.task = asyncio.ensure_future(self._pump(ffmpeg))
        await self.send({"type": "stream_status", "state": "started", "url": self.url,
                          "video_id": resolved.video_id, "title": resolved.title})

    async def _pump(self, ffmpeg):
        loop = asyncio.get_running_loop()
        chunk_bytes = int(SR * 0.25) * 4  # 0.25 s per read for low latency
        got_data = False
        try:
            try:
                first = await asyncio.wait_for(
                    loop.run_in_executor(None, ffmpeg.stdout.read, chunk_bytes),
                    timeout=STREAM_FIRST_DATA_TIMEOUT_S)
            except asyncio.TimeoutError:
                await self.send({"type": "stream_status", "state": "error",
                                  "detail": f"no audio within {STREAM_FIRST_DATA_TIMEOUT_S:.0f}s "
                                            "— is the URL live/reachable?"})
                return
            if first:
                got_data = True
                await self.seg.feed(np.frombuffer(first, dtype=np.float32))
            while True:
                data = await loop.run_in_executor(None, ffmpeg.stdout.read, chunk_bytes)
                if not data:
                    break
                got_data = True
                await self.seg.feed(np.frombuffer(data, dtype=np.float32))
            await self.seg.flush("stream-end")
            detail = ""
            if not got_data:
                detail = "no audio received from ffmpeg"
                try:  # process is dead by now; pull ffmpeg's real complaint
                    err = self.procs[0].stderr.read().decode("utf-8", errors="replace").strip()
                    if err:
                        detail = err.splitlines()[-1]
                except Exception:
                    pass
            await self.send({"type": "stream_status",
                             "state": "ended" if got_data else "error",
                             "detail": detail})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surfaced to the page instead of dying silently
            log.exception("stream pump failed")
            await self.send({"type": "stream_status", "state": "error", "detail": str(e)})

    def stop(self):
        self._stopped = True
        if self.task:
            self.task.cancel()
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        self.procs = []


# --- channel registry ---------------------------------------------------
# A channel is one independently-transcribed audio source: a server-pulled
# YouTube/stream URL, or a browser microphone. Stream channels run until
# explicitly stopped (no browser needs to stay open); mic channels are tied
# to the WebSocket connection that owns them.

@dataclass
class Channel:
    id: str
    name: str
    source_type: str  # "youtube" | "mic"
    url: Optional[str]
    language: str
    initial_prompt: Optional[str]
    segmenter: Segmenter
    status: str = "starting"
    puller: Optional[StreamPuller] = None
    owner_ws: Optional[WebSocket] = None
    created_at: float = field(default_factory=time.time)
    video_id: Optional[str] = None  # resolved YouTube video id, for the frontend's embedded player

    def public(self) -> dict:
        return {
            "id": self.id, "name": self.name, "source_type": self.source_type,
            "url": self.url, "language": self.language, "status": self.status,
            "created_at": self.created_at, "video_id": self.video_id,
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
    """The `send` callback given to a channel's Segmenter/StreamPuller: tags
    every event with channel_id, broadcasts to all dashboards, keeps the
    channel registry's status in sync (so a dashboard that connects *after*
    a stream error still sees "error", not a stale "live"), and persists
    finalized segments to Postgres — the corpus future fine-tuning draws from."""
    async def send(event: dict):
        await broadcast({**event, "channel_id": channel_id})
        channel = CHANNELS.get(channel_id)

        if channel and event.get("type") == "stream_status":
            new_status = _STREAM_STATE_TO_STATUS.get(event.get("state"))
            if new_status and new_status != channel.status:
                channel.status = new_status
                if db.is_connected():
                    try:
                        await db.set_channel_status(channel_id, new_status)
                    except Exception:
                        log.exception("failed to update status for channel %s", channel_id)
            if event.get("video_id") and not channel.video_id:
                channel.video_id = event["video_id"]

        if event.get("type") == "segment" and db.is_connected():
            try:
                await db.insert_segment(
                    channel_id, text=event["text"], language=event.get("lang") or "",
                    t0_s=event.get("t0"), t1_s=event.get("t1"),
                    infer_s=event.get("infer_s"), rtf=event.get("rtf"),
                    reason=event.get("reason"), source=event.get("source"),
                    backend=BACKEND_NAME, device=ACTIVE_DEVICE,
                )
            except Exception:
                log.exception("failed to persist segment for channel %s", channel_id)
    return send


async def start_channel(name: str, source_type: str, url: Optional[str], language: str,
                         initial_prompt: Optional[str] = None,
                         owner_ws: Optional[WebSocket] = None) -> Channel:
    if source_type not in ("youtube", "mic"):
        raise ValueError(f"unknown source: {source_type!r} (expected 'youtube' or 'mic')")
    name = (name or "").strip()[:MAX_CHANNEL_NAME_CHARS] or ("Mic" if source_type == "mic" else "Channel")
    if source_type == "youtube":
        rejection = validate_stream_url(url or "", ALLOWED_STREAM_HOSTS, ALLOW_ANY_STREAM_HOST)
        if rejection:
            raise ValueError(rejection)
    if initial_prompt and len(initial_prompt) > MAX_INITIAL_PROMPT_CHARS:
        initial_prompt = initial_prompt[:MAX_INITIAL_PROMPT_CHARS]

    channel_id = await db.create_channel(name, source_type, url, language) \
        if db.is_connected() else str(uuid.uuid4())

    send = _channel_send(channel_id)

    def get_language():
        return channel.language

    def get_prompt():
        return channel.initial_prompt

    segmenter = Segmenter(send, get_language, get_prompt, source=source_type)
    channel = Channel(id=channel_id, name=name, source_type=source_type, url=url,
                       language=language, initial_prompt=initial_prompt,
                       segmenter=segmenter, owner_ws=owner_ws)
    CHANNELS[channel_id] = channel

    if source_type == "mic":
        # mic audio is ready as soon as the browser starts feeding frames — no
        # resolution step, unlike a stream URL.
        channel.status = "live"
        if db.is_connected():
            await db.set_channel_status(channel_id, "live")

    # Broadcast immediately (status "starting" for youtube) rather than after
    # URL resolution, which can take up to STT_STREAM_RESOLVE_TIMEOUT_S: with
    # multiple channels being added concurrently, blocking here would both
    # delay the UI's feedback and serialize channel creation on this
    # connection behind one slow resolution.
    await broadcast({"type": "channel_added", "channel": channel.public()})

    if source_type == "youtube":
        puller = StreamPuller(url, segmenter, send)
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
    if db.is_connected():
        try:
            await db.set_channel_status(channel_id, "stopped")
        except Exception:
            log.exception("failed to update channel status for %s", channel_id)
    await broadcast({"type": "channel_removed", "channel_id": channel_id, "reason": reason})


# --- HTTP API -----------------------------------------------------------

class ChannelCreateRequest(BaseModel):
    name: str
    source: str = "youtube"  # mic channels must start from the browser over /ws (need live audio)
    url: Optional[str] = None
    language: str = "ur"
    initial_prompt: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    ready = ACTIVE_DEVICE is not None
    body = {
        "status": "ok" if ready else "loading",
        "backend": BACKEND_NAME,
        "device": ACTIVE_DEVICE,
        "model": MODEL,
        "worker_threads": WORKER_THREADS,
        "active_channels": len(CHANNELS),
        "database": "connected" if db.is_connected() else "disconnected",
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.get("/channels")
def list_channels_endpoint():
    return [c.public() for c in CHANNELS.values()]


@app.post("/channels")
async def create_channel_endpoint(req: ChannelCreateRequest):
    if req.source == "mic":
        raise HTTPException(400, "mic channels need live audio — start them from the browser over /ws")
    try:
        channel = await start_channel(req.name, req.source, req.url, req.language, req.initial_prompt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return channel.public()


@app.delete("/channels/{channel_id}")
async def delete_channel_endpoint(channel_id: str):
    if channel_id not in CHANNELS:
        raise HTTPException(404, "channel not found")
    await stop_channel(channel_id, reason="stopped_by_user")
    return {"status": "stopped"}


@app.get("/channels/{channel_id}/segments")
async def channel_segments_endpoint(channel_id: str, limit: int = 50):
    if not db.is_connected():
        raise HTTPException(503, "database not connected — no history available")
    limit = max(1, min(limit, 500))
    return await db.list_segments(channel_id, limit)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if WS_TOKEN and not hmac.compare_digest(ws.query_params.get("token") or "", WS_TOKEN):
        await ws.close(code=1008)  # policy violation
        return
    await ws.accept()
    SUBSCRIBERS.add(ws)
    mic_channel_id: Optional[str] = None

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
                    source = data.get("source")
                    prompt = data.get("initial_prompt") or None
                    if source == "mic":
                        if mic_channel_id:
                            await stop_channel(mic_channel_id, reason="replaced")
                            mic_channel_id = None
                        try:
                            channel = await start_channel(
                                data.get("name", "Mic"), "mic", None,
                                data.get("language", "ur"), prompt, owner_ws=ws)
                        except ValueError as e:
                            await send({"type": "error", "detail": str(e)})
                            continue
                        mic_channel_id = channel.id
                        await send({"type": "mic_channel_ready", "channel_id": channel.id})
                    elif source == "youtube":
                        try:
                            await start_channel(
                                data.get("name", "Channel"), "youtube", data.get("url", ""),
                                data.get("language", "ur"), prompt)
                        except ValueError as e:
                            await send({"type": "error", "detail": str(e)})
                    else:
                        await send({"type": "error", "detail": f"unknown source {source!r}"})

                elif kind == "channel_stop":
                    cid = data.get("channel_id")
                    if cid:
                        await stop_channel(cid, reason="stopped_by_user")
                        if cid == mic_channel_id:
                            mic_channel_id = None

                elif kind == "config":
                    channel = CHANNELS.get(data.get("channel_id"))
                    if channel:
                        channel.language = data.get("language", channel.language)
                        if "initial_prompt" in data:
                            prompt = data.get("initial_prompt") or None
                            if prompt and len(prompt) > MAX_INITIAL_PROMPT_CHARS:
                                prompt = prompt[:MAX_INITIAL_PROMPT_CHARS]
                            channel.initial_prompt = prompt

                elif kind == "flush":  # mic paused/stopped client-side, channel stays open
                    cid = data.get("channel_id") or mic_channel_id
                    channel = CHANNELS.get(cid) if cid else None
                    if channel:
                        await channel.segmenter.flush("stop")

            elif msg.get("bytes") is not None:
                if not mic_channel_id or mic_channel_id not in CHANNELS:
                    continue  # no active mic channel on this connection; drop stray audio
                payload = msg["bytes"]
                if len(payload) > MAX_WS_BINARY_BYTES:
                    await send({"type": "error", "detail": "binary frame too large, dropped"})
                    continue
                await CHANNELS[mic_channel_id].segmenter.feed(np.frombuffer(payload, dtype=np.float32))
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        log.exception("unhandled error in /ws connection")
    finally:
        SUBSCRIBERS.discard(ws)
        if mic_channel_id:
            await stop_channel(mic_channel_id, reason="disconnected")


if __name__ == "__main__":
    if HOST not in ("127.0.0.1", "localhost", "::1") and not WS_TOKEN:
        log.warning("STT_HOST=%s exposes this server beyond localhost with no STT_WS_TOKEN set — "
                     "anyone who can reach it can stream your mic input and pull arbitrary "
                     "allow-listed URLs through your server. Set STT_WS_TOKEN before doing this.",
                     HOST)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
