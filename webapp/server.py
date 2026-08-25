"""Live Whisper transcription server — model runs on the Intel Iris Xe iGPU.

Three input paths, all through the same segmenter + GPU pipeline:
  1. Browser microphone  — float32 16 kHz PCM streamed over the WebSocket
  2. Demo clips          — routed through the browser's audio graph like a mic
  3. Live/remote streams — server pulls any yt-dlp-supported URL (YouTube live,
                           VOD, radio), ffmpeg re-paces it to realtime

Run:  .venv\\Scripts\\python.exe webapp\\server.py   then open http://127.0.0.1:8000
"""
import asyncio
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from audio_io import find_ffmpeg  # noqa: E402

MODEL_DIR = PROJECT / "models" / "whisper-large-v3-int8-ov"
DEMO_CLIPS = {
    "ur": PROJECT / "audio" / "news_test.wav",
    "en": PROJECT / "audio" / "english_test.wav",
}

SR = 16000
MIN_SEG_S = 2.5       # don't transcribe fragments shorter than this on silence
MAX_SEG_S = 8.0       # hard flush — bounds latency
SIL_WIN_S = 0.7       # trailing window checked for silence
SIL_RMS = 0.010
SKIP_RMS = 0.004      # whole-segment RMS below this = no speech, drop it
PARTIAL_MIN_S = 1.2   # start showing live partials once the buffer has this much
PARTIAL_GAP_S = 0.8   # min wall-clock between partial decodes

app = FastAPI()
pipe = None
gpu = ThreadPoolExecutor(max_workers=1)  # one inference at a time on the iGPU


@app.on_event("startup")
def load_model():
    global pipe
    import openvino_genai
    print(f"loading {MODEL_DIR.name} on GPU ...", flush=True)
    t0 = time.perf_counter()
    pipe = openvino_genai.WhisperPipeline(str(MODEL_DIR), device="GPU")
    print(f"ready in {time.perf_counter() - t0:.1f}s", flush=True)


def transcribe(samples: np.ndarray, language: str):
    config = pipe.get_generation_config()
    config.task = "transcribe"
    if language != "auto":
        config.language = f"<|{language}|>"
    t0 = time.perf_counter()
    result = pipe.generate(samples, config)
    return str(result).strip(), time.perf_counter() - t0


class Segmenter:
    """Buffers a PCM stream, cuts on silence (hard cap MAX_SEG_S), transcribes
    finals and rolling partials on the GPU, and reports via `send` (async)."""

    def __init__(self, send, get_language, source: str):
        self.send = send
        self.get_language = get_language
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
        samples = np.concatenate(self.buf)
        self.buf, self.buf_len = [], 0
        seg_dur = len(samples) / SR
        t_start = self.clock
        self.clock += seg_dur
        if float(np.sqrt(np.mean(samples ** 2))) < SKIP_RMS:
            await self.send({"type": "partial", "text": ""})
            return  # silence-only segment — transcribing it just invites hallucination
        lang = self.get_language()
        text, dt = await self.loop.run_in_executor(gpu, transcribe, samples, lang)
        if text:
            await self.send({
                "type": "segment", "text": text, "t0": round(t_start, 1),
                "t1": round(self.clock, 1), "infer_s": round(dt, 2),
                "rtf": round(seg_dur / dt, 2), "lang": lang,
                "reason": reason, "source": self.source,
            })

    async def partial(self):
        self.partial_busy = True
        try:
            if not self.buf:
                return
            my_gen = self.gen
            samples = np.concatenate(self.buf)
            if float(np.sqrt(np.mean(samples ** 2))) < SKIP_RMS:
                return
            text, _ = await self.loop.run_in_executor(
                gpu, transcribe, samples, self.get_language())
            if self.gen == my_gen and text:  # buffer wasn't flushed mid-decode
                await self.send({"type": "partial", "text": text})
        finally:
            self.partial_busy = False
            self.last_partial = self.loop.time()


class StreamPuller:
    """yt-dlp -> ffmpeg (-re, realtime-paced) -> float32 16 kHz PCM -> Segmenter."""

    def __init__(self, url: str, segmenter: Segmenter, send):
        self.url = url
        self.seg = segmenter
        self.send = send
        self.procs = []
        self.task = None

    async def start(self):
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
               "--ffmpeg-location", str(Path(find_ffmpeg()).parent)]
        deno = PROJECT / "tools" / "deno" / "deno.exe"
        if deno.exists():  # lets yt-dlp solve YouTube player JS challenges
            cmd += ["--js-runtimes", f"deno:{deno}"]
        cmd += ["-f", "bestaudio/best", "-o", "-", self.url]
        ytdlp = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffmpeg = subprocess.Popen(
            [find_ffmpeg(), "-v", "error", "-re", "-i", "pipe:0",
             "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
            stdin=ytdlp.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        ytdlp.stdout.close()  # let ffmpeg own the pipe
        self.procs = [ytdlp, ffmpeg]
        self.task = asyncio.ensure_future(self._pump(ffmpeg))
        await self.send({"type": "stream_status", "state": "started", "url": self.url})

    async def _pump(self, ffmpeg):
        loop = asyncio.get_running_loop()
        chunk_bytes = int(SR * 0.5) * 4  # 0.5 s per read
        got_data = False
        try:
            while True:
                data = await loop.run_in_executor(None, ffmpeg.stdout.read, chunk_bytes)
                if not data:
                    break
                got_data = True
                await self.seg.feed(np.frombuffer(data, dtype=np.float32))
            await self.seg.flush("stream-end")
            detail = ""
            if not got_data:
                detail = "no audio received — is the URL valid / supported by yt-dlp?"
                try:  # both processes are dead by now; pull yt-dlp's real complaint
                    err = self.procs[0].stderr.read().decode("utf-8", errors="replace")
                    errors = [l for l in err.splitlines() if l.startswith("ERROR")]
                    if errors:
                        detail = errors[-1]
                except Exception:
                    pass
            await self.send({"type": "stream_status",
                             "state": "ended" if got_data else "error",
                             "detail": detail})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surfaced to the page instead of dying silently
            await self.send({"type": "stream_status", "state": "error", "detail": str(e)})

    def stop(self):
        if self.task:
            self.task.cancel()
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        self.procs = []


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/demo-audio/{lang}")
def demo_audio(lang: str):
    clip = DEMO_CLIPS.get(lang)
    if not clip or not clip.exists():
        return {"error": f"no demo clip for '{lang}'"}
    return FileResponse(clip, media_type="audio/wav")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state = {"language": "ur"}

    async def send(obj):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            pass  # client went away mid-send; the receive loop will close us

    mic_seg = Segmenter(send, lambda: state["language"], "mic")
    stream_seg = Segmenter(send, lambda: state["language"], "stream")
    puller = None

    try:
        while True:
            msg = await ws.receive()
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "config":
                    state["language"] = data.get("language", "ur")
                elif kind == "flush":  # mic stopped / clip ended
                    await mic_seg.flush("stop")
                elif kind == "stream_start":
                    if puller:
                        puller.stop()
                    puller = StreamPuller(data.get("url", ""), stream_seg, send)
                    await puller.start()
                elif kind == "stream_stop":
                    if puller:
                        puller.stop()
                        puller = None
                    await send({"type": "stream_status", "state": "stopped"})
            elif msg.get("bytes") is not None:
                await mic_seg.feed(np.frombuffer(msg["bytes"], dtype=np.float32))
    except WebSocketDisconnect:
        pass
    finally:
        if puller:
            puller.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
