# NGMI AI Services

Monorepo for three independent broadcast-media-intelligence services, each
pulling live signal out of a YouTube/live-stream channel or an uploaded
video: what's being **said** (speech-to-text), what's being **shown in
text** (Urdu ticker OCR), and **who's on screen** (face recognition).

Each service is a standalone FastAPI + WebSocket app with its own `.venv`,
its own dependencies, its own port, and its own dashboard UI — they don't
import from one another and none is required for the others to run. They
live in one repo because they share the same shape (channel-based live
pipeline, YouTube/upload ingestion, SSRF-safe URL allow-listing, stall
detection/reconnect) and the same operator audience.

| Service | Folder | Port | What it does |
|---|---|---|---|
| **Speech-to-text** | [`ngmi-stt-backend/`](ngmi-stt-backend/) | `8000` | Multi-channel live Urdu/English transcription (Whisper via OpenVINO or faster-whisper), mic input, Postgres-backed corpus for future fine-tuning |
| **Urdu OCR** | [`ngmi-ocr-backend/`](ngmi-ocr-backend/) | `8020` | Reads news-ticker/sticker Urdu text via a LoRA-tuned Qwen2-VL-2B vision-language model (Qaari) |
| **Face recognition** | [`ngmi-face-recognation-backend/`](ngmi-face-recognation-backend/) | `8010` | Open-set face detection + recognition (YuNet + SFace) against a small enrolled photo gallery |

Each folder's own `README.md` is the full reference for that service
(approach, tuning knobs, known limitations). This file covers the
repo-wide setup and how the three fit together.

## Repo layout

```
ngmi-ai-services/
├── ngmi-stt-backend/               speech-to-text service
├── ngmi-ocr-backend/               Urdu OCR service
├── ngmi-face-recognation-backend/  face recognition service
├── .github/workflows/ci.yml        CI (lint + pytest)
└── .gitignore                      shared ignore rules (.venv/, models/,
                                     tools/, per-service uploads/gallery data)
```

Each service folder is self-contained:

```
<service>/
├── .venv/            local virtual env (gitignored, created per machine)
├── requirements.txt
├── scripts/           backend logic (model loading, inference, CLI tools)
├── webapp/
│   ├── server.py       FastAPI + WebSocket app, `python webapp/server.py` to run
│   └── static/index.html   dashboard UI
└── models/ or faces/  model weights / enrollment data (gitignored — see
                        each service's README for how to fetch/build them)
```

## Prerequisites

- **Python 3.11+** (each service manages its own `.venv`; nothing is shared)
- **ffmpeg** on `PATH` — for live YouTube/stream channels, use a **master**
  build (e.g. [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)),
  not a stable point release (a stable 7.0.2 build was found to segfault on
  live YouTube HLS specifically)
- **deno** on `PATH` (or under each service's `tools/deno/`) — yt-dlp needs
  a JS runtime to solve YouTube's player challenges for reliable extraction
- **Docker** (optional) — only needed for the STT service's Postgres corpus
- A GPU is **not** required for any of the three; each defaults to CPU and
  documents an optional GPU/CUDA path in its own README

## Setup — all three services

Each service has its **own separate `.venv`** — there's no shared install
step, and no venv is shared between services. Every command block below is
independent and self-contained, run from the repo root; `.venv\Scripts\...`
always refers to that one service's own virtual env, never another's.

```powershell
# --- STT backend (Windows PowerShell) ---
cd ngmi-stt-backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py small-int8
.venv\Scripts\python webapp\server.py            # -> http://127.0.0.1:8000
cd ..
```

```powershell
# --- OCR backend ---
cd ngmi-ocr-backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python webapp\server.py            # -> http://127.0.0.1:8020
# first run downloads the base model + LoRA adapter (~4.5GB) from Hugging Face
cd ..
```

```powershell
# --- Face recognition backend ---
cd ngmi-face-recognation-backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_face_models.py
# populate faces/gallery/<person-name>/*.jpg, then:
.venv\Scripts\python scripts\enroll_faces.py --calibrate
.venv\Scripts\python webapp\server.py            # -> http://127.0.0.1:8010
cd ..
```

All three can run at once — start each in its own terminal (each block
above ends back at the repo root, so it's safe to run them one after
another in the same terminal too, just not concurrently in one). They're
on different ports and share nothing at runtime. Each has a `GET /health`
endpoint that reports readiness (model/device loaded, gallery size, DB
connection, etc.) — useful for confirming a service actually came up
before opening its dashboard.

### Already set up? Just run this

Once each service's `.venv` exists, dependencies are installed, and any
one-time model download/enrollment is done, day-to-day startup is just —
**one terminal per service**, each started from the repo root:

```powershell
# terminal 1 — STT, http://127.0.0.1:8000
cd ngmi-stt-backend; .venv\Scripts\python webapp\server.py
```

```powershell
# terminal 2 — OCR, http://127.0.0.1:8020
cd ngmi-ocr-backend; .venv\Scripts\python webapp\server.py
```

```powershell
# terminal 3 — Face recognition, http://127.0.0.1:8010
cd ngmi-face-recognation-backend; .venv\Scripts\python webapp\server.py
```

That's the whole day-to-day workflow; everything above the "Already set
up?" heading is one-time setup.

See each service's own `README.md` for: full CLI tooling (STT's
transcribe/WER-eval pipeline), model/hardware notes, every environment
variable, the REST API, and known limitations.

## Security note (applies to all three)

None of the three services has authentication yet. Each binds to
`127.0.0.1` by default and validates channel URLs against a domain
allow-list (`scripts/url_safety.py` in every service) before yt-dlp/ffmpeg
touches them. Do not bind any of them to `0.0.0.0`/a LAN address on an
untrusted network without adding auth first — the STT service supports an
optional `STT_WS_TOKEN`; the other two have no equivalent yet.

## Branching

- `main` — stable
- `dev` — active development

## Tests

Only the STT service currently has a test suite (pure-logic modules —
`urdu_norm`, `script_audit`, `wer_eval`, `backends`, `url_safety`,
`webvtt` — no GPU/model/audio file needed):

```bash
cd ngmi-stt-backend
pip install -r requirements-test.txt
python -m pytest tests/ -v
```
