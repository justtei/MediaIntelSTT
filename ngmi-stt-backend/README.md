# STT Model Framework — multi-channel Urdu/English broadcast transcription

Runs Whisper **on CPU by default** — no GPU required to get started — via
either OpenVINO or `faster-whisper` (CTranslate2). Two ways to use it:

- **CLI pipeline** (`scripts/`) — one-shot transcription of a local audio
  file, plus the tooling to build an evaluation/fine-tuning dataset (script
  audit, WER scoring, WebVTT export).
- **Live multi-channel dashboard** (`webapp/`) — add any number of YouTube/
  live-stream URLs (e.g. several news channels at once) plus your own mic;
  each is transcribed **independently and concurrently**, captions appear
  live under each channel's card, and every finalized segment is persisted
  to Postgres as a growing corpus for future fine-tuning. Handles mixed
  Urdu/English speech via Whisper's per-segment language auto-detection.

GPU (Intel iGPU via OpenVINO) and CUDA (NVIDIA via `faster-whisper`) are
fully implemented and are one flag/env var away whenever that hardware is
available — see [GPU / CUDA (optional)](#gpu--cuda-optional) below. Nothing
here assumes you have one; see the latency note there for what CPU-only
multi-channel transcription can and can't do.

## Quick start — venv → server → dashboard

The whole path from a fresh clone to live captions in the browser:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1

# 2. Install Python dependencies into it
pip install -r requirements.txt

# 3. Make sure ffmpeg and deno are on PATH — see "Setup" below for exactly
#    why (short version: a stable ffmpeg release segfaults on live YouTube
#    HLS, and yt-dlp needs deno for reliable YouTube extraction)

# 4. Download a Whisper model — small-int8 (~245 MB) is what the server
#    loads by default. large-v3-int8 (~1.6 GB) has better accuracy but
#    measured 0.2x realtime on CPU (even one channel) — too slow for *live*
#    captioning without GPU/CUDA; see the Model notes section below.
python scripts/download_models.py small-int8
# python scripts/download_models.py                # large-v3-int8, only if you have GPU/CUDA or don't mind delay

# 5. (Optional) start Postgres so finalized captions get saved for fine-tuning
docker compose up -d

# 6. Start the live transcription server

# Option A: Small Model (Recommended — Ultra-Fast, Real-Time with Zero Delay)
# PowerShell (Windows):
$env:STT_MODEL="models/whisper-small-int8-ov"; $env:STT_WORKER_THREADS="1"; .venv\Scripts\python webapp/server.py
# Bash (Linux/macOS):
STT_MODEL=models/whisper-small-int8-ov STT_WORKER_THREADS=1 python webapp/server.py

# Option B: Medium Model (Balanced — Better Accuracy than Small, 3x Faster than Large)
# PowerShell (Windows):
$env:STT_MODEL="models/whisper-medium-int8-ov"; $env:STT_WORKER_THREADS="1"; .venv\Scripts\python webapp/server.py
# Bash (Linux/macOS):
STT_MODEL=models/whisper-medium-int8-ov STT_WORKER_THREADS=1 python webapp/server.py

# Option C: Large-v3 Model (High Accuracy, Higher Compute)
# PowerShell (Windows):
$env:STT_MODEL="models/whisper-large-v3-int8-ov"; $env:STT_WORKER_THREADS="1"; .venv\Scripts\python webapp/server.py
# Bash (Linux/macOS):
STT_MODEL=models/whisper-large-v3-int8-ov STT_WORKER_THREADS=1 python webapp/server.py
```

Leave that running, then open **http://127.0.0.1:8000** in your browser:

1. Click one of the **Quick add** buttons (Dunya News,
   Samaa TV, Express News, 92 News HD, BOL News, Aaj News) — it starts
   transcribing that live channel immediately, or
2. Paste any YouTube/live URL into the **Channel name** + **YouTube / live
   stream URL** fields and click **Add channel**, or
3. Click **Start microphone** to transcribe your own voice instead.

Live captions (proper Nastaliq rendering for Urdu) appear under each
channel's card as it speaks — no further steps needed. `Ctrl+C` in the
terminal stops the server; channels don't need to be stopped first.

### Already installed? Just run this

Skip steps 1–5 above if you've already created the venv, installed
dependencies, and downloaded a model at least once:

```powershell
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
$env:STT_MODEL="models/whisper-small-int8-ov"; $env:STT_WORKER_THREADS="1"; python webapp/server.py
```

```bash
# Linux / macOS:
source .venv/bin/activate
STT_MODEL=models/whisper-small-int8-ov STT_WORKER_THREADS=1 python webapp/server.py
```

Then open **http://127.0.0.1:8000** — same three options as above (Quick
add / paste a URL / start the mic). That's the entire day-to-day workflow;
everything before this point is one-time setup.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**ffmpeg** must be on `PATH`. For the CLI pipeline (local files) any recent
build works. **For the live dashboard's YouTube/stream channels, use an
ffmpeg "master" build (e.g. https://github.com/BtbN/FFmpeg-Builds), not a
stable point release** — a stable 7.0.2 static build was found to segfault
reading live YouTube HLS specifically (reproduced independent of this
project, see `PRODUCTION_READINESS.md`); the master build did not.

**deno** should be on `PATH` (or under `tools/deno/`) for reliable YouTube
extraction — yt-dlp needs a JS runtime to solve YouTube's player challenges;
without one, extraction is degraded/unreliable. Install:
`curl -fsSL https://deno.land/install.sh | sh`

**Postgres** (optional but needed to persist the corpus) via Docker:

```bash
docker compose up -d
```

The server runs fine without it — transcription still works, segments just
aren't saved (check `/health`'s `"database"` field, or the dashboard header's
`db ✓`/`db ✗` indicator).

## 1. One-time model download

```bash
python scripts/download_models.py                # large-v3 int8 (~1.6 GB) — Urdu quality pick
python scripts/download_models.py small-int8      # optional: fast model for quick iterations
```

## 2. Get real test audio (not Common Voice / FLEURS)

Pick three ~10-minute clips deliberately: one anchor monologue, one
multi-speaker panel with crosstalk, one with background music or a
phone-quality remote guest.

```bash
python scripts/fetch_audio.py "<news_clip_url>" --name anchor_mono --section "00:10:00-00:20:00"
```

Output lands in `audio/<name>.wav` as 16 kHz mono (what Whisper expects).

## 3. Transcribe (CPU by default)

```bash
python scripts/transcribe.py audio/anchor_mono.wav                     # OpenVINO, CPU, Urdu forced
python scripts/transcribe.py audio/anchor_mono.wav --language auto     # language ID
python scripts/transcribe.py audio/anchor_mono.wav --backend faster-whisper  # CTranslate2 CPU + VAD
python scripts/transcribe.py audio/anchor_mono.wav --initial-prompt "Imran Khan, Nawaz Sharif, Islamabad"
```

Prints timestamped segments, elapsed time, and realtime factor; writes the
plain transcript to `transcripts/<clip>.<backend>.txt` and a caption track
to `transcripts/<clip>.<backend>.vtt` (WebVTT).

## 4. Audit the script before trusting numbers

```bash
python scripts/script_audit.py transcripts/anchor_mono.openvino.txt --language ur
python scripts/script_audit.py transcripts/english_test.openvino.txt --language en
```

Flags romanized (Latin) output and Arabic-drift codepoints (`ي ك ة` instead of
`ی ک ہ`) — both silently corrupt WER if unchecked. `--language` picks which
script to validate against (`ur`/`en`/`auto`); omit it and it defaults to `ur`.

## 5. Normalized WER

Hand-correct ~5 minutes of reference into `refs/<clip>.txt` (see
`refs/README.md` for the full workflow — this step requires a human
listening to the audio, it can't be generated), then:

```bash
python scripts/wer_eval.py refs/anchor_mono.txt transcripts/anchor_mono.openvino.txt --raw
```

Normalization (shared with the audit, `scripts/urdu_norm.py`): NFC, Arabic→Urdu
codepoint folding, diacritic stripping, punctuation removal. Raw WER on Urdu is
close to meaningless without it.

## 6. Live multi-channel dashboard

```powershell
# Small Model (Recommended for real-time live streaming with zero delay):
$env:STT_MODEL="models/whisper-small-int8-ov"; $env:STT_WORKER_THREADS="1"; .venv\Scripts\python webapp/server.py

# Large-v3 Model (For maximum accuracy):
$env:STT_MODEL="models/whisper-large-v3-int8-ov"; $env:STT_WORKER_THREADS="1"; .venv\Scripts\python webapp/server.py
```

Open `http://127.0.0.1:8000`:

- **Add a channel**: paste a YouTube/live URL (restricted to an allow-listed
  set of domains — `scripts/url_safety.py`), give it a name, pick a
  language, click **Add channel**. It starts transcribing server-side
  immediately and keeps running even if you close the tab — reopen the
  dashboard and it's still there.
- **Quick add**: one-click buttons for eight major Pakistani news channels
  (Dunya News, Samaa TV, Express News, 92 News HD, BOL
  News, Aaj News) — handles verified via live YouTube search, not guessed.
  These default to `language: ur` (not `auto`) — see the language note
  below for why. The button greys out with a ✓ once that channel is active
  and re-enables when you stop it. Edit `FAMOUS_PK_CHANNELS` in
  `webapp/static/index.html` to add more.
- **Microphone**: click **Start microphone** to add your own mic as another
  channel, transcribed the same way as any stream.
- Add as many channels as you want; each gets its own card with live
  partial text and a scrolling log of finalized captions below it — this is
  the "multiple news channels at once" case.
- **Video/audio preview**: each channel's video plays muted in its card.
  Click **Listen** to unmute one — it automatically mutes whichever channel
  was previously playing, so only one is ever audible at a time. Some
  channels disable YouTube embedding entirely (a channel-owner setting, not
  something this project controls); those show "Preview blocked by
  channel" but keep transcribing normally — the caption pipeline pulls
  audio server-side, independent of the browser video embed.
- **Language**: default is **Auto**, which detects Urdu vs. English *per
  segment* — the right choice for mixed-language broadcast speech (a host
  who code-switches mid-sentence). Force `ur`/`en` per channel if you know
  the content is single-language.

Runs on CPU out of the box; every knob (backend, device, model, worker
concurrency, Segmenter tuning, auth, resource limits) is environment-
configurable — see the module docstring in `webapp/server.py` for the full
`STT_*` reference, including `STT_WORKER_THREADS` (how many channels can
transcribe truly in parallel — each thread loads its own model instance, so
RAM cost is N × model size).

Before exposing this beyond `127.0.0.1`, set `STT_WS_TOKEN` — see the
security note in that docstring.

### REST API

- `GET /channels` — list active channels and their status
- `POST /channels` — `{"name", "source": "youtube", "url", "language"}` (mic
  channels need live audio and can only start from the browser over `/ws`)
- `DELETE /channels/{id}` — stop a channel
- `GET /channels/{id}/segments?limit=50` — recent transcript history (needs DB)
- `GET /health` — backend/device/worker-pool/DB status

## 7. The corpus (for future fine-tuning)

Every finalized segment from every channel — mic or stream — is written to
Postgres (`db/schema.sql`: `channels` + `segments` tables) with language,
timing, inference time/RTF, and which backend/device produced it. That's
the growing dataset the requirements call for: once you have enough real
broadcast segments, `segments` is what you'd export from to build LoRA
fine-tuning splits (see `PRODUCTION_READINESS.md` for what that next step
still needs — real hand-corrected references and GPU compute, neither of
which persistence alone provides).

```sql
docker compose exec postgres psql -U stt -d stt -c \
  "select channel_id, language, count(*), avg(rtf) from segments group by 1,2;"
```

## GPU / CUDA (optional)

Everything above runs on CPU by default. If you have GPU hardware:

```bash
# Intel iGPU (Iris Xe, UHD, Arc) via OpenVINO — CLI:
python scripts/transcribe.py audio/clip.wav --device GPU
# webapp:
STT_DEVICE=GPU python webapp/server.py

# NVIDIA via CTranslate2 — CLI:
python scripts/transcribe.py audio/clip.wav --backend faster-whisper --device cuda
# webapp:
STT_BACKEND=faster-whisper STT_DEVICE=cuda python webapp/server.py
```

Both backends fall back to CPU automatically if the requested device fails
to load (driver missing, no compute runtime, etc.) — you'll see a warning in
the log, not a crash.

**Latency note, two separate effects measured on this project's own
8-core/15GB CPU dev machine:**
1. **Model speed, even with one channel**: `small-int8` ran at 1.2–2.6×
   realtime — genuinely live. `large-v3-int8` ran at **0.2× realtime for a
   single channel** (39s to transcribe 8s of audio) — this is why the
   dashboard defaults to `small-int8`, not `large-v3-int8`'s better
   accuracy; a live captioning server that can't keep up with real time
   defeats the point. This matches the CPU caveat in the model table below.
2. **Channel count vs. `STT_WORKER_THREADS`**: beyond N = worker threads,
   extra concurrent channels queue behind the busy ones — a hardware limit,
   not a code one. Measured worst case: 8 channels sharing 2 worker threads
   on `large-v3-int8` compounded to 0.21× realtime — captions still worked,
   just minutes behind. The dashboard warns about this directly (a banner
   appears whenever channel count exceeds `STT_WORKER_THREADS`, and any
   segment slower than realtime is flagged inline) rather than leaving you
   to wonder why nothing's appearing.

For real accuracy at live speed with many simultaneous channels, GPU/CUDA
is what makes that realistic — CPU is for development, evaluation, and
modest channel counts with a fast model.

### Model notes

| Model | Why / why not |
|---|---|
| `small-int8` (webapp/live default) | Measured 1.2–2.6× realtime on CPU — genuinely live. Lower accuracy than large-v3; good enough to confirm the pipeline works, worth upgrading once you have GPU/CUDA |
| `large-v3-int8` (CLI default) | Best Urdu quality; measured **0.2× realtime on CPU even for a single channel** — fine for the CLI's one-shot/offline use (`scripts/transcribe.py`, `scripts/wer_eval.py`), not for the live dashboard on CPU. Use it in the dashboard only with GPU/CUDA (`STT_DEVICE=GPU`/`cuda`) or if you explicitly accept multi-second-to-minute delay |
| `large-v3-int4` | Try if int8 feels too slow but you still want better-than-small accuracy; verify quality on your own clips (untested in this project so far) |
| `large-v3-turbo-int8` | Fast, but decoder distillation disproportionately hurts low-resource languages — benchmark against large-v3 on your Urdu clips before adopting |

- OpenVINO backend has no built-in VAD; for broadcast audio with long music
  beds, compare against the `faster-whisper` backend (`vad_filter=True`) to
  see how much hallucination VAD is saving you.
- For live/low-latency use, GPU or CUDA is what makes realtime feasible —
  CPU is for development, evaluation, and offline batch transcription.

## Tests

```bash
pip install -r requirements-test.txt
python -m pytest tests/ -v
```

Covers the pure-logic modules (`urdu_norm`, `script_audit`, `wer_eval`,
`backends`, `url_safety`, `webvtt`) — no GPU, model download, or audio file
needed. Runs in CI on every push (`.github/workflows/ci.yml`, lint + pytest).

See `PRODUCTION_READINESS.md` for what's hardened vs. what's still blocked
on real hardware/data.
