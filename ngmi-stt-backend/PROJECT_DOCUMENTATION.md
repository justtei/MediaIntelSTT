# STT Model Framework — End-to-End Project Documentation

**Generated:** 2026-08-25
**Repo root:** `c:\Amjad Ali\STTModelFramework`
**Current branch:** `main` (1 commit: `8a27d8c Initial commit: STT model framework`)

This document describes what the project is, how every piece fits together, what
has actually been verified to work, and every issue found while auditing the
codebase and its current working-tree state (including files that aren't
committed yet).

---

## 1. What this project is

A Windows-only toolkit for running **OpenAI Whisper speech-to-text on an Intel
integrated GPU (Iris Xe)** via **OpenVINO**, purpose-built around **Urdu**
transcription quality, with English as a secondary/comparison language. It has
two faces:

1. **A CLI pipeline** (`scripts/`) — download a model, fetch real broadcast
   audio, transcribe it, sanity-check the script (catch romanization / wrong
   Unicode drift), and compute a normalized Word Error Rate against a
   hand-corrected reference.
2. **A live web app** (`webapp/`) — a FastAPI server + single-page vanilla-JS
   UI that streams microphone audio, bundled demo clips, or a remote
   URL (YouTube/radio/etc. via `yt-dlp`) through the same GPU pipeline in
   near-real-time, over a WebSocket.

The stated hardware target (from `README.md`) is a specific machine: **Intel
Iris Xe iGPU, 15.7 GB shared RAM**. The framework is explicitly scoped for
*evaluation*, not production/live deployment — the README says large-v3-int8
runs "well below realtime on Iris Xe — fine for evaluation, not live use."

---

## 2. Repository layout

Only 12 files are actually tracked in git; everything else (models, venv,
vendored tools, fetched audio, generated transcripts) is `.gitignore`d and
regenerated locally per machine.

```
STTModelFramework/
├── README.md                    tracked — the only user-facing docs before this file
├── .gitignore                   tracked — ignores .venv/ models/ tools/ audio/ transcripts/ *.log __pycache__/
├── scripts/
│   ├── audio_io.py              tracked — shared ffmpeg decode-to-PCM helper
│   ├── download_models.py       tracked — pulls OpenVINO IR Whisper models from HF Hub
│   ├── fetch_audio.py           tracked — yt-dlp wrapper for grabbing real broadcast clips
│   ├── transcribe.py            tracked — CLI: file in, transcript out (OpenVINO or faster-whisper)
│   ├── script_audit.py          tracked — flags romanization / Arabic-vs-Urdu codepoint drift
│   ├── urdu_norm.py             tracked — shared Urdu text normalizer (used by audit + WER)
│   └── wer_eval.py              tracked — normalized WER/CER via jiwer
├── webapp/
│   ├── server.py                tracked — FastAPI + WebSocket live-transcription server
│   └── static/index.html        tracked — single-page UI (mic / demo clips / remote stream)
├── models/                      GITIGNORED, present locally (~1.75 GB)
│   ├── whisper-large-v3-int8-ov/   1.5 GB — default model
│   └── whisper-small-int8-ov/      245 MB — fast-iteration model
├── tools/                       GITIGNORED, present locally (~522 MB)
│   ├── ffmpeg-master-latest-win64-gpl/   vendored ffmpeg/ffprobe/ffplay binaries
│   └── deno/deno.exe                     lets yt-dlp solve YouTube JS player challenges
├── audio/                       GITIGNORED — 2 test clips (english_test.wav, news_test.wav)
├── transcripts/                 GITIGNORED — 2 output transcripts from past runs
├── .venv/                       GITIGNORED — Python 3.11.9 virtualenv (~617 MB)
├── .claude/                     UNTRACKED — Claude Code local settings (not project code)
├── kernel.errors.txt            UNTRACKED — GPU kernel-compile warning dump (see §7.1)
└── webapp/kernel.errors.txt     UNTRACKED — duplicate of the above
```

Total working-tree footprint is **~2.9 GB**, almost entirely models/venv/tools
that a fresh clone must regenerate — the actual source is ~700 lines of Python
+ one HTML file.

---

## 3. Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │              OpenVINO GPU pipeline          │
                 │  openvino_genai.WhisperPipeline                │
                 │  (models/whisper-large-v3-int8-ov, device=GPU) │
                 └───────────────▲─────────────────▲─────────────┘
                                 │                 │
              ┌──────────────────┘                 └──────────────────┐
              │                                                        │
   ┌──────────┴──────────┐                                 ┌──────────┴──────────┐
   │   CLI: transcribe.py │                                 │   webapp/server.py    │
   │  file → PCM (ffmpeg) │                                 │  1 GPU worker thread   │
   │  → pipe.generate()   │                                 │  (ThreadPoolExecutor,  │
   │  → transcripts/*.txt │                                 │   max_workers=1)       │
   └──────────┬───────────┘                                 └──────────┬─────────────┘
              │                                                        │
   ┌──────────┴───────────┐                          ┌─────────────────┼─────────────────┐
   │ script_audit.py       │                          │ Segmenter (mic) │ Segmenter (stream)│
   │ (romanization / drift │                          │ silence-cut     │ silence-cut       │
   │  detector)             │                          │ + rolling       │ + rolling         │
   └──────────┬─────────────┘                         │ partial decode  │ partial decode    │
              │                                        └────────┬────────┴─────────┬─────────┘
   ┌──────────┴───────────┐                                     │                  │
   │ wer_eval.py            │                          browser mic (WS binary)   StreamPuller
   │ (needs refs/*.txt,      │                                                    yt-dlp | ffmpeg -re
   │  never created — §7.3)  │                                                    (server-side pull)
   └────────────────────────┘
```

Shared building block: `urdu_norm.py` — NFC normalization, Arabic→Urdu
codepoint folding (`ي→ی`, `ك→ک`, `ة/ه→ہ`, `ى→ی`), diacritic stripping,
punctuation removal — used by both `script_audit.py` and `wer_eval.py` so the
"is this actually Urdu" check and the WER score agree on what counts as noise.

---

## 4. End-to-end CLI workflow

### 4.1 One-time setup
```powershell
python scripts\download_models.py                # large-v3 int8 (~1.6 GB)
python scripts\download_models.py small-int8      # optional fast model
python scripts\smoke_test.py                      # proves GPU path works
```
- `download_models.py` maps 4 short keys (`large-v3-int8`, `large-v3-int4`,
  `small-int8`, `large-v3-turbo-int8`) to `OpenVINO/whisper-*-ov` HF repos and
  calls `snapshot_download`. Only **two** of the four (`large-v3-int8`,
  `small-int8`) have actually been downloaded in this working copy; `int4` and
  `turbo-int8` are documented as options but untested here.
- `smoke_test.py` enumerates OpenVINO devices, hard-fails if no `GPU*` device
  is present, then compiles the model and runs 3 seconds of synthetic noise
  through it — this is an infrastructure check, not a quality check.

### 4.2 Get real test audio
```powershell
python scripts\fetch_audio.py "<url>" --name anchor_mono --section "00:10:00-00:20:00"
```
`fetch_audio.py` shells out to `yt-dlp` (as `python -m yt_dlp`), extracts
audio, and post-processes with ffmpeg to 16 kHz mono WAV directly via
`--postprocessor-args`. `--section` uses yt-dlp's `--download-sections` +
`--force-keyframes-at-cuts` to clip without downloading the whole source.
README explicitly steers away from canned benchmark corpora (Common Voice /
FLEURS) toward real broadcast audio.

### 4.3 Transcribe
```powershell
python scripts\transcribe.py audio\anchor_mono.wav                     # GPU, Urdu forced
python scripts\transcribe.py audio\anchor_mono.wav --language auto     # language ID
python scripts\transcribe.py audio\anchor_mono.wav --backend faster-whisper  # CPU baseline
```
`audio_io.py` decodes *any* container ffmpeg understands to float32 mono
16 kHz PCM in one subprocess call (`-f f32le -`). `transcribe.py` then either:
- **OpenVINO path**: loads `openvino_genai.WhisperPipeline` on `--device`
  (default `GPU`), forces `<|ur|>` unless `--language auto`, calls
  `pipe.generate()` once on the whole array (no chunking/streaming — the CLI
  path is not designed for long files), and reconstructs segment lines from
  `result.chunks`.
- **faster-whisper path**: CPU, int8 compute type, `vad_filter=True`,
  `beam_size=5` — a deliberately different engine used as a sanity baseline,
  not swapped in for production use.

Output: console printout of timestamped lines + realtime factor, plus a
plain-text transcript written to `transcripts\<clip>.<backend>.txt`.

### 4.4 Audit the script
```powershell
python scripts\script_audit.py transcripts\anchor_mono.openvino.txt
```
Counts alphabetic characters, buckets them into Perso-Arabic / Latin / other,
checks for Urdu-specific letters (`ٹڈڑںھہۃکگیے`), flags 4 specific
Arabic-drift codepoints, and passes only if **≥90% Perso-Arabic AND at least
one Urdu-specific marker letter is present**. This exists because Whisper can
silently degrade into romanized output or bleed generic-Arabic codepoints
instead of Urdu ones, both of which would corrupt a WER score without this
check. **Caveat:** this heuristic assumes the target language is Urdu — see
§7.4, it currently misclassifies correct English output as "SUSPECT."

### 4.5 Normalized WER
```powershell
python scripts\wer_eval.py refs\anchor_mono.txt transcripts\anchor_mono.openvino.txt --raw
```
Applies `urdu_norm.normalize()` to both reference and hypothesis, then scores
with `jiwer.process_words` (word-level sub/del/ins breakdown) and `jiwer.cer`.
`--raw` additionally prints unnormalized WER for comparison — the README notes
raw WER on Urdu is "close to meaningless" without normalization, given how
much of the "error" would otherwise just be codepoint/diacritic/punctuation
noise. **This step requires a hand-corrected `refs/<clip>.txt` file that does
not exist anywhere in this project** — see §7.3.

---

## 5. Live web app (`webapp/`)

### 5.1 Server (`server.py`)
FastAPI app, single GPU worker (`ThreadPoolExecutor(max_workers=1)`) so mic
and remote-stream transcription never contend for the GPU concurrently — they
queue instead. Model loads once at startup via the (deprecated) `@app.on_event("startup")`
hook.

**Three audio sources, one pipeline:**
1. **Browser mic** — an `AudioWorkletNode` in the page buffers ≥2048 samples
   and posts them over the WebSocket as raw binary float32 PCM.
2. **Demo clips** — `/demo-audio/{lang}` serves `audio/news_test.wav` (ur) or
   `audio/english_test.wav` (en); the browser decodes and routes it through
   the *same* AudioWorklet graph as the mic, so it exercises the identical
   client-side path.
3. **Remote/live streams** — `StreamPuller` spawns `yt-dlp -f bestaudio/best -o -`
   piped into `ffmpeg -re -i pipe:0 -f f32le pipe:1` (the `-re` flag re-paces
   output to real time so a VOD doesn't flood the segmenter), and the server
   reads 0.5 s chunks from ffmpeg's stdout in a thread executor. If `tools/deno/deno.exe`
   exists, yt-dlp is given `--js-runtimes deno:...` so it can solve YouTube's
   player-JS challenges.

**`Segmenter`** is the core streaming logic, one instance per source
(`mic_seg`, `stream_seg`):
- Buffers incoming chunks; flushes (transcribes + emits a final `segment`
  message) when either `MAX_SEG_S=8.0` s is hit (hard cap, bounds latency) or
  the trailing `SIL_WIN_S=0.7` s window's RMS drops below `SIL_RMS=0.010`
  after at least `MIN_SEG_S=2.5` s buffered.
- Segments whose whole-buffer RMS is below `SKIP_RMS=0.004` are dropped
  without transcribing, specifically to avoid Whisper hallucinating text on
  near-silence.
- Independently, once the buffer holds `PARTIAL_MIN_S=1.2` s it also fires a
  non-blocking "partial" decode (throttled to one per `PARTIAL_GAP_S=0.8` s)
  so the UI can show live, provisional text ahead of the final segment. A
  generation counter (`self.gen`) invalidates a partial result if the buffer
  was flushed while that decode was in flight.

**WebSocket protocol** (`/ws`), JSON text frames from client, binary frames
for mic PCM:
- client → server: `{type:"config", language}`, `{type:"flush"}`,
  `{type:"stream_start", url}`, `{type:"stream_stop"}`, and raw binary PCM
- server → client: `{type:"partial", text}`, `{type:"segment", text, t0, t1,
  infer_s, rtf, lang, reason, source}`, `{type:"stream_status", state, ...}`

### 5.2 Frontend (`webapp/static/index.html`)
Single self-contained HTML file (no build step, no framework). Notable
details: RTL auto-detection via a Unicode range regex (`/[؀-ۿ]/`) applied to
both partial and final text to switch `dir="rtl"` and a Nastaliq/Urdu font
stack; a level meter driven by the same worklet callback that ships audio;
per-segment display of inference time and realtime factor (`rtf`) pulled
straight from the server payload — useful for judging whether a given segment
duration was cheap or expensive to decode.

---

## 6. Dependencies & environment

- **Python**: 3.11.9, in a local venv at `.venv/` (correctly created *at this
  project's current path* — confirmed working, not stale; see §7.5 for a
  related but different issue).
- **Key packages actually installed** (from `pip freeze` in `.venv`):

  | Package | Version |
  |---|---|
  | openvino | 2026.2.1 |
  | openvino-genai | 2026.2.1.0 |
  | openvino-tokenizers | 2026.2.1.0 |
  | faster-whisper | 1.2.1 |
  | ctranslate2 | 4.8.1 |
  | fastapi | 0.141.1 |
  | uvicorn | 0.52.0 |
  | jiwer | 4.0.0 |
  | numpy | 2.4.6 |
  | huggingface_hub | 1.26.0 |
  | yt-dlp | 2026.7.4 |

- **Vendored, not installed system-wide**: ffmpeg/ffprobe/ffplay under
  `tools/ffmpeg-master-latest-win64-gpl/bin/`, and `deno.exe` under
  `tools/deno/` (only used to help yt-dlp bypass YouTube JS challenges).
- **No `requirements.txt`, `pyproject.toml`, or environment file exists
  anywhere in the repo** — see §7.2, this is the single biggest reproducibility
  gap.

---

## 7. Issues found

Ordered roughly by how much they'd bite someone picking this project up.

### 7.1 — Unresolved GPU kernel-compile warnings (open, cause not diagnosed)
`kernel.errors.txt` exists at both the repo root and `webapp/kernel.errors.txt`
(both untracked, both dated 2026-08-25, i.e. from the most recent local
session), containing two Intel Graphics Compiler (CISA) kernel-header errors:
```
Error in CISA routine with name: kernel
  Input V38 = [256, 260) intersects with V37 = [256, 260)

Error in CISA routine with name: kernel
  Explicit input 2 must not follow an implicit input 0
```
These are low-level Intel GPU shader-compilation diagnostics, not Python
tracebacks. Evidence suggests they are **not currently fatal**: `webapp/server.log`
from the same session shows the model loaded successfully ("ready in 15.5s")
and `transcripts/english_test.openvino.txt` was written minutes later in that
same run. So the GPU pipeline compiled and produced output despite these
errors being logged. What's **not known**: whether these warnings indicate a
kernel falling back to a slower/different code path, a driver version
mismatch on this specific Iris Xe unit, or silent accuracy loss on whatever
segment triggered them. This needs a real investigation (Intel graphics
driver version check, OpenVINO GPU plugin issue tracker) before it can be
called "harmless."

### 7.2 — No dependency manifest (reproducibility gap)
There is no `requirements.txt`, `pyproject.toml`, or `environment.yml`
anywhere in the repo. The only record of what to `pip install` and in what
order lives in `.claude/settings.local.json`'s permission-command history —
a Claude Code internal artifact, not project documentation, and not something
a new contributor would know to look at. Anyone cloning this repo fresh has
no authoritative way to reproduce the environment other than reverse-engineering
`import` statements across 9 files. **Recommendation:** pin a `requirements.txt`
(or `pyproject.toml`) from the current working `.venv`'s `pip freeze` output.

### 7.3 — WER evaluation has never actually been run
Step 5 of the README (`wer_eval.py`) requires a hand-corrected reference file
at `refs/<clip>.txt`. **No `refs/` directory exists anywhere in this project**,
and it isn't even in `.gitignore` (unlike `audio/`/`transcripts/`/`models/`,
which are). This means the accuracy-measurement half of the framework —
arguably the actual point of an STT *evaluation* framework — has never been
exercised end-to-end. There is no evidence anywhere in this repo of what WER
this setup actually achieves on Urdu audio. The two transcripts that do exist
(`transcripts/news_test.openvino.txt`, `transcripts/english_test.openvino.txt`)
are raw model output only, never scored against ground truth.

### 7.4 — `script_audit.py` false-positives on correct English output
Running the audit tool against the real `english_test.openvino.txt` transcript
(verified during this review) gives:
```
letters: 458
  Perso-Arabic script:   0.0%
  Latin (romanized?):  100.0%
Urdu-specific letters seen: 0 (NONE)
VERDICT: SUSPECT — check for romanization or wrong-language decode
```
This is **correct, expected English text** flagged as suspect, because the
heuristic hardcodes the assumption that the target language is Urdu (`arabic/n
>= 0.9 and markers`). The tool has no `--language` flag or English-aware
branch, so it cannot be used to validate the English demo path at all — a real
functional gap given the project explicitly supports English as a secondary
language (there's a whole English demo clip and UI button for it, and
`transcribe.py --language auto` exists specifically to test language ID).

### 7.5 — Stale absolute paths from a prior project location
Several files still reference `E:\AvaPro\STTModelFramework`, an earlier
location this project evidently lived at before being moved/copied to its
current path `c:\Amjad Ali\STTModelFramework`:
- `.claude/settings.local.json` — every cached permission command
- `webapp/server.err.log` and `webapp/server.log` (dated 2026-07-31, i.e. from
  before the move)
This is **cosmetic, not functionally broken** — the `.venv` itself was
verified to be correctly built at the *current* path (`pyvenv.cfg`'s
`command =` line points at `c:\Amjad Ali\STTModelFramework\.venv`, and
`import openvino_genai` succeeds), and the two `kernel.errors.txt` files and
today's `english_test.openvino.txt` are all dated after the move and use no
stale paths. Still, the old-path log files are misleading clutter if left in
place, and `.claude/settings.local.json` will keep offering irrelevant
`E:\AvaPro\...`-prefixed commands for autocomplete/permission matching.

### 7.6 — Uncommitted fix sitting in the working tree
`git status` shows 5 modified-but-unstaged files, all making the identical
one-line change: adding a guarded `sys.stdout.reconfigure(encoding="utf-8")`
before any print output —
`scripts/script_audit.py`, `scripts/smoke_test.py`, `scripts/transcribe.py`,
`scripts/wer_eval.py`, `webapp/server.py`. This is a real, correct fix (Windows
consoles default to a non-UTF-8 codepage, which would otherwise raise
`UnicodeEncodeError` or mangle output the moment Urdu script hits stdout) —
but it's inconsistently applied: `scripts/urdu_norm.py`, `scripts/fetch_audio.py`,
and `scripts/download_models.py` don't print Urdu text so they don't need it,
but it's worth double-checking no path was missed. **This diff has not been
committed**, so a fresh `git clone` of `main` would still hit the encoding
issue this fix addresses.

### 7.7 — `faster-whisper` CPU backend appears untested in this project
No `transcripts/*.faster-whisper.txt` output exists anywhere, despite
`--backend faster-whisper` being a documented, first-class option in both the
README and `transcribe.py`. There's no evidence this code path has ever
actually been run against real audio in this project (as opposed to just
being written). Given it's meant as the CPU sanity baseline to compare against
the GPU path, that comparison has apparently never been made.

### 7.8 — Untracked debug artifacts not covered by `.gitignore`
`kernel.errors.txt` (root) and `webapp/kernel.errors.txt` are untracked and
would get swept into a commit by an unwary `git add -A`/`git add .`. `.gitignore`
covers `*.log` but not `*.errors.txt`. Minor, but worth adding a rule (or
deleting these once §7.1 is understood) before they end up in git history.

### 7.9 — Web app: minor robustness/code-quality gaps
- `@app.on_event("startup")` is deprecated in FastAPI 0.141 (confirmed by the
  actual deprecation warning captured in `webapp/server.err.log`); it still
  works today but should move to a `lifespan` context manager before FastAPI
  removes it.
- `GET /demo-audio/{lang}` returns `{"error": ...}` with an implicit HTTP 200
  when the lang key or the clip file is missing, instead of a 4xx status —
  a client checking `response.ok` rather than parsing the body would miss the
  failure.
- The server hardcodes `device="GPU"` at startup with no CPU fallback and no
  try/except — if the Iris Xe device isn't enumerated (driver issue, different
  machine), the FastAPI startup hook raises an unguarded exception and the
  process dies with a raw traceback rather than a clear message (compare this
  to `smoke_test.py`, which does check `available_devices` first and exits
  cleanly).
- `StreamPuller` will run `yt-dlp` against *any* URL a client submits with no
  allow-listing. The server only binds to `127.0.0.1` today, so this isn't
  currently exploitable remotely, but it's worth flagging explicitly before
  anyone changes the bind host to `0.0.0.0` for LAN/demo access — at that
  point unrestricted server-side URL fetching becomes a real concern.

### 7.10 — No automated tests, no CI
There is no test suite (only the manual, human-run `scripts/smoke_test.py`),
and no `.github/workflows` or other CI configuration. All verification so far
has been ad hoc, interactive command runs (visible in `.claude/settings.local.json`'s
history) rather than anything repeatable or checked automatically.

### 7.11 — Model options documented but not fetched
The README table discusses `large-v3-int4` (try if int8 feels slow) and
`large-v3-turbo-int8` (fast but flags a specific caveat: "decoder distillation
disproportionately hurts low-resource languages — benchmark against large-v3
on your Urdu clips before adopting"). Neither has been downloaded in this
working copy, so that recommended benchmarking comparison — which the README
itself calls for — hasn't happened.

---

## 8. Summary: what's verified working vs. not

| Capability | Status |
|---|---|
| Model download (`large-v3-int8`, `small-int8`) | ✅ done, present on disk |
| GPU detected + Whisper compiles on Iris Xe | ✅ (per `smoke_test.py` design + `server.log` "ready in 15.5s") — but see §7.1 for unresolved kernel warnings |
| CLI transcription, OpenVINO backend, Urdu | ✅ produced real output (`transcripts/news_test.openvino.txt`) |
| CLI transcription, OpenVINO backend, English | ✅ produced real output (`transcripts/english_test.openvino.txt`) |
| CLI transcription, `faster-whisper` backend | ❌ no evidence it's been run (§7.7) |
| Script audit tool | ⚠️ works for Urdu, false-positives on English (§7.4) |
| Normalized WER scoring | ❌ never run — no `refs/` reference transcripts exist (§7.3) |
| Live web app (mic / demo clips) | ✅ server has been started and loaded the model successfully at least twice (2026-07-31, 2026-08-25) |
| Live web app (remote stream via yt-dlp) | ⚠️ extensive ad hoc testing visible in `.claude` history against YouTube/Dailymotion, but no committed evidence of a clean end-to-end run |
| Reproducible environment for a new clone | ❌ no dependency manifest (§7.2) |
| Encoding fix for Windows console output | ⚠️ written but uncommitted (§7.6) |

---

## 9. Recommended next steps, in priority order

1. **Commit the UTF-8 stdout fix** (§7.6) — it's correct and currently only
   protects whoever's local working tree it's sitting in.
2. **Freeze a `requirements.txt`** from the current `.venv` (§7.2) — highest
   leverage fix for anyone else (or a future you, on a fresh machine) trying
   to stand this project up.
3. **Create at least one `refs/<clip>.txt`** and run `wer_eval.py` for real
   (§7.3) — without this, there's no actual accuracy number backing the
   "large-v3-int8 is the Urdu quality pick" recommendation in the README.
4. **Add a `--language`/English-aware mode to `script_audit.py`**, or scope
   its docstring/usage to "Urdu-only" explicitly (§7.4).
5. **Investigate the CISA kernel errors** (§7.1) — check Intel graphics driver
   version, and whether OpenVINO's GPU plugin issue tracker has a known match
   for this error signature, before trusting GPU output uncritically.
6. **Clean up stale artifacts**: delete/gitignore the two `kernel.errors.txt`
   files once §7.1 is resolved, and be aware `.claude/settings.local.json`
   still carries `E:\AvaPro\...`-path commands from before the project moved.
7. **Run the `faster-whisper` CPU baseline** at least once and diff it against
   the OpenVINO GPU output on the same clip, per the README's own stated
   purpose for having two backends.
