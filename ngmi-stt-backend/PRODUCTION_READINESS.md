# Production readiness — status against requirements.md

Not committed yet (by request). Summarizes what changed in this pass and,
explicitly, what's still blocked by resources this environment doesn't have.

**CPU is the confirmed target for now** (no GPU available — user-confirmed,
not just an environment gap). Every default was flipped to CPU: `transcribe.py`
and `webapp/server.py` both default to `--device CPU` / no GPU-attempt at
startup, so there's no more "GPU load failed, falling back" warning on every
run. GPU (Intel iGPU via OpenVINO) and CUDA (NVIDIA via faster-whisper) are
fully implemented and untouched — switch back any time by setting
`--device GPU` / `STT_DEVICE=GPU` (OpenVINO) or `--device cuda` /
`STT_BACKEND=faster-whisper STT_DEVICE=cuda` (NVIDIA). No code changes needed
either direction.

## Done — code-level, verified

**Phase 0 (stabilization, issue backlog §3)**
- #1 `requirements.txt` (+ `requirements-test.txt`) frozen from a verified install
- #4 `script_audit.py --language {ur,en,auto}` — no longer false-positives on English
- #5 GPU→CPU (and CUDA→CPU) fallback, shared via `scripts/backends.py`
- #8 pytest suite (36 tests) + GitHub Actions CI (lint + test) at `.github/workflows/ci.yml`
- #9 FastAPI `lifespan` (deprecated `on_event` removed)
- #10 `/demo-audio/{lang}` returns real 404s instead of 200+error-body
- #13 `.gitignore` already covered `*.errors.txt` (pre-existing)
- #2 UTF-8 stdout fix was already committed in repo history (pre-existing)

**Phase 1 (production hardware path) — code made ready, not benchmarked**
- `scripts/backends.py`: one interface, two backends (OpenVINO, faster-whisper),
  used identically by the CLI and the webapp. `--device auto` tries the
  accelerator then CPU on both.
- `faster-whisper` now accepts `--device cuda` (was hardcoded to `cpu`).
- `webapp/server.py` is backend-agnostic via `STT_BACKEND`/`STT_DEVICE`/`STT_MODEL`
  env vars — the CUDA path is one env var away from being live, once CUDA hardware exists.
- Segmenter buffering constants (`MAX_SEG_S` etc.) are env-configurable
  (`STT_MAX_SEG_S`, ...) instead of hardcoded, so they can be retuned for
  CUDA-class speed without a code change.

**Phase 3 (production hardening)**
- #11 `scripts/url_safety.py`: blocks non-http(s) schemes and IP-literal hosts
  (SSRF/cloud-metadata protection), enforces a domain allow-list
  (`STT_ALLOWED_STREAM_HOSTS`) before any URL reaches yt-dlp/ffmpeg
- WebSocket token auth (`STT_WS_TOKEN`), constant-time comparison
- `GET /health` (200 ready / 503 loading, reports backend+device+model)
- Resource limits: WS binary frame cap (`STT_MAX_WS_BINARY_BYTES`), stream
  first-data timeout (`STT_STREAM_FIRST_DATA_TIMEOUT_S`), initial-prompt
  length cap (`STT_MAX_INITIAL_PROMPT_CHARS`)
- Structured logging (`logging` module, `STT_LOG_LEVEL`) replacing `print()`
  in the server; every per-connection failure path now logs + reports
  instead of crashing the connection
- Malformed-JSON / non-object WS messages handled gracefully

**FR7 (custom vocabulary)** — `--initial-prompt` on the CLI, `initial_prompt`
in the WS `config` message, `STT_INITIAL_PROMPT` as a server default.

**Phase 4 (partial)** — WebVTT export (`scripts/webvtt.py`); `transcribe.py`
now writes a `.vtt` alongside every `.txt` transcript.

Verified: 36/36 pytest passing, `pyflakes` clean across `scripts/`,
`webapp/server.py`, `tests/`, manual browser run confirming health-status
display, graceful demo-clip 404 handling, and a live SSRF-blocked stream
request end-to-end (screenshot-verified).

## Multi-channel dashboard + persistence corpus (new requirement)

Pivoted from a single-source demo tool to a server-managed multi-channel
architecture: any number of YouTube/live-stream channels plus a mic channel
run **concurrently**, each independently segmented/transcribed, broadcasting
to every connected dashboard and persisting every finalized segment.

- **`scripts/db.py` + `db/schema.sql`**: Postgres via `docker compose up -d`
  (`docker-compose.yml`). Chose Postgres over SQLite/MongoDB because this
  data is explicitly headed toward ML fine-tuning corpus export — real
  indexes over channel/language/time, FK integrity, comfortable at the row
  counts a 24/7 multi-channel service accumulates, free, trivial to run
  locally. `channels` + `segments` tables; `segments` is the corpus.
- **`webapp/server.py` channel registry**: `Channel` dataclass, `CHANNELS`
  dict, `start_channel()`/`stop_channel()`, broadcast to all dashboard
  `SUBSCRIBERS` tagged by `channel_id`. YouTube channels run independently
  of any browser tab (start once, keep transcribing); mic channels are tied
  to the owning WebSocket connection and stop when it disconnects.
- **Concurrency**: `STT_WORKER_THREADS` (default 2) — each worker thread
  loads its **own backend instance** rather than sharing one pipeline across
  threads, since OpenVINO/CTranslate2 pipelines aren't documented as safe
  for concurrent `generate()`/`transcribe()` calls from multiple threads.
  Pool is a `queue.Queue`, thread-safe by construction. RAM cost is N ×
  model size — defaulted conservatively for this dev machine's tight
  memory headroom; raise it on real deployment hardware.
- Graceful degradation: server runs fully (transcription, dashboards, REST)
  with **no Postgres running** — `db.is_connected()` gates every DB call;
  segments just aren't persisted, surfaced via `/health`'s `"database"`
  field and the dashboard's `db ✗ (not saving)` indicator. Confirmed by
  actually running the server with Postgres down (connection-refused
  traceback logged, not raised — server kept serving).
- FR7 (custom vocabulary) and language selection are now **per-channel**,
  not per-connection — each news channel can have its own language setting
  and prompt, matching "multiple channels at once" being independent.
- Frontend rebuilt as a card-grid dashboard: add-channel form, mic button,
  one card per channel with live partial + scrolling finalized captions
  ("text below" per requirement), status pills, per-channel language
  selector and stop button. Old single-source demo-clip UI removed
  (superseded).

### Two real bugs found and fixed via live end-to-end testing

Both would have shipped invisibly if verification had stopped at unit tests:

1. **ffmpeg segfaulted reading live YouTube HLS.** The vendored static
   ffmpeg 7.0.2 (johnvansickle.com stable release) crashed with SIGSEGV
   specifically when reading a live YouTube HLS manifest — reproduced with
   a bare `ffmpeg -i <manifest_url>` command, independent of this project's
   code, ruling out a code bug. Root-caused by testing yt-dlp's own
   internal ffmpeg-as-downloader path in isolation. Fix: swapped in an
   ffmpeg **master** build (BtbN/FFmpeg-Builds `linux64-gpl` latest), which
   does not crash on the same input — confirmed with real audio data
   (RMS 0.066, 5.0s, non-silent) before wiring it back into the app.
2. **`StreamPuller` architecture simplified as a side effect of the fix.**
   Rather than keep piping yt-dlp's own ffmpeg-downloader output into a
   second ffmpeg process (the original design, and the code path that hit
   bug #1), `StreamPuller` now resolves the direct media URL once via
   `yt-dlp -g` and hands that straight to a single ffmpeg process. Fewer
   processes, avoids yt-dlp's internal ffmpeg downloader entirely, lower
   latency to first audio.
3. **(Also fixed, found in the same pass)** `tools/deno/deno.exe` was a
   Windows-only path check — `StreamPuller` silently never found deno on
   Linux, degrading YouTube extraction per yt-dlp's own warning. Fixed to
   check `PATH` and any `deno*` binary under `tools/deno/`. Deno itself was
   installed (`curl -fsSL https://deno.land/install.sh | sh`, vendored into
   `tools/deno/`).
4. **Two ordering/concurrency bugs found only because two real channels
   were added back-to-back**: (a) `start_channel()` originally awaited the
   full URL-resolution step (up to 30s) before broadcasting `channel_added`,
   so the dashboard showed nothing until resolution finished, and multiple
   channels would have serialized behind each other on one connection —
   fixed to broadcast immediately and resolve in the background. (b) the
   channel registry's `status` field never updated after a stream error, so
   a dashboard that connected *after* a failure would see a stale "live" —
   fixed by syncing `channel.status` off `stream_status` events.

### Verified live, with real production data

Two real, currently-live YouTube news channels (Al Jazeera English, Sky
News) added concurrently through the actual dashboard UI, both transcribing
simultaneously and correctly:

```
Al Jazeera English — live — 1.21–2.65× realtime
  "capsized like this, also in that area of the Mediterranean."
  "Well indeed, some of the officials here are saying that the vessel..."
Sky News — live — 1.65–1.8× realtime
  "the most difficult part. The thing with this sort of vessel is
   that if they do go down, they don't do go down quite quickly."
```

Both faster than realtime on `small-int8`/CPU with `STT_WORKER_THREADS=2`.
SSRF allow-list, channel add/stop, and graceful-DB-down were all also
re-verified against the final code. Channels were stopped cleanly after
the test; confirmed no orphaned `yt-dlp`/`ffmpeg` processes.

**Not yet verified**: actual Postgres persistence end-to-end (segment rows
landing in `segments`). Blocked on Docker access in this session — see below.

## "Captions not showing" was CPU oversubscription, not a rendering bug

Reported: after following the Quick Start literally (default `large-v3-int8`,
default `STT_WORKER_THREADS=2`) and adding all 8 quick-add channels, video
played in every card but **no caption text ever appeared**. Investigated as
a potential frontend/rendering regression first — it wasn't one.

**Root cause, confirmed empirically**: 8 concurrent channels sharing 2 CPU
worker instances, each running `large-v3-int8`, on a machine already under
load (system load average 9–10 on 8 cores). Measured real-time factor
dropped to **0.21× — 38.77s to transcribe an 8-second segment** — roughly
5× slower than the 1.2–2.6× RT measured earlier with 1-2 channels on an
otherwise-idle machine. With 8 channels cycling through 2 slow workers,
per-channel latency compounds into minutes; long enough that "no captions
yet" looked identical to "broken."

**Verified the fix directly**: stopped 6 of 8 channels (down to 2, matching
the worker count) — captions started appearing within seconds, correctly
rendered in Nastaliq (confirmed via zoomed screenshot: `موسیقی`, a real
segment from Samaa TV). The rendering pipeline was never broken; it was
never getting CPU time to finish.

**Fixed**: the dashboard now proactively warns instead of failing silently.
- A capacity banner (`checkOversubscription()` in `webapp/static/index.html`)
  compares live channel count against `/health`'s `worker_threads` and warns
  when channels outnumber workers, with concrete next steps (stop channels /
  use `small-int8` / raise `STT_WORKER_THREADS` only if real cores back it).
- Each segment's RTF is now visually flagged (`⚠ slower than realtime`) when
  it drops below 1×, so lag is visible per-channel, not just guessed at.

**Guidance**: on CPU, don't run more channels than `STT_WORKER_THREADS`,
and size `STT_WORKER_THREADS` to real spare cores, not just "more is
better" — extra threads sharing already-saturated cores make every
channel slower, not just the new one. `large-v3-int8` is the quality
pick but the slowest; `small-int8` is far more viable for testing several
channels on CPU. GPU/CUDA is the actual answer for many channels at low
latency — see the Latency note in README.md.

## Nastaliq font rendering + an Urdu/Hindi language-ID finding

Fixed two real gaps found while verifying captions render properly:
1. `.card .partial[dir="rtl"]` (the live, in-progress caption) had lost its
   `font-family` in the multi-channel rewrite — only finalized segments got
   the Nastaliq stack, live text silently fell back to plain sans-serif.
2. The `font-family: "Noto Nastaliq Urdu", ...` stack only works if that
   font happens to be installed locally — on a machine without it, every
   fallback in the stack would also miss and it'd render as generic
   sans-serif, invisibly. Added an actual Google Fonts `<link>` for Noto
   Nastaliq Urdu (free, open-license, the standard web choice for this
   script) so it renders correctly regardless of the viewer's system fonts.
   Bumped line-height to ~2 for both partial and finalized RTL text —
   Nastaliq's diagonal stacking needs more vertical room than Naskh-style
   Arabic fonts or Latin text.

Verified live (Samaa TV, `language: ur` forced): both live-partial and
finalized captions render in genuine Nastaliq calligraphy (diagonal,
cascading letterforms — confirmed via zoomed screenshot, not just "text
appeared"), correctly right-aligned.

**Real finding, not a bug**: the same channel with `language: auto` produced
one segment in **Devanagari script** — Whisper's language auto-detection
occasionally misidentifies Urdu speech as Hindi (the spoken languages are
very close; short ~2.5–8s segments give the detector little to go on). This
is silently wrong for this project's purpose — the corpus is meant to be
Urdu, and Hindi-script output would contaminate it even though the words
are often intelligible. **Forcing a channel's language to `ur` (not `auto`)
avoided it entirely** in repeated testing. Given every Urdu channel tested
(Dunya News, Samaa TV) is Urdu-primary with only
occasional English words, `auto`'s main value — correctly identifying
pure-English segments — is a smaller win than the risk of silent Hindi
contamination. Recommendation for real deployment: default new Urdu-market
channels to `ur`, not `auto`; reserve `auto` for genuinely bilingual sources
where a large fraction of segments are pure English. Left the UI default as
`auto` rather than changing it unilaterally — this is a content-quality
tradeoff worth deciding deliberately, not a default to flip silently.

## Quick-add row for major Pakistani news channels

Eight one-click buttons (`FAMOUS_PK_CHANNELS` in `webapp/static/index.html`):
Dunya News, Samaa TV, Express News, 92 News HD, BOL
News, Aaj News. Every handle was verified via live YouTube search before
use — a repeat of the earlier lesson: a guessed ARY handle 404'd, the real
one only surfaced by actually checking. Quick-add explicitly sends
`language: "ur"` rather than whatever the manual form's dropdown holds,
directly applying the Hindi-misdetection finding below — these are Urdu-
primary channels, so skip `auto`'s risk on purpose. Buttons track live
state: disabled with a ✓ while that channel is active, re-enabled when
stopped (matched by URL, not id, so it survives page reloads).

## Video/audio preview, exclusive-audio switching

Each YouTube channel card now embeds the actual live video via the YouTube
IFrame Player API (`webapp/static/index.html`), muted by default. A
**Listen** button per card unmutes that one and mutes whichever channel was
previously active — one audible channel at a time, matching the request.
Server side, `_resolve_media_url` (renamed from a single-URL resolver) now
uses `yt-dlp -j` instead of `-g` — same single network round-trip, but also
returns the video id, threaded through `StreamPuller` → the `stream_status`
event → `Channel.video_id` → `channel.public()`, so both brand-new and
reconnecting dashboards get it.

**Real constraint found, not a bug**: some channels
disable YouTube embedding at the channel-owner level — the IFrame player
fires an `onError`, showing YouTube's bare "Video unavailable" box. Since
this is unrelated to whether transcription works (that's a separate
server-side `yt-dlp -j` + direct ffmpeg pull, not the browser embed),
added an `onError` handler that disables the Listen button with "🚫 Preview
blocked by channel — captions still work" instead of leaving a broken
player. Confirmed: those channels kept transcribing correctly throughout.

### Verified live with real Urdu news channels

Dunya News(`@DunyanewsOfficial`), and Samaa TV (`@Samaatv`) — four real, live Pakistani
news broadcasts — added **concurrently** (`STT_WORKER_THREADS=2`,
`small-int8`/CPU) through the actual dashboard. All four transcribed real
Urdu speech correctly, e.g.:

```
Dunya News — live, video preview working (real broadcast visible)
Samaa TV — live, video preview working
```

Confirmed handle-guessing is unreliable (`@arynewsofficial` 404'd; the real
handle is `@ArynewsTvofficial`) — verified all four via YouTube search
before use, not guessed blind.

Two more real bugs found via this test and fixed:
5. **Stale status after page reload**: `addChannelCard()` hardcoded the pill
   to "starting" regardless of the channel's actual current status in the
   `channels_snapshot`/`channel_added` payload — a dashboard reconnecting to
   an already-live channel showed "starting" until the *next* transcription
   event happened to fire. Fixed to read `channel.status` from the payload.
6. **Redundant `onReady` mute() call** removed from player creation — it
   raced a fast "Listen" click (playerVars `mute:1` already guarantees the
   required muted-autoplay start; the extra call could clobber an unmute
   the user had already triggered).

**Observed, not a bug**: one channel's live partial text on `small-int8`
briefly fell into a repetition loop ("بانس کو پھر سے..." repeated ~15×) — a
known Whisper failure mode on ambiguous/noisy audio, and exactly the
scenario the README already flags: OpenVINO has no built-in VAD, unlike the
`faster-whisper` backend (`vad_filter=True`). Not something to patch here;
production channel monitoring on real broadcast audio is the reason #6
(faster-whisper vs OpenVINO comparison) and VAD are called out as real
follow-up work.

## Blocked — needs resources this session doesn't have

- **Phase 1 benchmarking**: deferred by explicit choice — no GPU available
  for now. No NVIDIA GPU in this machine (`nvidia-smi` absent; only an Intel
  UHD 620 iGPU with no compute runtime installed, no passwordless sudo to add
  one). The CUDA code path is implemented, defaults are CPU-first, and GPU/CUDA
  is a one-flag opt-in (`--device GPU`/`cuda`) — it will run once pointed at
  real hardware, but it has never actually executed here and there's nothing
  to benchmark against the OpenVINO numbers until that hardware exists.
- **Issue #3 (WER ground truth)**: `refs/` scaffold + workflow doc exist
  (`refs/README.md`), but hand-correcting a transcript requires a human
  listening to the source audio — that can't be fabricated. No real
  broadcast clip has been fetched/corrected yet.
- **Issue #6 (faster-whisper vs OpenVINO accuracy diff)**: both backends
  were smoke-tested (they run without error, including on real Urdu speech
  for the OpenVINO path — `"ایسا بہترین"` output confirmed earlier), but a
  meaningful accuracy *comparison* needs the same real clip run through both
  backends and scored against a hand-corrected reference, which needs #3 first.
- **Issue #7 (CISA GPU kernel warnings)**: specific to the original Iris Xe
  Windows machine; not reproducible here (different hardware, different OS,
  no GPU compute runtime even loaded). Nothing to diagnose in this environment.
- **Issue #14 / Phase 2 (int4/turbo benchmarking, LoRA fine-tuning)**: needs
  GPU compute hours plus the same real, hand-corrected dataset as #3 — doubly
  blocked given no GPU for now. Not attempted; would need dozens of downloaded
  model variants and real audio to produce a number worth trusting.
- **Phase 4 remainder (OpenSearch archival, editorial correction loop,
  shadow-mode live-broadcast testing, CG vendor integration)**: these are
  infrastructure/process items — a running search cluster, an editorial
  workflow, and access to an actual on-air feed — not something to stub out
  blind. Flagging rather than building untested integration code against
  systems that don't exist here.
- **Postgres persistence, end-to-end**: `docker compose up -d` needs the
  `docker` daemon, and this session's shell user is not in the `docker`
  group (`docker ps` → permission denied) with no passwordless `sudo` to fix
  it. The DB layer (`scripts/db.py`), schema, and graceful-degradation path
  are all written and the server correctly detects/reports DB-down — but
  actual rows landing in `segments` has not been observed. **To close this
  out**: run `docker compose up -d` yourself (or `sudo usermod -aG docker
  $USER` + new login so this session can), then `python webapp/server.py`
  and add a channel — `/health` should flip to `"database": "connected"`
  and rows should appear in `segments` within a few seconds of live speech.

## Config reference (new env vars, all optional)

`STT_BACKEND`, `STT_DEVICE`, `STT_MODEL`, `STT_WORKER_THREADS`,
`STT_INITIAL_PROMPT`, `STT_HOST`, `STT_PORT`, `STT_WS_TOKEN`,
`STT_ALLOWED_STREAM_HOSTS`, `STT_ALLOW_ANY_STREAM_HOST`,
`STT_MAX_SEG_S`/`STT_MIN_SEG_S`/`STT_SIL_WIN_S`/`STT_SIL_RMS`/`STT_SKIP_RMS`/
`STT_PARTIAL_MIN_S`/`STT_PARTIAL_GAP_S`, `STT_MAX_WS_BINARY_BYTES`,
`STT_STREAM_FIRST_DATA_TIMEOUT_S`, `STT_STREAM_RESOLVE_TIMEOUT_S`,
`STT_MAX_INITIAL_PROMPT_CHARS`, `STT_LOG_LEVEL`, `DATABASE_URL`. Full
descriptions are in the `webapp/server.py` module docstring.

## Known accepted limitations

- No global cap on concurrent `StreamPuller`s across connections/channels —
  each channel is independently bounded (one puller, one segmenter), but
  nothing stops an operator from adding more channels than
  `STT_WORKER_THREADS` + available CPU/bandwidth can actually keep up with;
  they'll just queue and fall behind realtime rather than fail. Acceptable
  for a single-operator dashboard; would need admission control before
  multi-tenant exposure.
- Live-stream URL resolution (`yt-dlp -g`) took anywhere from 8s to 30s+
  across repeated test runs against the same real YouTube channel — YouTube-
  side extraction latency is inherently variable, not something this
  project controls. `STT_STREAM_RESOLVE_TIMEOUT_S` (default 30s) is a
  judgment call; raise it if channels are timing out on a slow network.
