# NGMI Urdu OCR

Standalone Urdu news-ticker / sticker text extraction for live video — no
face recognition, no speech-to-text, just: read the Urdu text scrolling
along the bottom of a live YouTube channel, any YouTube video link, or an
uploaded video file, and any secondary "sticker" text along the side.
Sibling of a face-recognition project built the same way.

## Approach

Not a traditional detector+recognizer OCR pipeline. Nastaliq — the
cursive, diagonally-stacked script Pakistani news tickers are set in — is a
well-documented hard case for OCR. Before settling on an approach, three
options were evaluated against real ticker crops:

- **EasyOCR** (`ur` language pack) — unusable. ~0.02 confidence and
  unreadable output even on a clean, synthetically-rendered sample.
- **PaddleOCR** (`ar`, Arabic-script model — no dedicated Urdu model
  exists) — captures overall word shape and rhythm, but substitutes the
  seven Urdu-only letters (ھ ٹ ڈ ڑ ے ں گ) with similar-looking Arabic
  ones, since it was never trained on them.
- **Qaari** (`oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct`, a LoRA
  fine-tune of `Qwen/Qwen2-VL-2B-Instruct` trained specifically on real
  Nastaliq broadcast fonts — Jameel Noori Nastaleeq, NotoNastaliqUrdu, and
  others) — the clear winner. Real, correctly-spelled Urdu-specific words
  and phrases came through intact where the other two failed.

This matches a recent survey's conclusion at scale
("[From Press to Pixels: Evolving Urdu Text Recognition](https://arxiv.org/abs/2505.13943)",
2025): traditional OCR toolkits (including UTRNet, a purpose-built Urdu
model) underperform badly on real Nastaliq text, and VLM-based recognition
meaningfully outperforms them.

**Packaging quirk worth knowing about:** the published Qaari adapter was
trained against Unsloth's repackaged Qwen2-VL-2B, whose module tree
differs from the official `transformers` implementation this project
loads. Loading it the naive way (`PeftModel.from_pretrained()` against the
official base model) silently no-ops — every one of its 648 LoRA weights
reports as "missing" and generation quietly falls back to the un-adapted
base model, which hallucinates plausible-looking nonsense instead of
transcribing. `scripts/ocr_backend.py`'s `_remap_adapter_state_dict()`
fixes this by renaming the stored tensor keys to the layout the official
model actually expects. `OcrBackend.__init__` hard-fails
(`OcrBackendError`) if any adapter weight still fails to map, specifically
so this can't silently regress back to the hallucinating-base-model
failure mode.

## Hardware reality check

This model needs real inference time — seconds per image on a modern GPU,
tens of seconds per crop on CPU. On this project's original dev machine
(Quadro M2000, 4GB VRAM, compute capability 5.2 / Maxwell), neither
4-bit quantization (bitsandbytes' kernels don't support pre-Turing GPUs —
confirmed via `python -m bitsandbytes`, which fails outright on this card)
nor plain fp16 (the model doesn't fit in 4GB, so Windows falls back to
slow memory-swapping) gave any speedup over CPU — both measured at ~25-30s
per crop. `OCR_DEVICE` defaults to `cpu`; set it to `cuda:0` if you have a
real GPU with enough VRAM (8GB+, Turing or newer strongly recommended for
`bitsandbytes` 4-bit quantization to actually help).

Given that, treat this app as a **periodic ticker-headline digest, not a
real-time feed**: `OCR_SAMPLE_INTERVAL_S` defaults to 45s per channel, and
every channel's OCR calls share one single-worker queue (the model isn't
documented safe for concurrent `generate()` calls), so running several
channels at once multiplies latency rather than parallelizing it.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

ffmpeg (and, for reliable YouTube extraction, `deno`) are expected under
`tools/` or on `PATH` — see `scripts/ffmpeg_util.py`.

## Run

```powershell
.venv\Scripts\python webapp\server.py
```

Then open http://127.0.0.1:8020. Add a live YouTube channel or any
YouTube video link, or upload a local video file — the dashboard shows
each channel's video with its configured ticker/sticker crop regions
outlined, and the latest Urdu reading for each below it. The first request
downloads the base model + adapter (~4.5GB total) from Hugging Face.

Key env vars (all optional): `OCR_HOST`/`OCR_PORT` (default
127.0.0.1:8020), `OCR_SAMPLE_INTERVAL_S` (default 45), `OCR_DEVICE`
(default `cpu`), `OCR_REGIONS` (default `bottom,side`),
`OCR_BOTTOM_Y0`/`OCR_BOTTOM_Y1`/`OCR_SIDE_X0`/`OCR_SIDE_X1` (crop
fractions — defaults assume a bottom ticker in the lowest ~18% of frame
and a side sticker in the rightmost ~22%; adjust per channel layout),
`OCR_MAX_UPLOAD_BYTES`, `OCR_ALLOWED_STREAM_HOSTS` /
`OCR_ALLOW_ANY_STREAM_HOST`, `OCR_STREAM_STALL_TIMEOUT_S` /
`OCR_STREAM_MAX_RESTARTS` — see the module docstring in `webapp/server.py`
for the full list.

## Layout

```
scripts/
  ocr_backend.py             OCR model loading (base + LoRA adapter, with the
                              Unsloth-key-remapping fix) + read_text()
  ffmpeg_util.py              locates ffmpeg (PATH or tools/)
  url_safety.py                SSRF-safe allow-list for channel URLs
webapp/
  server.py                    FastAPI + WebSocket server, FramePuller,
                                region cropping
  static/index.html            dashboard UI
```

## Known limitations

- Nastaliq OCR is still an open problem even with the best approach found
  here — expect a meaningful error rate, especially on small, blurry, or
  heavily-compressed ticker crops. Treat readings as assistive, not
  authoritative.
- Slow by nature, not by bug — see "Hardware reality check" above. Don't
  expect sub-minute freshness with more than one or two channels running.
- Default crop regions (bottom 18%, right 22%) are a reasonable starting
  guess, not calibrated to any specific channel's actual layout — adjust
  the `OCR_BOTTOM_*` / `OCR_SIDE_*` env vars to match what you're actually
  watching.
- No authentication exists yet — do not bind `OCR_HOST` beyond localhost
  on an untrusted network (mirrors the same caveat in the sibling
  face-recognition project).
