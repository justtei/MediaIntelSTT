# STT Model Framework — Whisper on Intel Iris Xe (Windows)

Runs Whisper **on the iGPU** via OpenVINO (int8), with a faster-whisper CPU
baseline, real-broadcast test-audio fetching, an Urdu script audit, and
normalized WER scoring.

All commands below use the project venv:

```powershell
.\.venv\Scripts\Activate.ps1     # or prefix commands with .\.venv\Scripts\python.exe
```

## 1. One-time setup

```powershell
python scripts\download_models.py                # large-v3 int8 (~1.6 GB) — Urdu quality pick
python scripts\download_models.py small-int8     # optional: fast model for quick iterations
python scripts\smoke_test.py                     # proves the Iris Xe path works end-to-end
```

The first GPU compile of a model takes a few minutes; OpenVINO caches it, so
subsequent loads are fast.

## 2. Get real test audio (not Common Voice / FLEURS)

Pick three ~10-minute clips deliberately: one anchor monologue, one
multi-speaker panel with crosstalk, one with background music or a
phone-quality remote guest.

```powershell
python scripts\fetch_audio.py "<news_clip_url>" --name anchor_mono --section "00:10:00-00:20:00"
```

Output lands in `audio\<name>.wav` as 16 kHz mono (what Whisper expects).

## 3. Transcribe

```powershell
# iGPU (OpenVINO), Urdu forced:
python scripts\transcribe.py audio\anchor_mono.wav

# language ID test:
python scripts\transcribe.py audio\anchor_mono.wav --language auto

# CPU baseline (faster-whisper int8 + VAD filter):
python scripts\transcribe.py audio\anchor_mono.wav --backend faster-whisper
```

Prints timestamped segments, elapsed time, and realtime factor; writes the
plain transcript to `transcripts\<clip>.<backend>.txt`.

## 4. Audit the script before trusting numbers

```powershell
python scripts\script_audit.py transcripts\anchor_mono.openvino.txt
```

Flags romanized (Latin) output and Arabic-drift codepoints (`ي ك ة` instead of
`ی ک ہ`) — both silently corrupt WER if unchecked.

## 5. Normalized WER

Hand-correct ~5 minutes of reference into `refs\<clip>.txt`, then:

```powershell
python scripts\wer_eval.py refs\anchor_mono.txt transcripts\anchor_mono.openvino.txt --raw
```

Normalization (shared with the audit, `scripts\urdu_norm.py`): NFC, Arabic→Urdu
codepoint folding, diacritic stripping, punctuation removal. Raw WER on Urdu is
close to meaningless without it.

## Model notes for this hardware (Iris Xe, 15.7 GB shared RAM)

| Model | Why / why not |
|---|---|
| `large-v3-int8` (default) | Best Urdu quality that fits comfortably; expect well below realtime on Iris Xe — fine for evaluation, not live use |
| `small-int8` | Iteration speed while wiring things up |
| `large-v3-int4` | Try if int8 feels too slow; verify quality on your own clips |
| `large-v3-turbo-int8` | Fast, but decoder distillation disproportionately hurts low-resource languages — benchmark against large-v3 on your Urdu clips before adopting |

- OpenVINO backend has no built-in VAD; for broadcast audio with long music
  beds, compare against the faster-whisper backend (`vad_filter=True`) to see
  how much hallucination VAD is saving you.
- ffmpeg is vendored in `tools\` — nothing else needs to be installed
  system-wide.
