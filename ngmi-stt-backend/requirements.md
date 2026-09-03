# requirements.md
## STT Model Framework → Production Urdu Broadcast STT
### Grounded in the existing `STTModelFramework` codebase (audit dated 2026-08-25)

**Scope:** This supersedes prior generic requirements documents. It is grounded in the **actual existing project** (`c:\Amjad Ali\STTModelFramework`) — its real architecture, its real dependencies, and the 11 concrete issues found in the audit — and defines what's needed to take it from an evaluation toolkit to a production-ready Urdu broadcast STT system.

---

## 1. Current State (as audited)

| Aspect | Current reality |
|---|---|
| Purpose | Windows-only evaluation toolkit: Whisper on Intel Iris Xe iGPU via OpenVINO, Urdu-focused, English secondary |
| Components | CLI pipeline (`scripts/`) + live web app (`webapp/`, FastAPI + WebSocket + vanilla JS) |
| Inference engines | `openvino_genai.WhisperPipeline` (GPU, primary), `faster-whisper` (CPU, baseline comparison) |
| Streaming approach | Custom `Segmenter` class — silence-cut buffering with a hard 8s cap, RMS-based silence detection, throttled partial decodes |
| Explicitly scoped as | **Evaluation only** — README states large-v3-int8 on Iris Xe runs "well below realtime — fine for evaluation, not live use" |
| Source size | ~700 lines Python + 1 HTML file, 12 tracked files |
| Known-good | Model download, GPU detection/compile, CLI transcription (Urdu + English), web app server startup — all verified working at least once |
| Known-broken/missing | No dependency manifest, WER evaluation never actually run, script audit false-positives on English, uncommitted encoding fix, no tests/CI |

**This document's job**: close the gap between "evaluation toolkit that works on one specific Windows machine with an Iris Xe iGPU" and "production system that can run reliably, be reproduced by anyone, be measured accurately, and eventually handle live broadcast throughput."

---

## 2. Critical Architectural Decision: Hardware Target for Production

### 2.1 The core issue
The existing project's own README says its primary path (Iris Xe + OpenVINO) is **not viable for live use** — it runs below realtime. This is a hard constraint, not a tuning problem: an integrated GPU with 15.7GB *shared* system RAM is fundamentally under-provisioned for real-time large-v3 inference, no matter how well the software around it is fixed.

### 2.2 Decision
- **Keep OpenVINO/Iris Xe as a supported evaluation/dev target** — it's genuinely useful for local development on a laptop without a discrete GPU, script auditing, and WER benchmarking work. Don't rip it out.
- **Add a CUDA + `faster-whisper` production path** as the primary target for anything live. `faster-whisper` is already a first-class backend in `transcribe.py` (`--backend faster-whisper`) — today it only runs on CPU in this project; the production requirement is to run it on an NVIDIA GPU (8–16GB class, per earlier hardware constraints established for this project) via CTranslate2's CUDA support.
- **`device`/`backend` becomes an explicit, tested configuration axis**, not an assumption baked into one machine. Both `transcribe.py` and `webapp/server.py` need a real fallback path (see Issue Fix #9 below) rather than a hardcoded `device="GPU"` that crashes if Iris Xe isn't present.

| Target | Engine | Use |
|---|---|---|
| Dev/eval (current machine) | OpenVINO + Iris Xe | Local development, script audit tooling, offline WER benchmarking |
| Production (live) | `faster-whisper` (CTranslate2) + NVIDIA CUDA GPU | Live streaming captioning, throughput-sensitive workloads |
| Production (archival/batch) | `faster-whisper` + CUDA, or OpenVINO on a server-grade Intel GPU if that's the deployment target | Offline transcription of archived packages |

---

## 3. Issue Backlog (from audit §7, prioritized for production readiness)

| # | Issue | Priority | Fix required |
|---|---|---|---|
| 1 | No dependency manifest | **P0 — blocking** | Freeze `requirements.txt` (or migrate to `pyproject.toml`) from current `.venv`'s `pip freeze`; pin exact versions for `openvino`, `openvino-genai`, `faster-whisper`, `ctranslate2`, `fastapi`, `jiwer`, `yt-dlp`. No production deployment proceeds without this. |
| 2 | Uncommitted UTF-8 stdout fix | **P0 — blocking** | Commit the `sys.stdout.reconfigure(encoding="utf-8")` change across the 5 modified files immediately. Verify `urdu_norm.py`, `fetch_audio.py`, `download_models.py` genuinely don't need it (confirm no Urdu text ever hits their stdout). |
| 3 | WER evaluation never actually run | **P0 — blocking** | Create real hand-corrected reference transcripts under `refs/`; run `wer_eval.py` end-to-end; this is the actual accuracy evidence needed before trusting any model recommendation. Add `refs/` to version control (it should NOT be gitignored — it's ground truth data, not a build artifact). |
| 4 | `script_audit.py` false-positives on English | **P1 — high** | Add a `--language {ur,en,auto}` flag; branch the heuristic so English output is validated against a Latin-script expectation instead of a Perso-Arabic one. Without this, the English demo path has zero automated quality validation. |
| 5 | Hardcoded `device="GPU"`, no fallback | **P1 — high** | Wrap model load in try/except; on failure (or explicit `--device auto`), fall back to `faster-whisper` CPU/CUDA path with a clear log message instead of an unguarded crash. Mirror `smoke_test.py`'s existing device-enumeration check into `server.py` and `transcribe.py`. |
| 6 | `faster-whisper` backend untested in this project | **P1 — high** | Run it against the same clips used for the OpenVINO path; diff the two outputs; this comparison is explicitly called for in the README and hasn't happened. Required before it can be trusted as the production engine (Section 2.2). |
| 7 | CISA GPU kernel-compile warnings undiagnosed | **P1 — high** | Investigate before trusting any Iris Xe output uncritically: check Intel graphics driver version, cross-reference OpenVINO GPU plugin issue tracker for this exact error signature. Since Iris Xe is now scoped as dev/eval only (Section 2.2) rather than production, this is downgraded from blocking to high — but still needs resolution so eval numbers are trustworthy. |
| 8 | No automated tests, no CI | **P1 — high** | Minimum viable: pytest coverage for `urdu_norm.py` (pure function, easy to test), `script_audit.py` heuristics (once #4 is fixed), and `wer_eval.py` scoring logic. A GitHub Actions (or equivalent) workflow running these on every push. Full integration testing of the GPU pipeline is harder to automate (hardware-dependent) — document that gap explicitly rather than skipping it silently. |
| 9 | `@app.on_event("startup")` deprecated | P2 — medium | Migrate to FastAPI `lifespan` context manager before it's removed upstream. |
| 10 | `/demo-audio/{lang}` returns 200 on error | P2 — medium | Return proper 4xx status codes on missing lang/file, so clients checking `response.ok` behave correctly. |
| 11 | `StreamPuller` has no URL allow-listing | P2 — medium, **P0 if bind host changes** | Fine today bound to `127.0.0.1`. **Before any change to `0.0.0.0` or LAN/public exposure, this becomes blocking** — add domain allow-listing or an explicit trusted-source list before opening the server beyond localhost. |
| 12 | Stale `E:\AvaPro\...` paths in logs/`.claude` settings | P3 — low | Cosmetic. Clean up old log files; regenerate `.claude/settings.local.json` history naturally over time. |
| 13 | Untracked `kernel.errors.txt` not gitignored | P3 — low | Add `*.errors.txt` to `.gitignore`, or delete once Issue #7 is resolved. |
| 14 | `large-v3-int4` / `large-v3-turbo-int8` documented but never benchmarked | P2 — medium | Run the README's own recommended comparison — especially the turbo variant's low-resource-language caveat needs verifying against real Urdu clips before ever considering it for production, given the known decoder-distillation accuracy risk on languages like Urdu. |

---

## 4. Functional Requirements (production scope)

- **FR1** — Reproducible environment: fresh clone → documented setup → working pipeline, no tribal knowledge required
- **FR2** — Accurate, evidenced WER reporting per audio condition (studio/field/panel/code-switched), not just "it ran"
- **FR3** — Live streaming transcription meeting a defined latency target on production hardware (Section 2), with the existing `Segmenter` logic as the starting point, tuned/validated against that hardware
- **FR4** — Correct language-aware script validation for both Urdu and English output
- **FR5** — Graceful degradation: device unavailable → documented fallback, not a crash
- **FR6** — Multi-source ingestion preserved and hardened: mic, demo clips, remote stream (`yt-dlp`) — with the URL allow-listing gap (#11) closed before any non-localhost exposure
- **FR7** — Custom Urdu vocabulary/prompting support (politician names, places, orgs) — not yet present in the current codebase, needs adding via Whisper's `initial_prompt` mechanism at minimum

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Reproducibility | `requirements.txt`/`pyproject.toml` present and accurate; documented setup steps beyond the current README's model-download instructions |
| Reliability | No unguarded crashes on missing hardware/device; health-check endpoint on `webapp/server.py` |
| Security | No unrestricted server-side URL fetching once exposed beyond localhost (#11); WebSocket auth before any non-local deployment |
| Testability | Automated tests for pure-logic components at minimum (normalization, scoring, audit heuristics) |
| Observability | Existing per-segment `infer_s`/`rtf` reporting in the WebSocket protocol is a good foundation — extend it with structured logging to a file/log aggregator, not just console/browser display |
| Latency (production, CUDA target) | Sub-2s end-to-end, consistent with the earlier project-wide latency revision for Whisper-based streaming architectures |

---

## 6. Data & Evaluation Requirements

### 6.1 Immediate priority: close the WER-never-run gap (Issue #3)
1. Take the two existing transcripts (`transcripts/news_test.openvino.txt`, `transcripts/english_test.openvino.txt`) or fresh clips from `fetch_audio.py`
2. Hand-correct them into `refs/<clip>.txt` — this is the actual ground-truth work the framework has been missing since inception
3. Run `wer_eval.py` for real; record the resulting WER/CER as the first real accuracy baseline for this project
4. Repeat across a broader clip set covering studio/field/panel/code-switched conditions (per the broader project data-condition matrix established earlier) — the existing `fetch_audio.py` (real broadcast audio via `yt-dlp`, not canned benchmark corpora) is well-suited for this; it just hasn't been paired with reference transcripts yet

### 6.2 Broader data sourcing (unchanged from earlier analysis)
No public "Pakistani news channel" dataset exists — this project's own approach (pulling real broadcast clips via `fetch_audio.py` + hand correction) is actually the right instinct and should be scaled up, not replaced. Public corpora (Common Voice Urdu, FLEURS, CSaLT) remain useful only as supplementary bootstrap data, not primary training/eval material.

### 6.3 Fine-tuning path
Once `refs/`-backed WER numbers exist and show the stock models need improvement:
- Apply the LoRA/QLoRA fine-tuning approach established earlier (Whisper-medium primary, large-v3 QLoRA as a gated stretch option), sized to an 8–16GB CUDA GPU
- Fine-tuned checkpoints then need to be validated on **both** inference backends — `faster-whisper`/CTranslate2 for the CUDA production path, and converted to OpenVINO IR format for continued eval-machine compatibility, since `download_models.py` currently only pulls pre-converted `OpenVINO/whisper-*-ov` repos, not custom fine-tuned weights

---

## 7. Repository & Engineering Hygiene Requirements

- `requirements.txt` (or `pyproject.toml`) committed and kept current — no more relying on `.claude/settings.local.json` command history as the de facto install log
- `refs/` added as a real, tracked data directory (not gitignored)
- `.gitignore` updated to cover `*.errors.txt`
- CI workflow (GitHub Actions or equivalent): lint + pytest on every push, at minimum for `urdu_norm.py`, `script_audit.py`, `wer_eval.py`
- Encoding fix (Issue #2) committed immediately — currently a correct fix sitting uncommitted, protecting nobody but the current local working tree
- Stale-path cleanup (Issue #12) — low priority, but worth a single pass once other P0/P1 items are done

---

## 8. Roadmap

### Phase 0 — Stabilize the existing codebase (P0/P1 issue closure)
- Commit the encoding fix (#2)
- Freeze `requirements.txt` (#1)
- Create `refs/` ground truth and run real WER for the first time (#3)
- Fix `script_audit.py` language-awareness (#4)
- Add device fallback logic (#5)
- Run and diff the `faster-whisper` CPU baseline (#6)
- Investigate CISA kernel warnings (#7)
- Stand up minimal pytest + CI (#8)

**This phase alone turns the project from "runs on one machine, mostly untested" into a reproducible, evidenced baseline — do this before any new features.**

### Phase 1 — Production hardware path
- Stand up `faster-whisper` on a CUDA-capable 8–16GB GPU (local or rented cloud instance)
- Benchmark against the existing Iris Xe/OpenVINO numbers on the same clips — establish which is faster/more accurate on real Urdu broadcast audio
- Confirm `Segmenter`'s buffering parameters (`MAX_SEG_S`, `SIL_WIN_S`, `SIL_RMS`, `PARTIAL_MIN_S`, `PARTIAL_GAP_S`) still make sense at CUDA-class inference speed — they were tuned against Iris Xe's slower realtime factor and may need retuning

### Phase 2 — Data & fine-tuning
- Scale up `fetch_audio.py`-sourced clips across studio/field/panel/code-switched conditions
- Build the `refs/` reference set out into real train/val/test splits
- Fine-tune Whisper-medium (LoRA) per Section 6.3, evaluate against the now-real WER baseline

### Phase 3 — Production hardening
- WebSocket auth, URL allow-listing for `StreamPuller` (#11) — required before any non-localhost exposure
- FastAPI `lifespan` migration (#9), proper error status codes (#10)
- Structured logging, health-check endpoint, load testing for concurrent stream handling

### Phase 4 — Broadcast integration
- CG system output integration (WebVTT/SCC), archival search (OpenSearch), editorial correction loop
- Shadow-mode testing against live broadcast feeds before any on-air exposure

---

## 9. Risk Register

| Risk | Mitigation |
|---|---|
| Iris Xe/OpenVINO path mistaken for production-viable | Explicitly documented as dev/eval only (Section 2); production requires the CUDA path |
| CISA kernel warnings mask silent accuracy loss | Investigate before trusting any Iris Xe-derived WER number as authoritative (#7) |
| No ground-truth data means no real accuracy evidence | Phase 0's `refs/` creation is the highest-leverage, most overdue fix in the whole project |
| `StreamPuller` exposed beyond localhost without allow-listing | Hard gate: no bind-host change to `0.0.0.0`/LAN until #11 is closed |
| `Segmenter` tuning assumes Iris Xe's slow inference speed | Re-validate/retune buffering parameters once running on CUDA (Phase 1) — faster inference changes the latency/accuracy trade-off the current constants were tuned for |
| Fine-tuned checkpoints incompatible with OpenVINO eval path | Budget explicit conversion work (PyTorch/CTranslate2 → OpenVINO IR) if continued dual-backend support matters |
| No CI means regressions go unnoticed | Phase 0 CI setup, even minimal, catches this early |

---

## 10. Definition of Done

**Phase 0 (stabilization) is done when:**
- [ ] `requirements.txt` exists, committed, verified against a clean `.venv` rebuild
- [ ] Encoding fix committed
- [ ] At least one `refs/<clip>.txt` exists and `wer_eval.py` has produced a real, recorded WER number
- [ ] `script_audit.py` correctly validates both Urdu and English output
- [ ] `faster-whisper` backend has been run and diffed against OpenVINO output on the same clip
- [ ] Device-load failure produces a clear fallback/error, not an unguarded crash
- [ ] CISA kernel warnings have a documented root cause (or documented as "known benign, tracked upstream")
- [ ] pytest + CI running on every push, covering at minimum `urdu_norm.py` and `wer_eval.py`

**Production-ready (full) is done when**, in addition to the above:
- [ ] Live streaming meets its latency target on the CUDA production path, measured, not assumed
- [ ] WER meets acceptance thresholds across studio/field/panel/code-switched conditions with real evidence
- [ ] No unrestricted server-side URL fetching in any non-localhost deployment
- [ ] Structured logging + health checks in place
- [ ] Fine-tuned model (if used) validated on the actual production inference backend