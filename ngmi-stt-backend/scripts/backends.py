"""Pluggable STT backends shared by the CLI (transcribe.py) and the live web
app (webapp/server.py), so both get the same tested device-fallback behavior
instead of duplicating (and drifting on) load/fallback logic.

  OpenVINOBackend       - Iris Xe/Intel iGPU or CPU, via openvino_genai
  FasterWhisperBackend  - CPU or NVIDIA CUDA, via CTranslate2

Both implement transcribe(samples, language) -> (text, [Segment]).
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


class BackendLoadError(RuntimeError):
    """Raised when a backend can't be loaded on any candidate device."""


def _collapse_repetition(text: str, max_repeats: int = 2) -> str:
    """Whisper occasionally spirals into repeating the same short phrase many
    times on ambiguous/noisy audio (a well-documented failure mode, worse
    without VAD to filter that audio out first). Generation-time controls
    meant to prevent this aren't reliable here — openvino_genai's
    no_repeat_ngram_size is a silent no-op under greedy decoding (verified:
    identical output with/without it), and beam search would fix it but
    roughly doubles per-segment latency, which live captioning can't afford.
    So this collapses any run of an immediately-repeated 1-4 word phrase down
    to at most `max_repeats` occurrences after the fact, engine-agnostic."""
    words = text.split()
    n = len(words)
    if n < 3:
        return text
    out = []
    i = 0
    while i < n:
        matched = False
        for plen in (1, 2, 3, 4):
            if i + plen * 2 > n:
                continue
            phrase = words[i:i + plen]
            reps = 1
            j = i + plen
            while j + plen <= n and words[j:j + plen] == phrase:
                reps += 1
                j += plen
            if reps > max_repeats:
                out.extend(phrase * max_repeats)
                i = j
                matched = True
                break
        if not matched:
            out.append(words[i])
            i += 1
    return " ".join(out)


class Backend:
    name: str
    device: str

    def transcribe(
        self,
        samples,
        language: str,
        initial_prompt: str | None = None,
        is_partial: bool = False,
        return_timestamps: bool = False,
    ):
        raise NotImplementedError


def _candidates(device: str, cpu_name: str, primary_name: str):
    """Yield devices to try in order. 'auto' tries the accelerator then CPU;
    an explicit device is tried alone, then CPU as a last-resort fallback
    (unless CPU was what was asked for)."""
    device = (device or "auto")
    if device.lower() == "auto":
        yield primary_name
        yield cpu_name
        return
    yield device
    if device != cpu_name:
        yield cpu_name


class OpenVINOBackend(Backend):
    name = "openvino"

    def __init__(self, model_dir, device: str = "GPU"):
        import openvino_genai

        self.pipe = None
        tried = []
        for candidate in _candidates(device, cpu_name="CPU", primary_name="GPU"):
            tried.append(candidate)
            try:
                logger.info("loading %s on %s ...", model_dir, candidate)
                if candidate in ("GPU", "GPU.0", "GPU.1"):
                    self.pipe = openvino_genai.WhisperPipeline(
                        str(model_dir), device=candidate, PERFORMANCE_HINT="LATENCY"
                    )
                else:
                    self.pipe = openvino_genai.WhisperPipeline(str(model_dir), device=candidate)
                self.device = candidate
                break
            except Exception as e:
                logger.warning("OpenVINO load on %s failed: %s", candidate, e)
        if self.pipe is None:
            raise BackendLoadError(f"OpenVINO pipeline failed to load on any of {tried}")

    def transcribe(
        self,
        samples,
        language: str,
        initial_prompt: str | None = None,
        is_partial: bool = False,
        return_timestamps: bool = False,
    ):
        config = self.pipe.get_generation_config()
        config.task = "transcribe"
        config.return_timestamps = return_timestamps
        # Discourage (not block) reusing already-generated tokens, to reduce how
        # often a repetition spiral starts in the first place. 1.3 measurably
        # degraded normal (non-looping) segments in an A/B test; 1.1 was
        # indistinguishable from no penalty at all on 3/4 test segments while
        # still providing real anti-repetition pressure — _collapse_repetition()
        # below is the actual backstop against severe loops, not this value.
        config.repetition_penalty = 1.1
        if is_partial:
            config.max_new_tokens = 24
        else:
            # More tokens for longer segments so a longer MAX_SEG_S (more
            # acoustic context -> measurably better accuracy in testing) doesn't
            # get truncated by a fixed cap.
            dur_s = len(samples) / 16000 if hasattr(samples, "__len__") else 3.0
            config.max_new_tokens = min(120, max(24, int(dur_s * 20)))
        if language != "auto":
            config.language = f"<|{language}|>"
        if initial_prompt:
            config.initial_prompt = initial_prompt

        result = self.pipe.generate(samples, config)
        chunks = [Segment(c.start_ts, c.end_ts, _collapse_repetition(c.text.strip())) for c in (result.chunks or [])]
        return _collapse_repetition(str(result).strip()), chunks


class FasterWhisperBackend(Backend):
    name = "faster-whisper"

    def __init__(self, model: str = "large-v3", device: str = "cpu", compute_type: str | None = None):
        import ctranslate2
        from faster_whisper import WhisperModel

        self.model = None
        tried = []
        for candidate in _candidates(device, cpu_name="cpu", primary_name="cuda"):
            tried.append(candidate)
            compute = compute_type or ("float16" if candidate == "cuda" else "int8")
            supported = ctranslate2.get_supported_compute_types(candidate)
            if compute not in supported:
                fallback = "float32" if "float32" in supported else next(iter(supported), compute)
                logger.warning("faster-whisper: compute type %r not supported on %s (supports %s) — "
                                "falling back to %r", compute, candidate, sorted(supported), fallback)
                compute = fallback
            try:
                logger.info("loading faster-whisper %s on %s (%s) ...", model, candidate, compute)
                candidate_model = WhisperModel(model, device=candidate, compute_type=compute)
                # ctranslate2 validates device/compute-type support lazily: construction can
                # succeed even when the device's compiled kernels don't actually cover this GPU
                # (e.g. an older CUDA compute capability no longer shipped by newer ctranslate2
                # builds), and the real failure only surfaces on the first inference call. Smoke
                # test with silence here so that failure is caught in this fallback loop instead
                # of on live audio.
                list(candidate_model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])
                self.model = candidate_model
                self.device = candidate
                break
            except Exception as e:
                logger.warning("faster-whisper load on %s (%s) failed: %s", candidate, compute, e)
        if self.model is None:
            raise BackendLoadError(f"faster-whisper failed to load on any of {tried}")

    def transcribe(
        self,
        samples,
        language: str,
        initial_prompt: str | None = None,
        is_partial: bool = False,
        return_timestamps: bool = False,
    ):
        segments, info = self.model.transcribe(
            samples,
            language=None if language == "auto" else language,
            vad_filter=False,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            repetition_penalty=1.3,
            initial_prompt=initial_prompt,
        )
        chunks, parts = [], []
        for s in segments:  # generator — consumes here, so callers can time this call
            text = _collapse_repetition(s.text.strip())
            chunks.append(Segment(s.start, s.end, text))
            parts.append(text)
        return _collapse_repetition(" ".join(parts).strip()), chunks


def load_backend(name: str, **kwargs) -> Backend:
    if name == "openvino":
        return OpenVINOBackend(**kwargs)
    if name == "faster-whisper":
        return FasterWhisperBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r} (expected 'openvino' or 'faster-whisper')")
