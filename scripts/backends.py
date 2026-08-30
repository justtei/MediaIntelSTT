"""Pluggable STT backends shared by the CLI (transcribe.py) and the live web
app (webapp/server.py), so both get the same tested device-fallback behavior
instead of duplicating (and drifting on) load/fallback logic.

  OpenVINOBackend       - Iris Xe/Intel iGPU or CPU, via openvino_genai
  FasterWhisperBackend  - CPU or NVIDIA CUDA, via CTranslate2

Both implement transcribe(samples, language) -> (text, [Segment]).
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


class BackendLoadError(RuntimeError):
    """Raised when a backend can't be loaded on any candidate device."""


class Backend:
    name: str
    device: str

    def transcribe(self, samples, language: str, initial_prompt: str | None = None):
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
                self.pipe = openvino_genai.WhisperPipeline(str(model_dir), device=candidate)
                self.device = candidate
                break
            except Exception as e:
                logger.warning("OpenVINO load on %s failed: %s", candidate, e)
        if self.pipe is None:
            raise BackendLoadError(f"OpenVINO pipeline failed to load on any of {tried}")

    def transcribe(self, samples, language: str, initial_prompt: str | None = None):
        config = self.pipe.get_generation_config()
        config.task = "transcribe"
        config.return_timestamps = True
        if language != "auto":
            config.language = f"<|{language}|>"
        if initial_prompt:
            config.initial_prompt = initial_prompt
        result = self.pipe.generate(samples, config)
        chunks = [Segment(c.start_ts, c.end_ts, c.text.strip()) for c in (result.chunks or [])]
        return str(result).strip(), chunks


class FasterWhisperBackend(Backend):
    name = "faster-whisper"

    def __init__(self, model: str = "large-v3", device: str = "cpu", compute_type: str | None = None):
        from faster_whisper import WhisperModel

        self.model = None
        tried = []
        for candidate in _candidates(device, cpu_name="cpu", primary_name="cuda"):
            tried.append(candidate)
            compute = compute_type or ("float16" if candidate == "cuda" else "int8")
            try:
                logger.info("loading faster-whisper %s on %s (%s) ...", model, candidate, compute)
                self.model = WhisperModel(model, device=candidate, compute_type=compute)
                self.device = candidate
                break
            except Exception as e:
                logger.warning("faster-whisper load on %s failed: %s", candidate, e)
        if self.model is None:
            raise BackendLoadError(f"faster-whisper failed to load on any of {tried}")

    def transcribe(self, samples, language: str, initial_prompt: str | None = None):
        segments, info = self.model.transcribe(
            samples,
            language=None if language == "auto" else language,
            vad_filter=True,
            beam_size=5,
            initial_prompt=initial_prompt,
        )
        chunks, parts = [], []
        for s in segments:  # generator — consumes here, so callers can time this call
            chunks.append(Segment(s.start, s.end, s.text.strip()))
            parts.append(s.text)
        return " ".join(parts).strip(), chunks


def load_backend(name: str, **kwargs) -> Backend:
    if name == "openvino":
        return OpenVINOBackend(**kwargs)
    if name == "faster-whisper":
        return FasterWhisperBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r} (expected 'openvino' or 'faster-whisper')")
