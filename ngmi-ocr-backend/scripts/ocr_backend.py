"""Urdu OCR via a small (2B) vision-language model fine-tuned specifically
on real Nastaliq broadcast fonts (Jameel Noori Nastaleeq, NotoNastaliqUrdu,
and others) -- oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct, a LoRA adapter
on top of Qwen/Qwen2-VL-2B-Instruct.

This is not a traditional detector+recognizer OCR pipeline. Nastaliq -- the
cursive, diagonally-stacked script Pakistani news tickers are set in -- is a
well-documented hard case for OCR: general toolkits (Tesseract, EasyOCR,
PaddleOCR) and even purpose-built Urdu OCR models (UTRNet) were evaluated
against real ticker crops before choosing this approach, and all produced
near-unreadable output (EasyOCR: ~0.02 confidence on a clean synthetic
render). A recent survey ("From Press to Pixels: Evolving Urdu Text
Recognition", 2025, arXiv:2505.13943) reaches the same conclusion at scale:
VLM-based recognition meaningfully outperforms traditional OCR specifically
on Nastaliq script. Qaari narrows that further to a small, purpose-tuned
model rather than a general-purpose multi-billion-parameter VLM.

Packaging quirk: the published adapter was trained against Unsloth's
repackaged Qwen2-VL-2B, whose module tree differs from the official
transformers implementation (no ".language_model." wrapper for the text
layers, one fewer ".model." level for the vision tower). Loading it with a
plain `PeftModel.from_pretrained()` against the official base model
silently no-ops -- every one of its 648 LoRA weights reports as "missing"
and generation quietly falls back to the un-adapted base model, which
hallucinates plausible-looking nonsense instead of transcribing. See
`_remap_adapter_state_dict()` below for the fix.

Hardware note: this model needs real inference time -- seconds on a modern
GPU, tens of seconds per image on CPU. On this project's original dev
hardware (Quadro M2000, 4GB VRAM, compute capability 5.2), neither 4-bit
quantization (bitsandbytes' kernels don't support pre-Turing/Maxwell GPUs)
nor plain fp16 (the model doesn't fit in 4GB, so Windows silently falls
back to slow memory-swapping) provided any speedup over CPU -- both
measured at the same ~25-30s per crop. `read_text()` is written to run on
whatever device it's given; if you have real GPU headroom this will be
correspondingly faster, but the webapp's default sampling interval is
tuned for the CPU-speed case.
"""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_MODEL_ID = "oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct"

PROMPT = "Read the Urdu text in this image and transcribe it exactly, character for character."

# A single ticker-line crop rarely holds more than ~15-20 Urdu words. A
# generous budget like 128 tokens, combined with a strong repetition
# guard, was observed to make the model ramble into unrelated hallucinated
# text (invented names/topics) to fill the space once the real transcript
# ran out -- capping it short is a big part of what keeps output honest.
MAX_NEW_TOKENS = 48

# Substring-matched against leaf module names throughout the whole model
# (vision tower + language model) -- see the docstring above for why the
# adapter's own target_modules regex from adapter_config.json isn't reused
# directly: it's written against Unsloth's module names, not these.
_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv", "proj", "fc1", "fc2",
]


class OcrBackendError(RuntimeError):
    """Raised when the OCR model or adapter can't be loaded correctly."""


def _remap_adapter_state_dict(raw: dict) -> dict:
    """Renames the adapter's stored tensor keys from Unsloth's module
    layout to the official transformers Qwen2-VL layout it's actually
    being loaded onto. A no-op for any key that doesn't need it."""
    remapped = {}
    for k, v in raw.items():
        if k.startswith("base_model.model.visual."):
            nk = k.replace("base_model.model.visual.", "base_model.model.model.visual.", 1)
        elif k.startswith("base_model.model.model.layers."):
            nk = k.replace(
                "base_model.model.model.layers.", "base_model.model.model.language_model.layers.", 1)
        else:
            nk = k
        remapped[nk] = v
    return remapped


class OcrBackend:
    """Loads the base model + Urdu-Nastaliq LoRA adapter once; answers
    read_text() calls against BGR numpy image crops (as produced by
    cv2/ffmpeg). generate() is not documented thread-safe -- like the
    sibling face-recognition project's FaceBackend, every read_text() call
    must run on the same single worker thread, never concurrently."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        t0 = time.time()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        logger.info("loading OCR base model %s (device=%s) ...", BASE_MODEL_ID, device)
        self.model_base = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, torch_dtype=dtype, device_map=device)
        self.processor = AutoProcessor.from_pretrained(ADAPTER_MODEL_ID)

        adapter_dir = Path(snapshot_download(ADAPTER_MODEL_ID))
        raw = load_file(str(adapter_dir / "adapter_model.safetensors"))
        remapped = _remap_adapter_state_dict(raw)

        lora_config = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
            target_modules=_LORA_TARGET_MODULES,
        )
        self.model = get_peft_model(self.model_base, lora_config)
        _missing, unexpected = set_peft_model_state_dict(self.model, remapped)
        if unexpected:
            raise OcrBackendError(
                f"{len(unexpected)} adapter weight(s) didn't map onto the base model -- "
                f"Urdu OCR would silently run on the un-adapted (hallucinating) base model")
        self.model.eval()
        logger.info("OCR backend ready in %.1fs", time.time() - t0)

    def read_text(self, frame_bgr: np.ndarray) -> str:
        """Runs the vision-language model once against a BGR image (a crop
        of a video frame, or a whole uploaded image) and returns its
        best-effort Urdu transcription. Slow -- always call via a
        dedicated worker thread, never on the async event loop."""
        img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Plain greedy decoding with no repetition guard at all
            # degenerates into looping the same phrase on ambiguous/
            # low-information crops (observed directly: a real ticker frame
            # produced the same six-word clause six times in a row). A mild
            # repetition_penalty discourages that without banning ordinary
            # Urdu word repetition outright -- no_repeat_ngram_size was
            # tried too but is too blunt: forbidding every repeated 3-gram
            # pushed the model to invent unrelated new content (hallucinated
            # names/topics) instead, which is worse than the loop it fixed.
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.15,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def load_image_bgr(path) -> Optional[np.ndarray]:
    """Mirrors the sibling face-recognition project's helper: returns None
    (not an exception) on an unreadable file."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("could not read image: %s", path)
    return img
