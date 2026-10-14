"""Base class for local 4-bit VLM inference.

All local models share:
- 4-bit NF4 quantization via bitsandbytes (mandatory on 8 GB VRAM RTX 5070)
- transformers >= 4.57 processor/model auto-classes
- deterministic generation (do_sample=False, temperature=None) so C2/C3 are
  comparable across runs
"""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class LocalResponse:
    raw: str
    parsed_yes: bool | None
    model: str
    provider: str
    latency_s: float
    error: str | None = None


class LocalVLM(ABC):
    """Each open-weight VLM subclasses this. Implements lazy model loading + cleanup."""

    provider: str = "local"
    model_id: str = ""

    def __init__(self, device: str | None = None, dtype: str = "4bit"):
        # Default stays 4-bit (the legacy hardcoded behavior) so existing
        # constructors on VRAM-constrained local GPUs don't silently flip to
        # a ~9GB bf16 load (cross-model review). A100 runners that need the
        # bf16 regime (wheel_runner, kv_truncation_runner) pass it EXPLICITLY
        # via their --dtype flags and record effective_dtype in every row.
        self.device = device or "cuda"
        self.dtype = dtype
        self._model = None
        self._processor = None

    @abstractmethod
    def _load(self) -> None:
        """Load model + processor with 4-bit quant. Sets self._model and self._processor."""
        ...

    @abstractmethod
    def _single_turn(
        self,
        images: Sequence[Path],
        prompt: str,
        system: str,
        max_new_tokens: int = 8,
    ) -> str:
        ...

    @abstractmethod
    def _two_turn(
        self,
        sample_image: Path,
        sample_prompt: str,
        test_image: Path,
        test_prompt: str,
        system: str,
        max_new_tokens: int = 8,
    ) -> str:
        ...

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._load()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def call_single(self, images: Sequence[Path], prompt: str, system: str,
                    max_new_tokens: int = 8) -> LocalResponse:
        from ..utils.parsing import parse_yes_no
        self.ensure_loaded()
        t0 = time.monotonic()
        try:
            text = self._single_turn(images, prompt, system, max_new_tokens=max_new_tokens)
            return LocalResponse(
                raw=text,
                parsed_yes=parse_yes_no(text),
                model=self.model_id,
                provider=self.provider,
                latency_s=time.monotonic() - t0,
            )
        except Exception as e:                                  # noqa: BLE001
            return LocalResponse(
                raw="", parsed_yes=None, model=self.model_id, provider=self.provider,
                latency_s=time.monotonic() - t0,
                error=f"{type(e).__name__}: {e}",
            )

    def call_two_turn(
        self,
        sample_image: Path, sample_prompt: str,
        test_image: Path, test_prompt: str,
        system: str,
        max_new_tokens: int = 8,
    ) -> LocalResponse:
        from ..utils.parsing import parse_yes_no
        self.ensure_loaded()
        t0 = time.monotonic()
        try:
            text = self._two_turn(sample_image, sample_prompt, test_image, test_prompt, system,
                                  max_new_tokens=max_new_tokens)
            return LocalResponse(
                raw=text,
                parsed_yes=parse_yes_no(text),
                model=self.model_id,
                provider=self.provider,
                latency_s=time.monotonic() - t0,
            )
        except Exception as e:                                  # noqa: BLE001
            return LocalResponse(
                raw="", parsed_yes=None, model=self.model_id, provider=self.provider,
                latency_s=time.monotonic() - t0,
                error=f"{type(e).__name__}: {e}",
            )


def make_4bit_config():
    """4-bit NF4 BitsAndBytesConfig, double-quant on. Used by all local clients."""
    from transformers import BitsAndBytesConfig                 # noqa: WPS433
    import torch                                                # noqa: WPS433
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_kwargs_for_dtype(dtype: str) -> dict:
    """from_pretrained kwargs honoring the client's requested precision.

    'bfloat16'/'bf16' -> full bf16 weights, NO quantization (the A100-validated
    regime; ~9GB for a 4B model). '4bit'/'nf4'/'int4' -> NF4 BitsAndBytes
    (VRAM-constrained local GPUs). The 2026-08-07 assembly review found _load
    hardcoded 4-bit and silently ignored dtype='bfloat16' — which would have
    had Tier-2 BEHAVIOR measured at 4-bit while the geometry f(x) is read at
    bf16, breaking the closed-loop 'same model predicts itself' claim.
    Every runner must record the client's `effective_dtype` in its rows.
    """
    import torch                                                # noqa: WPS433
    d = dtype.lower()
    if d in ("bfloat16", "bf16"):
        return {"torch_dtype": torch.bfloat16}
    if d in ("4bit", "nf4", "int4"):
        return {"quantization_config": make_4bit_config()}
    raise ValueError(f"unknown dtype {dtype!r}; use 'bfloat16' or '4bit'")
