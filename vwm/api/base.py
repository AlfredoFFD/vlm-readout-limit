"""Common interface for proprietary VLM API clients."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Default model IDs. Env overrides (OPENAI_MODEL / ANTHROPIC_MODEL /
# GEMINI_MODEL) are resolved at CLIENT INIT, not import time — the old
# import-time read ran before load_dotenv(), so .env overrides were silently
# dead (this is how gemini-2.5-flash shipped when .env asked for 2.5-pro).
_DEFAULT_MODELS: dict[str, str] = {
    "openai":    "gpt-4o-2024-11-20",
    "anthropic": "claude-sonnet-4-6",
    # gemini-2.5-pro requires thinking mode (rejects thinking_budget=0), which breaks
    # the single-token YES/NO regime used by the other providers. 2.5-flash is the
    # appropriate frontier-tier Gemini for this paradigm; Pro-with-thinking is the
    # analogue of the open-models CoT condition and is a separate future experiment.
    "gemini":    "gemini-2.5-flash",
}
_ENV_KEYS = {"openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL",
             "gemini": "GEMINI_MODEL"}


def resolve_model(provider: str) -> str:
    """Env override (post-dotenv) or the frozen default for `provider`."""
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get(_ENV_KEYS.get(provider, ""), "") or \
        _DEFAULT_MODELS.get(provider, "")


class _LazyModelRegistry(dict):
    """Back-compat shim: MODEL_REGISTRY[provider] now resolves lazily."""

    def __getitem__(self, provider: str) -> str:          # pragma: no cover - trivial
        return resolve_model(provider)

    def get(self, provider: str, default: str = "") -> str:
        return resolve_model(provider) or default


MODEL_REGISTRY = _LazyModelRegistry(_DEFAULT_MODELS)


@dataclass
class APIResponse:
    raw: str
    parsed_yes: bool | None
    model: str
    provider: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class APIClient(ABC):
    """Each provider implements answer_yes_no(image_paths, prompt, history)."""

    provider: str = "base"

    def __init__(self, model: str | None = None, max_retries: int = 4, retry_base_s: float = 1.0):
        self.model = model or MODEL_REGISTRY.get(self.provider, "")
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s

    @abstractmethod
    def _single_turn(
        self,
        images: Sequence[Path],
        prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> tuple[str, dict[str, int]]:
        """Provider-specific single-turn call. Returns (text, usage)."""
        ...

    @abstractmethod
    def _two_turn(
        self,
        sample_image: Path,
        sample_prompt: str,
        test_image: Path,
        test_prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> tuple[str, dict[str, int]]:
        """Provider-specific two-turn call for the C2 sequential condition.

        Turn 1: send sample_image + sample_prompt, capture assistant ack.
        Turn 2: append assistant ack to context, send test_image + test_prompt.
        Returns (final_text, usage).
        """
        ...

    def call_single(
        self,
        images: Sequence[Path],
        prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> APIResponse:
        from ..utils.parsing import parse_yes_no  # avoid circular at import
        last_err: str | None = None
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                text, usage = self._single_turn(images, prompt, system, max_tokens=max_tokens)
                return APIResponse(
                    raw=text,
                    parsed_yes=parse_yes_no(text),
                    model=self.model,
                    provider=self.provider,
                    latency_s=time.monotonic() - t0,
                    prompt_tokens=usage.get("prompt", 0),
                    completion_tokens=usage.get("completion", 0),
                )
            except Exception as e:                              # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(self.retry_base_s * (2 ** attempt))
        return APIResponse(
            raw="", parsed_yes=None, model=self.model, provider=self.provider,
            latency_s=0.0, error=last_err,
        )

    def call_two_turn(
        self,
        sample_image: Path,
        sample_prompt: str,
        test_image: Path,
        test_prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> APIResponse:
        from ..utils.parsing import parse_yes_no
        last_err: str | None = None
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                text, usage = self._two_turn(
                    sample_image, sample_prompt, test_image, test_prompt, system, max_tokens
                )
                return APIResponse(
                    raw=text,
                    parsed_yes=parse_yes_no(text),
                    model=self.model,
                    provider=self.provider,
                    latency_s=time.monotonic() - t0,
                    prompt_tokens=usage.get("prompt", 0),
                    completion_tokens=usage.get("completion", 0),
                )
            except Exception as e:                              # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(self.retry_base_s * (2 ** attempt))
        return APIResponse(
            raw="", parsed_yes=None, model=self.model, provider=self.provider,
            latency_s=0.0, error=last_err,
        )
