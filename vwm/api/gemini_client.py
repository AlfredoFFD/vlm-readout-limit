"""Gemini client (gemini-2.5-pro) using the official `google-genai` SDK.

The new SDK (genai.Client) replaces the older google-generativeai package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .base import APIClient


class GeminiClient(APIClient):
    provider = "gemini"

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        # This client runs the single-token no-thinking regime
        # (thinking_budget=0 on every call). Gemini "pro" reasoning models
        # REJECT that config, which the blanket retry would convert into 600
        # silent error rows (cross-model review, 2026-08-07). Fail loud at
        # construction instead.
        if "pro" in (self.model or "").lower():
            raise ValueError(
                f"GeminiClient is pinned to the no-thinking single-token regime; "
                f"{self.model!r} requires thinking mode and would error on every "
                f"call. Use gemini-2.5-flash (default), or implement a separate "
                f"thinking-mode client as the CoT-condition analogue.")
        try:
            from google import genai                            # noqa: WPS433
            from google.genai import types as _genai_types      # noqa: WPS433
        except ImportError as e:
            raise RuntimeError("Install `google-genai` to use GeminiClient") from e
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing from environment")
        self._client = genai.Client(api_key=api_key)
        self._types = _genai_types

    def _load_image_part(self, path: Path):
        with open(path, "rb") as f:
            data = f.read()
        return self._types.Part.from_bytes(data=data, mime_type="image/png")

    def _single_turn(
        self,
        images: Sequence[Path],
        prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> tuple[str, dict[str, int]]:
        parts = [self._load_image_part(img) for img in images]
        parts.append(self._types.Part.from_text(text=prompt))
        contents = [self._types.Content(role="user", parts=parts)]
        # gemini-2.5-pro is a reasoning model — its hidden "thinking" trace eats the
        # token budget before any visible YES/NO is emitted. Disable thinking explicitly
        # and keep a small headroom on max_output_tokens for the visible answer.
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=max(max_tokens, 32),
            thinking_config=self._types.ThinkingConfig(thinking_budget=0),
        )
        resp = self._client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        text = (resp.text or "").strip()
        meta = getattr(resp, "usage_metadata", None)
        usage = {
            "prompt": getattr(meta, "prompt_token_count", 0) if meta else 0,
            "completion": getattr(meta, "candidates_token_count", 0) if meta else 0,
        }
        return text, usage

    def _two_turn(
        self,
        sample_image: Path,
        sample_prompt: str,
        test_image: Path,
        test_prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> tuple[str, dict[str, int]]:
        # Turn 1
        config1 = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=32,
            thinking_config=self._types.ThinkingConfig(thinking_budget=0),
        )
        turn1_user = self._types.Content(role="user", parts=[
            self._load_image_part(sample_image),
            self._types.Part.from_text(text=sample_prompt),
        ])
        resp1 = self._client.models.generate_content(
            model=self.model, contents=[turn1_user], config=config1,
        )
        ack = (resp1.text or "READY").strip()
        ack_msg = self._types.Content(role="model", parts=[self._types.Part.from_text(text=ack)])

        # Turn 2
        turn2_user = self._types.Content(role="user", parts=[
            self._load_image_part(test_image),
            self._types.Part.from_text(text=test_prompt),
        ])
        config2 = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=max(max_tokens, 32),
            thinking_config=self._types.ThinkingConfig(thinking_budget=0),
        )
        resp2 = self._client.models.generate_content(
            model=self.model,
            contents=[turn1_user, ack_msg, turn2_user],
            config=config2,
        )
        text = (resp2.text or "").strip()
        meta1 = getattr(resp1, "usage_metadata", None)
        meta2 = getattr(resp2, "usage_metadata", None)
        usage = {
            "prompt": (
                (getattr(meta1, "prompt_token_count", 0) if meta1 else 0)
                + (getattr(meta2, "prompt_token_count", 0) if meta2 else 0)
            ),
            "completion": (
                (getattr(meta1, "candidates_token_count", 0) if meta1 else 0)
                + (getattr(meta2, "candidates_token_count", 0) if meta2 else 0)
            ),
        }
        return text, usage
