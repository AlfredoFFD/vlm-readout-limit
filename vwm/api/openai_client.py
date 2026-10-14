"""OpenAI client (GPT-4o family) using the official `openai` SDK.

We use chat.completions with image_url=data:image/png;base64,... payloads. For batch
runs we expose a `build_batch_request` helper that emits JSONL lines suitable for the
50%-off Batch API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .base import APIClient
from ..utils.io import encode_image_b64


class OpenAIClient(APIClient):
    provider = "openai"

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        try:
            from openai import OpenAI                          # noqa: WPS433
        except ImportError as e:
            raise RuntimeError("Install `openai` to use OpenAIClient") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing from environment")
        self._client = OpenAI(api_key=api_key)

    # --- single-turn (C1 Simultaneous) ---
    def _single_turn(
        self,
        images: Sequence[Path],
        prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> tuple[str, dict[str, int]]:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = encode_image_b64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        msg = resp.choices[0].message.content or ""
        usage = {
            "prompt": getattr(resp.usage, "prompt_tokens", 0),
            "completion": getattr(resp.usage, "completion_tokens", 0),
        }
        return msg, usage

    # --- two-turn (C2 Sequential-with-context) ---
    def _two_turn(
        self,
        sample_image: Path,
        sample_prompt: str,
        test_image: Path,
        test_prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> tuple[str, dict[str, int]]:
        b64_sample = encode_image_b64(sample_image)
        # Turn 1
        first = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": sample_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sample}"}},
                ]},
            ],
            max_tokens=16,
            temperature=0.0,
        )
        ack = first.choices[0].message.content or "READY"

        # Turn 2 — append the assistant ack and send the test image
        b64_test = encode_image_b64(test_image)
        second = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": sample_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sample}"}},
                ]},
                {"role": "assistant", "content": ack},
                {"role": "user", "content": [
                    {"type": "text", "text": test_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_test}"}},
                ]},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        msg = second.choices[0].message.content or ""
        usage = {
            "prompt": (getattr(first.usage, "prompt_tokens", 0)
                       + getattr(second.usage, "prompt_tokens", 0)),
            "completion": (getattr(first.usage, "completion_tokens", 0)
                           + getattr(second.usage, "completion_tokens", 0)),
        }
        return msg, usage

    # --- Batch API request builder ---
    @staticmethod
    def build_batch_request(custom_id: str, model: str, messages: list[dict],
                            max_tokens: int = 8, temperature: float = 0.0) -> dict:
        """Return one JSONL row for the OpenAI Batch API (50% off, 24-h SLA)."""
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }
