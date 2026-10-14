"""Anthropic client (Claude Sonnet 4.6) using the official `anthropic` SDK.

Uses the Messages API with base64 image blocks. For batch runs we expose a
`build_batch_request` helper for the Anthropic Message Batches API (50% off).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .base import APIClient
from ..utils.io import encode_image_b64


class AnthropicClient(APIClient):
    provider = "anthropic"

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        try:
            import anthropic                                   # noqa: WPS433
        except ImportError as e:
            raise RuntimeError("Install `anthropic` to use AnthropicClient") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing from environment")
        self._client = anthropic.Anthropic(api_key=api_key)

    def _image_block(self, path: Path) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encode_image_b64(path),
            },
        }

    def _single_turn(
        self,
        images: Sequence[Path],
        prompt: str,
        system: str,
        max_tokens: int = 8,
    ) -> tuple[str, dict[str, int]]:
        content = [self._image_block(img) for img in images]
        content.append({"type": "text", "text": prompt})
        msg = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        usage = {
            "prompt": getattr(msg.usage, "input_tokens", 0),
            "completion": getattr(msg.usage, "output_tokens", 0),
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
        first = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=16,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [self._image_block(sample_image), {"type": "text", "text": sample_prompt}],
            }],
        )
        ack = "".join(b.text for b in first.content if hasattr(b, "text")) or "READY"

        # Turn 2
        second = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [self._image_block(sample_image), {"type": "text", "text": sample_prompt}],
                },
                {"role": "assistant", "content": ack},
                {
                    "role": "user",
                    "content": [self._image_block(test_image), {"type": "text", "text": test_prompt}],
                },
            ],
        )
        text = "".join(b.text for b in second.content if hasattr(b, "text"))
        usage = {
            "prompt": (getattr(first.usage, "input_tokens", 0)
                       + getattr(second.usage, "input_tokens", 0)),
            "completion": (getattr(first.usage, "output_tokens", 0)
                           + getattr(second.usage, "output_tokens", 0)),
        }
        return text, usage

    @staticmethod
    def build_batch_request(custom_id: str, model: str, system: str,
                            messages: list[dict], max_tokens: int = 8) -> dict:
        return {
            "custom_id": custom_id,
            "params": {
                "model": model,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "messages": messages,
            },
        }
