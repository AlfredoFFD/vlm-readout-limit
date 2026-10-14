"""Factory for instantiating API clients by short name."""

from __future__ import annotations

from .base import APIClient


def get_client(name: str, **kwargs) -> APIClient:
    name = name.lower()
    if name in {"openai", "gpt", "gpt-4o", "gpt4o"}:
        from .openai_client import OpenAIClient
        return OpenAIClient(**kwargs)
    if name in {"anthropic", "claude", "claude-sonnet"}:
        from .anthropic_client import AnthropicClient
        return AnthropicClient(**kwargs)
    if name in {"gemini", "google", "gemini-2.5-pro"}:
        from .gemini_client import GeminiClient
        return GeminiClient(**kwargs)
    if name.startswith("mock"):
        from .mock_client import MockClient
        # Allow either "mock" (default mode) or "mock:perfect", "mock:chance", etc.
        mode = name.split(":", 1)[1] if ":" in name else "deterministic_yes"
        return MockClient(mode=mode, **kwargs)
    raise ValueError(f"Unknown provider '{name}'. Known: openai | anthropic | gemini | mock[:mode]")
