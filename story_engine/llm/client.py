"""LLM client abstractions for scene generation.

This module uses the official OpenAI Python library for completions.
"""

import os
from typing import Protocol
import openai


class LLMClient(Protocol):
    """Protocol for LLM completion clients.

    Defines the interface that any LLM provider must implement.
    """
    def complete_json(self, prompt: str, *, temperature: float) -> str:
        ...


class OpenAICompatibleClient:
    """Client backed by the official OpenAI Python library.

    The constructor will configure `openai.api_key` and optionally
    `openai.api_base` when a custom base URL is provided.
    """
    def __init__(self, base_url: str | None, model: str, api_key: str | None = None) -> None:
        self.model = model
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if base_url:
            openai.api_base = base_url.rstrip("/")

    def complete_json(self, prompt: str, *, temperature: float) -> str:
        resp = openai.responses.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"]


class GroqClient(OpenAICompatibleClient):
    """Client for Groq API that reuses the OpenAI-compatible wrapper."""
    def __init__(self, base_url: str | None = None, model: str = "gemma-2-9b-it", api_key: str | None = None) -> None:
        super().__init__(base_url=base_url or "https://api.groq.com/openai/v1", model=model, api_key=api_key or os.getenv("GROQ_API_KEY", ""))
