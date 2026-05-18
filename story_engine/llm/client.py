"""LLM client abstractions for scene generation.

This module uses the official OpenAI Python library for completions.
"""

import os
import json
import logging
from typing import Protocol, TypeVar, Any
logger = logging.getLogger("story_engine.llm")
T = TypeVar("T")



class LLMClient(Protocol):
    """Protocol for LLM completion clients.

    Defines the interface that any LLM provider must implement.
    """
    def complete_json(self, prompt: str, *, temperature: float, system_prompt: str | None = None) -> str:
        ...


class OpenAICompatibleClient:
    """Client backed by the official OpenAI Python library.

    The constructor will configure `openai.api_key` and optionally
    `openai.api_base` when a custom base URL is provided.
    """
    def __init__(self, base_url: str | None, model: str, api_key: str | None = None) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ValueError("openai package is required to use OpenAICompatibleClient") from exc

        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        
        if not base_url:
            raise ValueError("base_url is required for OpenAICompatibleClient")
        
        if not api_key:
            raise ValueError("API key is required for OpenAICompatibleClient")
            
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def complete_json(self, prompt: str, *, temperature: float, system_prompt: str | None = None) -> str:
        messages = self._messages(prompt, system_prompt)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.debug("json_mode_unsupported model=%s error=%s", self.model, exc)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        
        return resp.choices[0].message.content

    def complete_structured(
        self,
        prompt: str,
        *,
        temperature: float,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> tuple[str, T]:
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=self._messages(prompt, system_prompt),
                temperature=temperature,
                response_format=response_model,
            )
        except Exception:
            raise

        message = completion.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"LLM refusal: {refusal}")

        parsed: Any = message.parsed
        raw = message.content
        if not raw:
            if hasattr(parsed, "model_dump"):
                raw = json.dumps(parsed.model_dump(), ensure_ascii=True)
            else:
                raw = json.dumps(parsed, ensure_ascii=True)

        return raw, parsed

    def _messages(self, user_payload: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_payload})
        return messages


class GroqClient(OpenAICompatibleClient):
    """Client for Groq API that reuses the OpenAI-compatible wrapper."""
    def __init__(self, base_url: str | None = None, model: str = "gemma-2-9b-it", api_key: str | None = None) -> None:
        super().__init__(base_url=base_url or "https://api.groq.com/openai/v1", model=model, api_key=api_key or os.getenv("GROQ_API_KEY", ""))
