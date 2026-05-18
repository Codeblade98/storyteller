"""Minimal live endpoint check for the Groq LLM API.

This test is intentionally small: it sends a short random prompt to the
endpoint, then checks that a JSON response comes back.
"""

from __future__ import annotations

import json
import os
import random
import urllib.error

import pytest
from dotenv import load_dotenv

from story_engine.llm.client import GroqClient


load_dotenv()


pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY is required for the live Groq endpoint test",
)


def test_groq_client_live_endpoint() -> None:
    prompt = random.choice([
        "Reply with JSON: {\"ok\": true}.",
        "Return a tiny JSON object with one key named ping.",
        "Answer in JSON only: {\"status\": \"up\"}.",
    ])

    base_url = os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
    client = GroqClient(
        model=os.getenv("GROQ_MODEL", "allam-2-7b"),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=base_url,
    )

    try:
        response_text = client.complete_json(prompt, temperature=0.0)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        pytest.fail(f"Groq endpoint returned {exc.code} {exc.reason}: {body}")

    with open("groq_test_response.txt", "w") as f:
        f.write(response_text)

    assert response_text, "Expected non-empty response from Groq endpoint"
