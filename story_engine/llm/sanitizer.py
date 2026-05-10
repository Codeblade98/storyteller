"""Sanitization utilities for LLM JSON responses."""


def sanitize_json_text(raw: str) -> str:
    """Extract and clean JSON from LLM response text.

    Handles responses containing markdown code blocks,
    extra text, and common formatting issues.

    Args:
        raw: Raw text from LLM potentially containing JSON.

    Returns:
        Extracted and cleaned JSON string.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start : end + 1].strip()
