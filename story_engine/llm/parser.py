"""JSON parsing utilities for LLM responses."""

import json
from typing import Any, TypeVar

from story_engine.llm.sanitizer import sanitize_json_text

T = TypeVar("T")


class JSONParser:
    """Parses JSON responses from LLMs into typed models.

    Sanitizes raw LLM output and validates against model schemas.
    """
    def parse_model(self, raw: str, model: type[T]) -> T:
        """Parse LLM response into a typed model instance.

        Args:
            raw: Raw response string from LLM.
            model: Target model class to parse into.

        Returns:
            Instance of the model class.
        """
        data = self.parse_dict(raw)
        if hasattr(model, "model_validate"):
            return model.model_validate(data)
        return model(**data)

    def parse_dict(self, raw: str) -> dict[str, Any]:
        """Parse LLM response into a dictionary.

        Args:
            raw: Raw response string from LLM.

        Returns:
            Parsed dictionary from the JSON.
        """
        return json.loads(sanitize_json_text(raw))
