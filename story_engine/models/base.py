import json
from dataclasses import asdict, replace
from typing import Any, Self


class ModelMixin:
    """Mixin providing utility methods for model serialization and copying."""
    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> Self:
        """Create a model instance from a dictionary of data.

        Args:
            data: Dictionary containing keyword arguments for the model.

        Returns:
            An instance of the model class initialized with the provided data.
        """
        return cls(**data)

    def model_copy(self, update: dict[str, Any] | None = None) -> Self:
        """Create a shallow copy of the model with optional updates.

        Args:
            update: Optional dictionary of fields to update in the copy.

        Returns:
            A new instance with updated fields.
        """
        return replace(self, **(update or {}))

    def model_dump(self) -> dict[str, Any]:
        """Convert the model to a dictionary.

        Returns:
            Dictionary representation of the model.
        """
        return asdict(self)

    def model_dump_json(self) -> str:
        """Convert the model to a JSON string.

        Returns:
            JSON string representation of the model.
        """
        return json.dumps(self.model_dump())
