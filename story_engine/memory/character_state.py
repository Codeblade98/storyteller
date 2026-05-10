"""Character state tracking for narrative."""

from typing import Any

from dataclasses import dataclass, field


@dataclass
class CharacterState:
    """Tracks state information for all story characters.

    Attributes:
        characters: Dictionary mapping character names to their attributes.
    """
    characters: dict[str, dict[str, Any]] = field(default_factory=dict)
