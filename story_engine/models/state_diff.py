from typing import Any

from dataclasses import dataclass, field

from story_engine.models.base import ModelMixin


@dataclass
class StateDiff(ModelMixin):
    """Represents changes to the narrative state after a scene is generated.

    Attributes:
        changes: Dictionary mapping dotted state paths to their new values.
    """
    changes: dict[str, Any] = field(default_factory=dict)
