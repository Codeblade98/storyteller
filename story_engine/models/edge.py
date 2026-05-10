from dataclasses import dataclass, field

from story_engine.models.base import ModelMixin


@dataclass
class EdgeContext(ModelMixin):
    """Represents a directed edge between two scenes in the dependency graph.

    Attributes:
        source: Identifier of the source scene.
        target: Identifier of the target scene.
        dependency_type: Type of dependency ('hard' or 'soft').
        required_context: List of state keys required when processing the target.
        ignored_context: List of state keys to exclude during processing.
    """
    source: str
    target: str
    dependency_type: str = "hard"
    required_context: list[str] = field(default_factory=list)
    ignored_context: list[str] = field(default_factory=list)
    transforms: dict[str, str] = field(default_factory=dict)
