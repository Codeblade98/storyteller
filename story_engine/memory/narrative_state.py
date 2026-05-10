"""Narrative progression state tracking."""

from dataclasses import dataclass, field


@dataclass
class NarrativeState:
    """Tracks overall story structure and progression.

    Attributes:
        active_arc: Current narrative arc or story phase.
        completed_scenes: List of completed scene IDs.
    """
    active_arc: str = "setup"
    completed_scenes: list[str] = field(default_factory=list)
