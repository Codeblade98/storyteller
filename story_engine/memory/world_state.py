"""Story world state tracking."""

from dataclasses import dataclass


@dataclass
class WorldState:
    """Tracks the story's world/setting state.

    Attributes:
        location: Current location or setting in the story.
        time: Current time or time period in the narrative.
    """
    location: str = "undisclosed"
    time: str = "beginning"
