"""Safety and content ratings for story generation."""

from dataclasses import dataclass, field


@dataclass
class SafetyState:
    """Tracks safety metrics and constraints for the story.

    Attributes:
        fear_score: Current fear/tension level in story (0-5).
        violence_score: Violence content level (should be 0 for children's stories).
        forbidden_elements: List of elements that must not appear.
    """
    fear_score: int = 0
    violence_score: int = 0
    forbidden_elements: list[str] = field(default_factory=list)
