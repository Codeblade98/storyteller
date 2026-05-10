"""Story writing style parameters."""

from dataclasses import dataclass


@dataclass
class StyleState:
    """Tracks narrative style and presentation parameters.

    Attributes:
        tone: Emotional tone of the story.
        vocab_level: Vocabulary complexity level.
        sentence_complexity: Sentence structure complexity (1-4).
    """
    tone: str = "hopeful"
    vocab_level: str = "simple"
    sentence_complexity: int = 2
