"""Story specification models for converting user input to detailed specifications."""

from typing import Literal

from dataclasses import dataclass

from story_engine.models.base import ModelMixin


AgeGroup = Literal["3-5", "6-8", "7-9", "9-12"]
"""Age group categories for story targeting."""

Length = Literal["short", "medium"]
"""Story length categories."""


@dataclass
class StoryInput(ModelMixin):
    """User input for generating a story.

    Attributes:
        topic: Main topic or subject of the story (2-120 characters).
        age_group: Target age group for the story.
        genre: Story genre (2-40 characters).
        fear_level: Level of acceptable fear/tension (0-5).
        length: Desired story length (short or medium).

    Raises:
        ValueError: If topic/genre lengths or fear_level are out of range.
    """
    topic: str
    age_group: AgeGroup = "7-9"
    genre: str = "fantasy"
    fear_level: int = 2
    length: Length = "short"

    def __post_init__(self) -> None:
        self.topic = " ".join(self.topic.strip().split())
        self.genre = " ".join(self.genre.strip().split())
        if not 2 <= len(self.topic) <= 120:
            raise ValueError("topic must be between 2 and 120 characters")
        if not 2 <= len(self.genre) <= 40:
            raise ValueError("genre must be between 2 and 40 characters")
        if self.fear_level < 0 or self.fear_level > 5:
            raise ValueError("fear_level must be between 0 and 5")


@dataclass
class StorySpec(ModelMixin):
    """Detailed specification generated from user input for story generation.

    Attributes:
        topic: Main topic of the story.
        age_group: Target age group.
        genre: Story genre.
        fear_level: Acceptable fear level (0-5).
        length: Story length.
        vocab_level: Vocabulary complexity level.
        sentence_complexity: Sentence structure complexity (1-4).
        allowed_conflict: Maximum allowed conflict intensity (0-5).
        forbidden_elements: List of elements that must not appear.
        moral_theme: Primary moral lesson.
        target_scene_count: Expected number of scenes to generate.

    Raises:
        ValueError: If numeric values are out of acceptable ranges.
    """
    topic: str
    age_group: AgeGroup
    genre: str
    fear_level: int
    length: Length
    vocab_level: Literal["early", "simple", "clear", "rich"]
    sentence_complexity: int
    allowed_conflict: int
    forbidden_elements: list[str]
    moral_theme: str
    target_scene_count: int

    def __post_init__(self) -> None:
        if not 0 <= self.fear_level <= 5:
            raise ValueError("fear_level must be between 0 and 5")
        if not 1 <= self.sentence_complexity <= 4:
            raise ValueError("sentence_complexity must be between 1 and 4")
        if not 0 <= self.allowed_conflict <= 5:
            raise ValueError("allowed_conflict must be between 0 and 5")
        if not 3 <= self.target_scene_count <= 10:
            raise ValueError("target_scene_count must be between 3 and 10")
