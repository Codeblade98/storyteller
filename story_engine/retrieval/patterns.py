"""Story pattern library for narrative structure templates."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from story_engine.models.base import ModelMixin


@dataclass
class StoryPattern(ModelMixin):
    """Predefined narrative structure pattern for story generation.

    Attributes:
        pattern_name: Name/identifier for the pattern.
        acts: List of narrative acts (e.g., setup, challenge, resolution).
        emotion_curve: Emotion intensity values for each act.
        genres: Story genres this pattern applies to.
        pacing_notes: Optional notes about pacing for the pattern.
    """
    pattern_name: str
    acts: list[str]
    emotion_curve: list[int]
    genres: list[str] = field(default_factory=list)
    pacing_notes: list[str] = field(default_factory=list)


class PatternLibrary:
    """Library of story patterns loaded from JSON files.

    Selects and adapts patterns based on genre and scene count needs.
    """
    def __init__(self, library_dir: Path | None = None) -> None:
        """Initialize the pattern library.

        Args:
            library_dir: Directory containing pattern JSON files. Uses default if None.
        """
        self.library_dir = library_dir or Path(__file__).parent / "pattern_library"

    def select(self, genre: str, target_scene_count: int) -> StoryPattern:
        """Select and adapt a pattern for the given genre and scene count.

        Finds a pattern matching the genre, then scales its acts and emotion curve
        to match the target scene count.

        Args:
            genre: Story genre to match patterns for.
            target_scene_count: Desired number of scenes.

        Returns:
            A StoryPattern adapted for the target scene count.
        """
        patterns = self._load_patterns()
        genre_lower = genre.lower()
        for pattern in patterns:
            if genre_lower in {item.lower() for item in pattern.genres}:
                return self._fit_scene_count(pattern, target_scene_count)
        return self._fit_scene_count(patterns[0], target_scene_count)

    def _load_patterns(self) -> list[StoryPattern]:
        """Load all story patterns from the library directory.

        Returns:
            List of loaded StoryPattern objects.

        Raises:
            RuntimeError: If no patterns are found in the library.
        """
        ## TODO: Add more patterns to the library and consider caching them if loading becomes expensive.
        patterns = []
        for path in sorted(self.library_dir.glob("*.json")):
            patterns.append(StoryPattern.model_validate(json.loads(path.read_text())))
        if not patterns:
            raise RuntimeError(f"No story patterns found in {self.library_dir}")
        return patterns

    def _fit_scene_count(self, pattern: StoryPattern, count: int) -> StoryPattern:
        """Adapt a pattern to a different scene count through interpolation.

        Args:
            pattern: Original pattern to adapt.
            count: Target number of scenes.

        Returns:
            A copy of the pattern with interpolated acts and emotion curve.
        """
        acts = pattern.acts
        curve = pattern.emotion_curve
        if len(acts) == count and len(curve) == count:
            return pattern

        ## TBD: More sophisticated interpolation could be done here, but for now we will just repeat elements as needed.
        fitted_acts = [acts[min(int(index * len(acts) / count), len(acts) - 1)] for index in range(count)]
        fitted_curve = [curve[min(int(index * len(curve) / count), len(curve) - 1)] for index in range(count)]
        return pattern.model_copy(update={"acts": fitted_acts, "emotion_curve": fitted_curve})
