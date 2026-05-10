from story_engine.models.story_spec import StoryInput, StorySpec


class StorySpecBuilder:
    """Converts high-level story input into detailed generation specifications.

    Determines narrative parameters like vocabulary, sentence complexity,
    and moral themes based on age group and story preferences.
    """
    def build(self, story_input: StoryInput) -> StorySpec:
        """Build a detailed story specification from user input.

        Determines vocabulary level, sentence complexity, scene count,
        and moral theme based on age group and input parameters.

        Args:
            story_input: High-level user input for story generation.

        Returns:
            A StorySpec ready for planning and generation.
        """
        target_scene_count = 6 if story_input.length == "short" else 8
        vocab_level, sentence_complexity = self._age_controls(story_input.age_group)
        moral_theme = self._infer_theme(story_input.topic)

        return StorySpec(
            topic=story_input.topic,
            age_group=story_input.age_group,
            genre=story_input.genre,
            fear_level=story_input.fear_level,
            length=story_input.length,
            vocab_level=vocab_level,
            sentence_complexity=sentence_complexity,
            allowed_conflict=min(story_input.fear_level, 3),
            forbidden_elements=["death", "gore", "explicit violence", "cruel punishment"],
            moral_theme=moral_theme,
            target_scene_count=target_scene_count,
        )

    def _age_controls(self, age_group: str) -> tuple[str, int]:
        """Determine vocabulary and sentence complexity for an age group.

        Args:
            age_group: Target age group.

        Returns:
            Tuple of (vocab_level, sentence_complexity).
        """
        if age_group == "3-5":
            return "early", 1
        if age_group in {"6-8", "7-9"}:
            return "simple", 2
        return "clear", 3

    def _infer_theme(self, topic: str) -> str:
        """Infer an appropriate moral theme from the story topic.

        Args:
            topic: The story topic.

        Returns:
            A moral theme string appropriate for the topic.
        """
        lowered = topic.lower()
        if "friend" in lowered:
            return "friendship"
        if "brave" in lowered or "fear" in lowered:
            return "courage"
        if "share" in lowered:
            return "generosity"
        return "kindness"
