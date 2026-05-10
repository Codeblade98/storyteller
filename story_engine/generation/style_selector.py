from story_engine.models.story_spec import StorySpec


class StyleSelector:
    """Selects appropriate style parameters for story generation.

    Determines tone, vocabulary level, and sentence complexity based on spec.
    """
    def select(self, spec: StorySpec) -> dict[str, str | int]:
        """Select style parameters appropriate for the story specification.

        Args:
            spec: Story specification defining parameters.

        Returns:
            Dictionary with style keys: tone, vocab_level, sentence_complexity.
        """
        return {
            "tone": "hopeful",
            "vocab_level": spec.vocab_level,
            "sentence_complexity": spec.sentence_complexity,
        }
