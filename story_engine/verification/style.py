from story_engine.models.scene import SceneOutput
from story_engine.models.story_spec import StorySpec


class StyleValidator:
    """Validates writing style matches story specification.

    Checks length, sentence complexity, and other style parameters.
    """
    def validate(self, output: SceneOutput, spec: StorySpec) -> list[str]:
        """Validate scene style against story specification.

        Args:
            output: Generated scene output to validate.
            spec: Story specification with style constraints.

        Returns:
            List of failure messages if constraints violated.
        """
        failures = []
        words = output.scene_text.split()
        if spec.length == "short" and len(words) > 180:
            failures.append("Scene is too long for compact MVP generation")
        if spec.sentence_complexity <= 2:
            long_sentences = [sentence for sentence in output.scene_text.split(".") if len(sentence.split()) > 24]
            if long_sentences:
                failures.append("Sentence complexity is too high for age group")
        return failures
