from story_engine.models.scene import SceneOutput
from story_engine.models.story_spec import StorySpec


class SafetyValidator:
    """Validates scenes for safety constraints and age-appropriateness.

    Checks for forbidden elements and unsafe fear/violence levels.
    """
    def validate(self, output: SceneOutput, spec: StorySpec) -> list[str]:
        """Validate scene safety against story constraints.

        Args:
            output: Generated scene output to validate.
            spec: Story specification with safety constraints.

        Returns:
            List of failure messages if any constraints violated.
        """
        text = output.scene_text.lower()
        failures = []
        for forbidden in spec.forbidden_elements:
            if forbidden.lower() in text:
                failures.append(f"Forbidden element appeared: {forbidden}")

        fear_score = int(output.metadata.get("fear_score", 0))
        violence_score = int(output.metadata.get("violence_score", 0))
        if fear_score > spec.fear_level:
            failures.append(f"Fear score {fear_score} exceeds limit {spec.fear_level}")
        if violence_score > 0:
            failures.append("Violence score must remain 0 for MVP children stories")
        return failures
