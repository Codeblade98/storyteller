from story_engine.models.scene import SceneOutput


class InvariantValidator:
    """Validates invariants that must hold for all scenes.

    Checks basic requirements like non-empty text and proper metadata.
    """
    def validate(self, output: SceneOutput) -> list[str]:
        """Validate scene invariants.

        Args:
            output: Generated scene output to validate.

        Returns:
            List of failure messages if invariants violated.
        """
        failures = []
        if not output.scene_text.strip():
            failures.append("Scene text is empty")
        if not isinstance(output.state_diff, dict):
            failures.append("State diff must be an object")
        if "full_history" in output.metadata:
            failures.append("Scene metadata attempted to retain full history")
        return failures
