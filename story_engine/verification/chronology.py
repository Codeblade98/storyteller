from story_engine.models.scene import SceneOutput


class ChronologyValidator:
    """Validates temporal consistency of scenes.

    Ensures scenes don't reference earlier completed scenes incorrectly.
    """
    def validate(self, output: SceneOutput, completed_scenes: list[str]) -> list[str]:
        """Validate temporal consistency of a scene.

        Args:
            output: Generated scene output to validate.
            completed_scenes: List of already completed scene IDs.

        Returns:
            List of failure messages if constraints violated.
        """
        failures = []
        previous_text = output.scene_text.lower()
        for scene_id in completed_scenes:
            if f"before {scene_id.lower()}" in previous_text:
                failures.append(f"Scene refers to happening before completed scene {scene_id}")
        if output.scene_id in completed_scenes:
            failures.append(f"Scene {output.scene_id} was already completed")
        return failures
