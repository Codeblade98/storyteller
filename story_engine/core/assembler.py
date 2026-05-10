from story_engine.models.scene import SceneOutput


class StoryAssembler:
    """Combines individual scene texts into a complete story.
    
    Joins scene outputs in order with appropriate spacing.
    """
    def assemble(self, scenes: list[SceneOutput]) -> str:
        """Assemble a complete story from ordered scene outputs.

        Args:
            scenes: List of SceneOutput objects in narrative order.

        Returns:
            Complete story text with scenes joined by double newlines.
        """
        return "\n\n".join(scene.scene_text.strip() for scene in scenes)
