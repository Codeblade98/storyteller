from typing import Any

from story_engine.models.scene import SceneOutput


class CharacterValidator:
    """Validates that scenes properly manage character state.

    Ensures characters are introduced and maintained appropriately.
    """
    def validate(self, output: SceneOutput, state: dict[str, Any]) -> list[str]:
        """Validate character state handling in a scene.

        Args:
            output: Generated scene output to validate.
            state: Current narrative state.

        Returns:
            List of failure messages if constraints violated.
        """
        failures = []
        characters = state.get("character_state", {})
        diff_keys = set(output.state_diff)
        introduced = [key for key in diff_keys if key.startswith("character_state.") or key.startswith("character.")]
        if not characters and not introduced:
            failures.append("Scene did not introduce or preserve any character state")
        return failures
