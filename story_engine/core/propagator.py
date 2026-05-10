from typing import Any

from story_engine.models.node import SceneNode


class StatePropagator:
    """Filters narrative state for individual scene generation.

    Determines which state keys are relevant for a given scene based on
    its plan and incoming dependencies.
    """
    def filter_for_node(self, state: dict[str, Any], node: SceneNode) -> dict[str, Any]:
        """Filter state to only keys required by a scene node.

        Includes context from the scene plan and all incoming edges,
        excluding explicitly ignored context.

        Args:
            state: Complete narrative state.
            node: Scene node requiring filtered state.

        Returns:
            Dictionary with only relevant state keys for this scene.
        """
        required = set(node.plan.required_context)
        for edge in node.incoming_edges:
            required.update(edge.required_context)
            required.difference_update(edge.ignored_context)

        if not required:
            required = {"style_state", "safety_state"}
        return {key: state[key] for key in sorted(required) if key in state}
