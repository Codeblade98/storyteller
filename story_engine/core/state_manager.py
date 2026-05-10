from copy import deepcopy
from typing import Any

from story_engine.models.story_spec import StorySpec


class StateManager:
    """Manages narrative state throughout story execution.

    Initializes state, applies diffs from scene outputs, and maintains
    a hierarchical state structure across multiple domains.
    """
    def initial_state(self, spec: StorySpec) -> dict[str, Any]:
        """Create the initial narrative state from a story specification.

        Initializes all state domains: world, character, narrative, style, safety.

        Args:
            spec: Story specification defining initial parameters.

        Returns:
            A complete initial state dictionary.
        """
        return {
            "world_state": {"location": "story_start", "time": "beginning"},
            "character_state": {},
            "narrative_state": {"active_arc": "setup", "completed_scenes": []},
            "style_state": {
                "tone": "hopeful",
                "vocab_level": spec.vocab_level,
                "sentence_complexity": spec.sentence_complexity,
            },
            "safety_state": {
                "fear_score": spec.fear_level,
                "violence_score": 0,
                "forbidden_elements": spec.forbidden_elements,
            },
        }

    def apply_diff(self, state: dict[str, Any], diff: dict[str, Any], scene_id: str) -> dict[str, Any]:
        """Apply state changes from a scene output to the current state.

        Handles dotted path updates, list appending, and special keys like
        'time.advance'. Automatically tracks completed scenes.

        Args:
            state: Current narrative state.
            diff: Dictionary of state changes with dotted paths as keys.
            scene_id: ID of the scene being applied.

        Returns:
            Updated state dictionary.
        """
        next_state = deepcopy(state)
        for path, value in diff.items():
            if path.endswith(".add"):
                self._append(next_state, path.removesuffix(".add"), value)
            elif path.endswith(".remove"):
                self._remove(next_state, path.removesuffix(".remove"), value)
            elif path == "time.advance":
                self._set(next_state, "world_state.time", value)
            else:
                self._set(next_state, path, value)

        completed = next_state.setdefault("narrative_state", {}).setdefault("completed_scenes", [])
        if scene_id not in completed:
            completed.append(scene_id)
        return next_state

    def _set(self, state: dict[str, Any], dotted_path: str, value: Any) -> None:
        """Set a value in state using a dotted path.

        Args:
            state: State dictionary to modify.
            dotted_path: Dotted path to the value (e.g., 'world.location').
            value: Value to set.
        """
        parts = self._normalize_path(dotted_path)
        cursor = state
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    def _append(self, state: dict[str, Any], dotted_path: str, value: Any) -> None:
        """Append value(s) to a list in state using a dotted path.

        Handles both single values and lists, avoiding duplicates.

        Args:
            state: State dictionary to modify.
            dotted_path: Dotted path to the list.
            value: Value or list of values to append.
        """
        parts = self._normalize_path(dotted_path)
        cursor = state
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        existing = cursor.setdefault(parts[-1], [])
        if not isinstance(existing, list):
            cursor[parts[-1]] = [existing]
            existing = cursor[parts[-1]]
        if isinstance(value, list):
            existing.extend(item for item in value if item not in existing)
        elif value not in existing:
            existing.append(value)

    def _remove(self, state: dict[str, Any], dotted_path: str, value: Any) -> None:
        parts = self._normalize_path(dotted_path)
        cursor = state
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        existing = cursor.get(parts[-1])
        if isinstance(existing, list):
            values = value if isinstance(value, list) else [value]
            cursor[parts[-1]] = [item for item in existing if item not in values]

    def _normalize_path(self, dotted_path: str) -> list[str]:
        """Normalize a dotted path using configured aliases.

        Handles shortcuts like 'world' -> 'world_state' and 'location' -> 'world_state.location'.

        Args:
            dotted_path: Path that may contain aliases.

        Returns:
            List of normalized path components.
        """
        aliases = {
            "character": "character_state",
            "characters": "character_state",
            "world": "world_state",
            "narrative": "narrative_state",
            "style": "style_state",
            "safety": "safety_state",
            "inventory": "character_state.inventory",
            "active_arc": "narrative_state.active_arc",
            "location": "world_state.location",
            "time": "world_state.time",
        }
        normalized = aliases.get(dotted_path, dotted_path)
        for prefix, replacement in aliases.items():
            if dotted_path.startswith(prefix + "."):
                normalized = replacement + dotted_path[len(prefix) :]
                break
        return normalized.split(".")
