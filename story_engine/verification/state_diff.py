from typing import Any

from story_engine.models.scene import SceneOutput


class StateDiffValidator:
    allowed_roots = {"world_state", "character_state", "narrative_state", "style_state", "safety_state"}
    aliases = {
        "character": "character_state",
        "characters": "character_state",
        "world": "world_state",
        "narrative": "narrative_state",
        "style": "style_state",
        "safety": "safety_state",
        "active_arc": "narrative_state.active_arc",
        "location": "world_state.location",
        "time": "world_state.time",
    }
    immutable_paths = {"narrative_state.completed_scenes"}

    def validate(self, output: SceneOutput, state: dict[str, Any]) -> list[str]:
        failures = []
        if not output.state_diff:
            failures.append("State diff is empty")
            return failures

        for path, value in output.state_diff.items():
            normalized = self.normalize_path(path)
            root = normalized.split(".")[0]
            if root not in self.allowed_roots and path != "time.advance":
                failures.append(f"Illegal state diff root: {path}")
            if normalized in self.immutable_paths:
                failures.append(f"Scene cannot mutate managed path: {path}")
            if path.endswith(".remove") and not self._can_remove(normalized.removesuffix(".remove"), value, state):
                failures.append(f"Cannot remove missing state value: {path}")
            if root == "safety_state" and "forbidden_elements" in normalized:
                failures.append("Scene cannot mutate safety forbidden elements")
        return failures

    def normalize_path(self, dotted_path: str) -> str:
        if dotted_path == "time.advance":
            return "world_state.time"
        normalized = self.aliases.get(dotted_path, dotted_path)
        for prefix, replacement in self.aliases.items():
            if dotted_path.startswith(prefix + "."):
                normalized = replacement + dotted_path[len(prefix) :]
                break
        return normalized.removesuffix(".add")

    def _can_remove(self, dotted_path: str, value: Any, state: dict[str, Any]) -> bool:
        cursor: Any = state
        for part in dotted_path.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return False
            cursor = cursor[part]
        if isinstance(cursor, list):
            return value in cursor
        return cursor == value
