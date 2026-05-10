from typing import Any, Literal

from dataclasses import dataclass, field

from story_engine.models.base import ModelMixin


@dataclass
class ScenePlan(ModelMixin):
    """Represents a planned scene with goals, dependencies, and constraints.

    Attributes:
        scene_id: Unique identifier for the scene.
        act: The narrative act this scene belongs to (e.g., 'setup', 'challenge').
        goal: The primary goal or objective for this scene.
        emotion_level: Intensity level of emotions in this scene (0-5).
        depends_on: List of scene IDs this scene depends on (deprecated in favor of hard_dependencies).
        hard_dependencies: List of scene IDs that must be completed before this scene.
        soft_dependencies: List of scene IDs that should preferably be completed first.
        required_context: List of state keys needed to generate this scene.
        constraints: List of constraints to apply during scene generation.
    """
    scene_id: str
    act: str
    goal: str
    emotion_level: int = 0
    depends_on: list[str] = field(default_factory=list)
    hard_dependencies: list[str] = field(default_factory=list)
    soft_dependencies: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class SceneOutput(ModelMixin):
    """Represents the generated output of a scene.

    Attributes:
        scene_id: Unique identifier for the scene.
        scene_text: The generated narrative text for the scene.
        state_diff: Dictionary of state changes to apply after this scene.
        metadata: Additional metadata about the scene generation.
        status: Generation status ('generated' or 'repaired').

    Raises:
        ValueError: If scene_text is empty.
    """
    scene_id: str
    scene_text: str
    state_diff: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: Literal["generated", "repaired"] = "generated"

    def __post_init__(self) -> None:
        if not self.scene_text:
            raise ValueError("scene_text must not be empty")
