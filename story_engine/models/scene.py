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
    act_id: str = ""
    state_updates: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.act_id:
            self.act_id = self.act


@dataclass
class ActNode(ModelMixin):
    """Represents a high-level act in the global story spine."""
    act_id: str
    act_number: int
    act_summary: str
    depends_on: list[str] = field(default_factory=list)
    expected_scene_count: int = 1
    expected_entrance_state: dict[str, Any] = field(default_factory=dict)
    expected_exit_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneDAGPlan(ModelMixin):
    """Scene DAG expansion for a single act."""
    act_id: str
    scenes: list[ScenePlan] = field(default_factory=list)


@dataclass
class GlobalBlackboard(ModelMixin):
    """Shared planning state mutated by scene state updates."""
    state: dict[str, Any] = field(default_factory=dict)

    def apply(self, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            self.state[key] = value

    def satisfies(self, expected: dict[str, Any]) -> bool:
        return all(self.state.get(key) == value for key, value in expected.items())


@dataclass
class StoryPlan(ModelMixin):
    """Nested two-level DAG plan for a story."""
    acts: list[ActNode]
    scene_dags: list[SceneDAGPlan]
    global_blackboard: GlobalBlackboard = field(default_factory=GlobalBlackboard)

    def flatten_scenes(self) -> list[ScenePlan]:
        scenes: list[ScenePlan] = []
        act_by_id = {act.act_id: act for act in self.acts}
        dag_by_act = {dag.act_id: dag for dag in self.scene_dags}

        for act in self.acts:
            act_scenes = list(dag_by_act.get(act.act_id, SceneDAGPlan(act_id=act.act_id)).scenes)
            if not act_scenes:
                continue

            prerequisite_terminal_scenes: list[str] = []
            for parent_act_id in act.depends_on:
                parent_scenes = dag_by_act.get(parent_act_id, SceneDAGPlan(act_id=parent_act_id)).scenes
                parent_scene_ids = {scene.scene_id for scene in parent_scenes}
                depended_on = {
                    dep
                    for scene in parent_scenes
                    for dep in [*scene.hard_dependencies, *scene.soft_dependencies, *scene.depends_on]
                    if dep in parent_scene_ids
                }
                prerequisite_terminal_scenes.extend(
                    scene.scene_id for scene in parent_scenes if scene.scene_id not in depended_on
                )

            first_scene = act_scenes[0]
            merged_hard = list(dict.fromkeys([*prerequisite_terminal_scenes, *first_scene.hard_dependencies]))
            if merged_hard != first_scene.hard_dependencies:
                act_scenes[0] = first_scene.model_copy(
                    update={
                        "depends_on": list(dict.fromkeys([*first_scene.depends_on, *prerequisite_terminal_scenes])),
                        "hard_dependencies": merged_hard,
                    }
                )

            for scene in act_scenes:
                scene.act_id = scene.act_id or act.act_id
                if scene.act_id not in act_by_id:
                    scene.act_id = act.act_id
                scenes.append(scene)
        return scenes


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
