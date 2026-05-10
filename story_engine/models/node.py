from dataclasses import dataclass, field

from story_engine.models.edge import EdgeContext
from story_engine.models.base import ModelMixin
from story_engine.models.scene import ScenePlan


@dataclass
class SceneNode(ModelMixin):
    """Represents a scene node in the story dependency graph.

    Attributes:
        scene_id: Unique identifier for the scene.
        plan: The ScenePlan containing the scene's goals and constraints.
        incoming_edges: List of edges pointing to this node from dependencies.
        outgoing_edges: List of edges pointing from this node to dependents.
    """
    scene_id: str
    plan: ScenePlan
    incoming_edges: list[EdgeContext] = field(default_factory=list)
    outgoing_edges: list[EdgeContext] = field(default_factory=list)
