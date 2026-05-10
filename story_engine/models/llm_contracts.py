from dataclasses import dataclass, field
from typing import Any, Literal

from story_engine.models.base import ModelMixin
from story_engine.models.scene import ScenePlan


LLMRole = Literal["planner", "generator", "extractor", "verifier"]


@dataclass
class PlanningContract(ModelMixin):
    scenes: list[dict[str, Any]]


@dataclass
class EdgeFilterContract(ModelMixin):
    required_context: list[str] = field(default_factory=list)
    ignored_context: list[str] = field(default_factory=list)
    transforms: dict[str, str] = field(default_factory=dict)


@dataclass
class SceneDraftContract(ModelMixin):
    scene_id: str
    scene_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffExtractionContract(ModelMixin):
    state_diff: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)


@dataclass
class SemanticVerificationContract(ModelMixin):
    ok: bool
    failures: list[str] = field(default_factory=list)
    confidence: float = 1.0


def planning_contract_to_scene_plans(contract: PlanningContract) -> list[ScenePlan]:
    return [ScenePlan.model_validate(scene) for scene in contract.scenes]
