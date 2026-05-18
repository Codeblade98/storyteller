from typing import Any, Literal

from pydantic import BaseModel, Field
from story_engine.models.scene import ActNode, SceneDAGPlan, ScenePlan, StoryPlan


LLMRole = Literal["planner", "generator", "extractor", "verifier"]
# TODO:
## Define act and scenes as separate schemas and use them as helper contracts in the main planning contract. This will allow us to reuse them in other contexts (e.g. scene generation) and also make the main contract more readable.

## Helper Contracts
class ActContract(BaseModel):
    act_id: str
    act_title: str = ""
    act_summary: str = ""
    expected_scene_count: int = 1
    depends_on: list[str] = Field(default_factory=list)
    
class SceneContract(BaseModel):
    scene_id: str
    scene_title: str = ""
    scene_description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

## LLM Constracts
class ActScenePlanningContract(BaseModel):
    act_id: str
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    

class SceneDraftContract(BaseModel):
    scene_id: str
    scene_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    
class ActPlanningContract(BaseModel):
    acts: list[ActContract] = Field(default_factory=list)    
    
    
class PlanningContract(BaseModel):
    acts: ActPlanningContract = Field(default_factory=ActPlanningContract)
    scene_dags: list[dict[str, Any]] = Field(default_factory=list)
    scenes: list[dict[str, Any]] = Field(default_factory=list)


class EdgeFilterContract(BaseModel):
    required_context: list[str] = Field(default_factory=list)
    ignored_context: list[str] = Field(default_factory=list)
    transforms: dict[str, str] = Field(default_factory=dict)


class DiffExtractionContract(BaseModel):
    state_diff: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    notes: list[str] = Field(default_factory=list)


class SemanticVerificationContract(BaseModel):
    ok: bool
    failures: list[str] = Field(default_factory=list)
    confidence: float = 1.0

class StorySpecContract(BaseModel):
    allowed_conflict: list[str] = Field(default_factory=list)
    forbidden_elements: list[str] = Field(default_factory=list)
    moral_theme: str = "Good wins over evil"
    target_act_count: int = 3

def planning_contract_to_story_plan(contract: PlanningContract) -> StoryPlan:
    acts = [ActNode.model_validate(act) for act in contract.acts]
    scene_dags = []
    for scene_dag in contract.scene_dags:
        scenes = [ScenePlan.model_validate(scene) for scene in scene_dag.get("scenes", [])]
        scene_dags.append(SceneDAGPlan(act_id=scene_dag["act_id"], scenes=scenes))
    return StoryPlan(acts=acts, scene_dags=scene_dags)


def planning_contract_to_scene_plans(contract: PlanningContract) -> list[ScenePlan]:
    return [ScenePlan.model_validate(scene) for scene in contract.scenes]
