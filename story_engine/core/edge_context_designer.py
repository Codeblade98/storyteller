from dataclasses import asdict
import logging

from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.models.edge import EdgeContext
from story_engine.models.llm_contracts import EdgeFilterContract
from story_engine.models.scene import ScenePlan

logger = logging.getLogger("story_engine.engine")


class EdgeContextDesigner:
    allowed_context = {"world_state", "character_state", "narrative_state", "style_state", "safety_state"}
    never_context = {"full_scene_history", "minor_environment_details"}

    def __init__(self, llm_runner: JSONTaskRunner | None = None) -> None:
        self.llm_runner = llm_runner

    def design(self, source_plan: ScenePlan, target_plan: ScenePlan, dependency_type: str) -> EdgeContext:
        proposed = self._llm_design(source_plan, target_plan, dependency_type)
        if proposed is None:
            logger.debug(
                "edge_context_used_fallback source=%s target=%s dependency_type=%s",
                source_plan.scene_id,
                target_plan.scene_id,
                dependency_type,
            )
            proposed = EdgeFilterContract(required_context=target_plan.required_context, ignored_context=list(self.never_context))
        else:
            logger.info(
                "edge_context_used_llm source=%s target=%s dependency_type=%s",
                source_plan.scene_id,
                target_plan.scene_id,
                dependency_type,
            )
        return EdgeContext(
            source=source_plan.scene_id,
            target=target_plan.scene_id,
            dependency_type=dependency_type,
            required_context=self._valid_context(proposed.required_context),
            ignored_context=self._valid_ignored(proposed.ignored_context),
            transforms={key: value for key, value in proposed.transforms.items() if key in self.allowed_context},
        )

    def _llm_design(
        self,
        source_plan: ScenePlan,
        target_plan: ScenePlan,
        dependency_type: str,
    ) -> EdgeFilterContract | None:
        if self.llm_runner is None:
            return None
        if not self.llm_runner.has_client("planner"):
            return None
        contract = self.llm_runner.run_json_task(
            role="planner",
            template_name="edge_context_prompt.yaml",
            output_model=EdgeFilterContract,
            payload={
                "source_scene": asdict(source_plan),
                "target_scene": asdict(target_plan),
                "dependency_type": dependency_type,
                "allowed_context": sorted(self.allowed_context),
                "never_context": sorted(self.never_context),
                "output_schema": {
                    "required_context": ["character_state"],
                    "ignored_context": ["full_scene_history"],
                    "transforms": {"character_state": "only named characters relevant to target goal"},
                },
            },
        )
        return contract

    def _valid_context(self, keys: list[str]) -> list[str]:
        return [key for key in keys if key in self.allowed_context]

    def _valid_ignored(self, keys: list[str]) -> list[str]:
        return sorted({key for key in keys if key in self.never_context or key not in self.allowed_context})
