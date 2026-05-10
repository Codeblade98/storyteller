from dataclasses import asdict
from typing import Any
import logging

from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.models.llm_contracts import DiffExtractionContract
from story_engine.models.node import SceneNode
from story_engine.models.scene import SceneOutput
from story_engine.models.story_spec import StorySpec

logger = logging.getLogger("story_engine.engine")


class StateDiffExtractor:
    def __init__(self, llm_runner: JSONTaskRunner | None = None) -> None:
        self.llm_runner = llm_runner

    def extract(
        self,
        *,
        draft: SceneOutput,
        node: SceneNode,
        spec: StorySpec,
        filtered_state: dict[str, Any],
    ) -> dict[str, Any]:
        llm_diff = self._llm_extract(draft, node, spec, filtered_state)
        if llm_diff is not None:
            logger.info("state_diff_extractor_used_llm scene_id=%s diff_keys=%s", node.scene_id, sorted(llm_diff))
            return llm_diff
        logger.info("state_diff_extractor_used_deterministic_fallback scene_id=%s", node.scene_id)
        return self._deterministic_extract(node, spec, filtered_state)

    def _llm_extract(
        self,
        draft: SceneOutput,
        node: SceneNode,
        spec: StorySpec,
        filtered_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.llm_runner is None:
            return None
        if not self.llm_runner.has_client("extractor"):
            logger.debug("state_diff_extractor_no_llm_client scene_id=%s", node.scene_id)
            return None
        contract = self.llm_runner.run_json_task(
            role="extractor",
            template_name="diff_extraction_prompt.yaml",
            output_model=DiffExtractionContract,
            payload={
                "scene": asdict(draft),
                "scene_plan": asdict(node.plan),
                "story_spec": asdict(spec),
                "filtered_state": filtered_state,
                "allowed_roots": ["world_state", "character_state", "narrative_state", "style_state", "safety_state"],
                "allowed_aliases": ["world", "character", "narrative", "style", "safety", "time.advance"],
                "output_schema": {
                    "state_diff": {"character.Milo.emotion": "curious"},
                    "confidence": 0.0,
                    "notes": ["short reason"],
                },
            },
        )
        if contract is None or contract.confidence < 0.35:
            logger.warning(
                "state_diff_extractor_llm_rejected scene_id=%s confidence=%s",
                node.scene_id,
                None if contract is None else contract.confidence,
            )
            return None
        return contract.state_diff

    def _deterministic_extract(
        self,
        node: SceneNode,
        spec: StorySpec,
        filtered_state: dict[str, Any],
    ) -> dict[str, Any]:
        protagonist = self._protagonist(filtered_state)
        if node.scene_id == "S1":
            return {
                f"character.{protagonist}.emotion": "curious",
                f"character.{protagonist}.inventory.add": ["small lantern"],
                "world.location": "sunlit forest",
                "narrative.active_arc": spec.topic.replace(" ", "_"),
            }
        if node.plan.act == "resolution":
            return {
                f"character.{protagonist}.emotion": "proud",
                "narrative.active_arc": "resolved",
                "time.advance": "evening",
            }
        return {
            f"character.{protagonist}.emotion": "braver",
            "world.location": "forest clearing",
            "time.advance": "a little later",
        }

    def _protagonist(self, filtered_state: dict[str, Any]) -> str:
        characters = filtered_state.get("character_state", {})
        if characters:
            return sorted(characters)[0]
        return "Milo"
