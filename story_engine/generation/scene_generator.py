from typing import Any
import logging

from story_engine.generation.prompt_builder import PromptBuilder
from story_engine.llm.client import LLMClient
from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.llm.parser import JSONParser
from story_engine.llm.retry import RetryPolicy
from story_engine.models.llm_contracts import SceneDraftContract
from story_engine.models.node import SceneNode
from story_engine.models.scene import SceneOutput
from story_engine.models.story_spec import StorySpec

logger = logging.getLogger("story_engine.engine")


class SceneGenerator:
    """Generates individual scenes using LLM or deterministic methods.

    Can use either a connected LLM client or fallback to deterministic
    generation for testing and development.
    """
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        llm_runner: JSONTaskRunner | None = None,
        prompt_builder: PromptBuilder | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Initialize the scene generator.

        Args:
            llm_client: Optional LLM client for generation. If None, uses deterministic fallback.
            prompt_builder: Optional custom prompt builder. Uses default if None.
            retry_policy: Optional retry policy for LLM calls. Uses default if None.
        """
        self.llm_client = llm_client
        self.llm_runner = llm_runner
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.retry_policy = retry_policy or RetryPolicy()
        self.parser = JSONParser()

    def generate(
        self,
        node: SceneNode,
        spec: StorySpec,
        filtered_state: dict[str, Any],
        repair_instructions: list[str] | None = None,
        attempt: int = 0,
    ) -> SceneOutput:
        """Generate a scene given planning context and state.

        Uses LLM if available, otherwise uses deterministic fallback generation.

        Args:
            node: Scene node defining the scene's goals.
            spec: Story specification for constraints.
            filtered_state: State relevant to this scene.
            repair_instructions: Optional list of fixes from verification failures.
            attempt: Generation attempt number (for temperature adjustment).

        Returns:
            A SceneOutput with generated text and state changes.
        """
        if self.llm_client:
            logger.info("scene_generator_using_direct_llm scene_id=%s attempt=%s", node.scene_id, attempt)
            prompt = self.prompt_builder.build_scene_prompt(node, spec, filtered_state, repair_instructions)
            raw = self.llm_client.complete_json(
                prompt,
                temperature=self.retry_policy.temperature_for(attempt),
            )
            draft = self.parser.parse_model(raw, SceneDraftContract)
            return SceneOutput(scene_id=draft.scene_id, scene_text=draft.scene_text, metadata=draft.metadata)
        if self.llm_runner and self.llm_runner.has_client("generator"):
            logger.info("scene_generator_using_role_llm scene_id=%s attempt=%s", node.scene_id, attempt)
            draft = self.llm_runner.run_json_task(
                role="generator",
                template_name="scene_generation.yaml",
                output_model=SceneDraftContract,
                payload={
                    "scene_id": node.scene_id,
                    "scene_goal": node.plan.goal,
                    "act": node.plan.act,
                    "constraints": node.plan.constraints,
                    "filtered_state": filtered_state,
                    "forbidden_elements": spec.forbidden_elements,
                    "repair_instructions": repair_instructions or [],
                    "output_schema": {
                        "scene_id": "string",
                        "scene_text": "string",
                        "metadata": {"fear_score": "integer", "violence_score": "integer"},
                    },
                },
            )
            if draft is not None:
                return SceneOutput(scene_id=draft.scene_id, scene_text=draft.scene_text, metadata=draft.metadata)
            logger.warning("scene_generator_llm_failed_using_fallback scene_id=%s attempt=%s", node.scene_id, attempt)
        elif self.llm_runner:
            logger.debug("scene_generator_no_llm_client scene_id=%s attempt=%s", node.scene_id, attempt)
        logger.info("scene_generator_using_deterministic_fallback scene_id=%s attempt=%s", node.scene_id, attempt)
        return self._deterministic_scene(node, spec, filtered_state, repair_instructions)

    def _deterministic_scene(
        self,
        node: SceneNode,
        spec: StorySpec,
        filtered_state: dict[str, Any],
        repair_instructions: list[str] | None,
    ) -> SceneOutput:
        """Generate a scene deterministically without LLM calls.

        Used for testing and when LLM is unavailable.

        Args:
            node: Scene node defining the scene.
            spec: Story specification.
            filtered_state: Relevant state for generation.
            repair_instructions: Optional repair hints to incorporate.

        Returns:
            A deterministically generated SceneOutput.
        """
        protagonist = self._protagonist(filtered_state)
        topic_noun = spec.topic.split()[0].lower()
        location = filtered_state.get("world_state", {}).get("location", "a bright path")
        if node.scene_id == "S1":
            text = (
                f"{protagonist} loved asking kind questions. "
                f"One morning, {protagonist} found a clue about {spec.topic} near {location}."
            )
        elif node.plan.act == "resolution":
            text = (
                f"By the end, {protagonist} understood that {spec.moral_theme} could make even a strange day feel safe. "
                f"The {topic_noun} mystery became a happy story to tell at home."
            )
        else:
            text = (
                f"{protagonist} followed the clue with careful steps. "
                f"When a mild problem appeared, {protagonist} chose kindness, listened closely, and helped everyone move forward."
            )

        if repair_instructions:
            text = text.replace("strange", "surprising")

        return SceneOutput(
            scene_id=node.scene_id,
            scene_text=text,
            state_diff={},
            metadata={
                "fear_score": min(spec.fear_level, node.plan.emotion_level),
                "violence_score": 0,
                "used_context_keys": sorted(filtered_state),
            },
        )

    def _protagonist(self, filtered_state: dict[str, Any]) -> str:
        """Determine the main character for the scene.

        Args:
            filtered_state: State containing character information.

        Returns:
            Name of the protagonist character.
        """
        characters = filtered_state.get("character_state", {})
        if characters:
            return sorted(characters)[0]
        return "Milo"
