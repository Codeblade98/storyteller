from pathlib import Path
from typing import Any

from story_engine.llm.json_runner import RenderedPrompt, _load_prompt_template, render_payload_markdown
from story_engine.models.node import SceneNode
from story_engine.models.story_spec import StorySpec


class PromptBuilder:
    """Constructs LLM prompts for scene generation from template and context.

    Loads a YAML template and populates it with scene-specific parameters.
    """
    def __init__(self, template_dir: Path | None = None) -> None:
        """Initialize the prompt builder.

        Args:
            template_dir: Directory containing prompt templates. Uses default if None.
        """
        self.template_dir = template_dir or Path(__file__).parents[1] / "prompts"

    def build_scene_prompt(
        self,
        node: SceneNode,
        spec: StorySpec,
        filtered_state: dict[str, Any],
        repair_instructions: list[str] | None = None,
    ) -> RenderedPrompt:
        """Build an LLM prompt for scene generation.

        Args:
            node: Scene node with generation goals.
            spec: Story specification with constraints.
            filtered_state: Relevant narrative state.
            repair_instructions: Optional repair hints from verification.

        Returns:
            Formatted system prompt and markdown user payload ready for LLM.
        """
        template = _load_prompt_template(self.template_dir / "scene_generation.yaml")
        payload = {
            "scene_id": node.scene_id,
            "scene_goal": node.plan.goal,
            "act": node.plan.act,
            "constraints": node.plan.constraints,
            "filtered_state": filtered_state,
            "forbidden_elements": spec.forbidden_elements,
            "output_schema": {
                "scene_id": "string",
                "scene_text": "string",
                "metadata": {"fear_score": "integer", "violence_score": "integer"},
            },
            "repair_instructions": repair_instructions or [],
        }
        payload_markdown = render_payload_markdown(payload)
        user_payload = template.user_payload.replace("{{payload_markdown}}", payload_markdown)
        user_payload = user_payload.replace("{{payload_json}}", payload_markdown)
        return RenderedPrompt(system_prompt=template.system_prompt, user_payload=user_payload)
