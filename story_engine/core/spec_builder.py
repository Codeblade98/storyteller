from story_engine.models.story_spec import StoryInput, StorySpec
from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.models.llm_contracts import StorySpecContract
from dataclasses import asdict

import logging

logger = logging.getLogger("story_engine.engine")

class StorySpecBuilder:
    """Converts high-level story input into detailed generation specifications.

    Determines narrative parameters like vocabulary, sentence complexity,
    and moral themes based on age group and story preferences.
    """
    def __init__(self, llm_runner: JSONTaskRunner | None = None) -> None:
        self.llm_runner = llm_runner
        
    def build(self, story_input: StoryInput) -> StorySpec:
        """Build a detailed story specification from user input.

        Determines vocabulary level, sentence complexity, scene count,
        and moral theme based on age group and input parameters.

        Args:
            story_input: High-level user input for story generation.

        Returns:
            A StorySpec ready for planning and generation.
        """
        vocab_level, sentence_complexity = self._age_controls(story_input.age_group)
        
        llm_inferred_spec = self._llm_infer_specs(story_input)
        if llm_inferred_spec is not None:
            allowed_conflict = llm_inferred_spec.allowed_conflict
            forbidden_elements = llm_inferred_spec.forbidden_elements
            moral_theme = llm_inferred_spec.moral_theme
            target_act_count = llm_inferred_spec.target_act_count
        else:
            allowed_conflict = []
            forbidden_elements = ["death", "gore", "explicit violence", "cruel punishment"]
            moral_theme = self._infer_theme(story_input.topic)
            target_act_count = 5 if story_input.length == "short" else 10

        return StorySpec(
            topic=story_input.topic,
            age_group=story_input.age_group,
            genre=story_input.genre,
            fear_level=story_input.fear_level,
            length=story_input.length,
            vocab_level=vocab_level,
            sentence_complexity=sentence_complexity,
            allowed_conflict=allowed_conflict,
            forbidden_elements=forbidden_elements,
            moral_theme=moral_theme,
            target_act_count=target_act_count,
        )

    def _age_controls(self, age_group: str) -> tuple[str, int]:
        """Determine vocabulary and sentence complexity for an age group.

        Args:
            age_group: Target age group.

        Returns:
            Tuple of (vocab_level, sentence_complexity).
        """
        if age_group == "3-5":
            return "early", 1
        if age_group in {"6-8", "7-9"}:
            return "simple", 2
        return "clear", 3

    def _infer_theme(self, topic: str) -> str:
        """Infer an appropriate moral theme from the story topic.

        Args:
            topic: The story topic.

        Returns:
            A moral theme string appropriate for the topic.
        """
        lowered = topic.lower()
        if "friend" in lowered:
            return "friendship"
        if "brave" in lowered or "fear" in lowered:
            return "courage"
        if "share" in lowered:
            return "generosity"
        return "kindness"

    def _llm_infer_specs(self, story_input: StoryInput) -> StorySpecContract | None:
        """Use an LLM to infer detailed story specifications from high-level input.

        Args:
            story_input: High-level user input for story generation.

        Returns:
            A StorySpecContract with inferred parameters, or None if inference fails.
        """
        try:
            contract = StorySpecContract()
            response = self.llm_runner.run_json_task(
                role="spec_builder",
                template_name="spec_builder_prompt.yaml",
                output_model=StorySpecContract,
                payload={"story_input": asdict(story_input)},
            )
            return StorySpecContract.model_validate(response)
        except Exception as e:
            logger.warning("LLM spec inference failed: %s", e)
            return None