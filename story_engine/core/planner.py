from dataclasses import asdict
import logging

from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.models.llm_contracts import PlanningContract, planning_contract_to_scene_plans
from story_engine.models.scene import ScenePlan
from story_engine.models.story_spec import StorySpec
from story_engine.retrieval.patterns import StoryPattern

logger = logging.getLogger("story_engine.engine")


class ActScenePlanner:
    """Generates detailed scene plans from a story specification and pattern.

    Creates ScenePlan objects for each scene with appropriate goals,
    dependencies, and constraints based on the story pattern.
    """
    def __init__(self, llm_runner: JSONTaskRunner | None = None) -> None:
        self.llm_runner = llm_runner

    def plan(self, spec: StorySpec, pattern: StoryPattern) -> list[ScenePlan]:
        """Generate scene plans matching the story specification and pattern.

        Creates sequential plans with appropriate dependencies, goals,
        and context requirements.

        Args:
            spec: Story specification containing constraints and settings.
            pattern: Story pattern defining acts and emotion curve.

        Returns:
            List of ScenePlan objects ready for DAG construction.
        """
        llm_plans = self._llm_plan(spec, pattern)
        if llm_plans is not None:
            logger.info("planner_used_llm scenes=%s", len(llm_plans))
            return llm_plans
        logger.info("planner_used_deterministic_fallback target_scene_count=%s", spec.target_scene_count)
        return self._fallback_plan(spec, pattern)

    def _llm_plan(self, spec: StorySpec, pattern: StoryPattern) -> list[ScenePlan] | None:
        if self.llm_runner is None:
            return None
        if not self.llm_runner.has_client("planner"):
            logger.debug("planner_no_llm_client")
            return None
        contract = self.llm_runner.run_json_task(
            role="planner",
            template_name="planner_prompt.yaml",
            output_model=PlanningContract,
            payload={
                "story_spec": asdict(spec),
                "pattern": asdict(pattern),
                "hard_constraints": {
                    "target_scene_count": spec.target_scene_count,
                    "allowed_state_keys": [
                        "world_state",
                        "character_state",
                        "narrative_state",
                        "style_state",
                        "safety_state",
                    ],
                    "scene_id_format": "S1..SN",
                    "forbidden_elements": spec.forbidden_elements,
                },
                "output_schema": {
                    "scenes": [
                        {
                            "scene_id": "S1",
                            "act": "string",
                            "goal": "string",
                            "emotion_level": 0,
                            "depends_on": [],
                            "hard_dependencies": [],
                            "soft_dependencies": [],
                            "required_context": ["style_state", "safety_state"],
                            "constraints": ["age_group=7-9"],
                        }
                    ]
                },
            },
        )
        if contract is None:
            logger.info("planner_llm_unavailable_or_failed")
            return None
        try:
            return self._normalize_llm_plans(planning_contract_to_scene_plans(contract), spec, pattern)
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("planner_llm_plan_rejected error=%s", exc)
            return None

    def _fallback_plan(self, spec: StorySpec, pattern: StoryPattern) -> list[ScenePlan]:
        plans: list[ScenePlan] = []
        for index in range(spec.target_scene_count):
            scene_number = index + 1
            scene_id = f"S{scene_number}"
            act = pattern.acts[index]
            dependency = [] if index == 0 else [f"S{scene_number - 1}"]
            goal = self._goal_for_scene(spec, act, scene_number, spec.target_scene_count)
            required_context = self._required_context(index, act)

            plans.append(
                ScenePlan(
                    scene_id=scene_id,
                    act=act,
                    goal=goal,
                    emotion_level=pattern.emotion_curve[index],
                    depends_on=dependency,
                    hard_dependencies=dependency,
                    soft_dependencies=[],
                    required_context=required_context,
                    constraints=[
                        f"age_group={spec.age_group}",
                        f"moral_theme={spec.moral_theme}",
                        f"max_fear={spec.fear_level}",
                    ],
                )
            )
        return plans

    def _normalize_llm_plans(self, plans: list[ScenePlan], spec: StorySpec, pattern: StoryPattern) -> list[ScenePlan]:
        if len(plans) != spec.target_scene_count:
            raise ValueError("LLM plan must match target scene count")

        normalized: list[ScenePlan] = []
        known_ids: set[str] = set()
        allowed_context = {"world_state", "character_state", "narrative_state", "style_state", "safety_state"}
        for index, plan in enumerate(plans):
            scene_id = f"S{index + 1}"
            dependencies = [] if index == 0 else [f"S{index}"]
            requested_hard = [dep for dep in plan.hard_dependencies or plan.depends_on if dep in known_ids]
            hard_dependencies = requested_hard or dependencies
            soft_dependencies = [dep for dep in plan.soft_dependencies if dep in known_ids and dep not in hard_dependencies]
            required_context = [key for key in plan.required_context if key in allowed_context]
            if not required_context:
                required_context = self._required_context(index, plan.act or pattern.acts[index])
            known_ids.add(scene_id)
            normalized.append(
                ScenePlan(
                    scene_id=scene_id,
                    act=plan.act or pattern.acts[index],
                    goal=plan.goal or self._goal_for_scene(spec, pattern.acts[index], index + 1, spec.target_scene_count),
                    emotion_level=max(0, min(5, int(plan.emotion_level))),
                    depends_on=hard_dependencies,
                    hard_dependencies=hard_dependencies,
                    soft_dependencies=soft_dependencies,
                    required_context=required_context,
                    constraints=[
                        f"age_group={spec.age_group}",
                        f"moral_theme={spec.moral_theme}",
                        f"max_fear={spec.fear_level}",
                    ],
                )
            )
        return normalized

    def _goal_for_scene(self, spec: StorySpec, act: str, scene_number: int, total: int) -> str:
        """Generate an appropriate goal for a scene given its context.

        Args:
            spec: Story specification for context.
            act: The narrative act for this scene.
            scene_number: Position of scene (1-indexed).
            total: Total number of scenes in the story.

        Returns:
            A goal string for the scene.
        """
        if scene_number == 1:
            return f"Introduce a child-friendly protagonist and the topic: {spec.topic}."
        if scene_number == total:
            return f"Resolve the {spec.topic} story with a gentle lesson about {spec.moral_theme}."
        if act == "discovery":
            return f"Reveal a safe, interesting discovery connected to {spec.topic}."
        if act == "challenge":
            return "Create a mild problem that requires thinking and cooperation."
        if act == "cooperation":
            return f"Show characters using {spec.moral_theme} to solve the problem."
        return f"Advance the {act} act while preserving continuity."

    def _required_context(self, index: int, act: str) -> list[str]:
        """Determine which state keys are required for a scene.

        Args:
            index: Scene position (0-indexed).
            act: The narrative act for this scene.

        Returns:
            List of state keys required for generation.
        """
        if index == 0:
            return ["style_state", "safety_state"]
        if act in {"challenge", "cooperation", "resolution"}:
            return ["world_state", "character_state", "narrative_state", "style_state", "safety_state"]
        return ["world_state", "character_state", "style_state", "safety_state"]
