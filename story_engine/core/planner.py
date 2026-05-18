from dataclasses import asdict
import logging

from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.models.llm_contracts import ActPlanningContract, ActScenePlanningContract
from story_engine.models.scene import ActNode, GlobalBlackboard, SceneDAGPlan, ScenePlan, StoryPlan
from story_engine.models.story_spec import StorySpec
from story_engine.retrieval.patterns import StoryPattern

logger = logging.getLogger("story_engine.engine")


class ActScenePlanner:
    """Generates detailed scene plans from a story specification and pattern.

    Creates ScenePlan objects for each scene with appropriate goals,
    dependencies, and constraints based on the story pattern.
    """
    def __init__(self, llm_runner: JSONTaskRunner | None = None) -> None:
        """Initialize the ActScenePlanner.

        Args:
            llm_runner: Optional `JSONTaskRunner` used to call LLM planning tasks.
        """
        self.llm_runner = llm_runner

    def plan(self, spec: StorySpec, pattern: StoryPattern) -> StoryPlan:
        """Generate a hierarchical act DAG with per-act scene DAG expansions.

        Args:
            spec: Story specification containing constraints and settings.
            pattern: Story pattern defining acts and emotion curve.

        Returns:
            A nested StoryPlan containing act and scene DAGs.
        """
        llm_plan = self._llm_plan(spec, pattern)
        if llm_plan is not None:
            logger.info("planner_used_llm acts=%s scenes=%s", len(llm_plan.acts), len(llm_plan.flatten_scenes()))
            return llm_plan
        logger.info("planner_used_deterministic_fallback target_scene_count=%s", spec.target_scene_count)
        return self._fallback_plan(spec, pattern)

    def _llm_plan(self, spec: StorySpec, pattern: StoryPattern) -> StoryPlan | None:
        """Attempt to generate a `StoryPlan` by querying the LLM planner.

        Returns the generated `StoryPlan` when successful, or `None` when the
        LLM runner is unavailable or the response cannot be normalized.
        """
        if self.llm_runner is None:
            return None
        if not self.llm_runner.has_client("planner"):
            logger.debug("planner_no_llm_client")
            return None
        act_count = min(3, spec.target_scene_count)
        act_contract = self.llm_runner.run_json_task(
            role="planner",
            template_name="planner_acts_prompt.yaml",
            output_model=ActPlanningContract,
            payload={
                "story_spec": asdict(spec),
                "pattern": asdict(pattern),
                "hard_constraints": {
                    "target_scene_count": spec.target_scene_count,
                    "target_act_count": act_count,
                    "act_id_format": "A1..AM",
                    "allowed_conflict": spec.allowed_conflict,
                    "forbidden_elements": spec.forbidden_elements,
                },
                "output_schema": {
                    "acts": [
                        {
                            "act_id": "apprpiate 1st act ID as string, denoting act number and used for referencing in scene dependencies (e.g. A1, A2, A3)",
                            "act_title": "appropriate 1st act title as string",
                            "act_summary": "a brief summary of the 1st act's narrative purpose and key events",
                            "depends_on": ["list of act IDs that this act depends on, if any, ensuring a valid DAG structure", "must only contain previous acts"],
                        },
                        {
                            "act_id": "apprpiate 2nd act ID as string, denoting act number and used for referencing in scene dependencies (e.g. A1, A2, A3)",
                            "act_title": "appropriate 2nd act title as string",
                            "act_summary": "a brief summary of the 2nd act's narrative purpose and key events",
                            "depends_on": ["list of act IDs that this act depends on, if any, ensuring a valid DAG structure", "must only contain previous acts"],
                        }
                    ],
                },
            },
        )
        if act_contract is None:
            logger.info("planner_act_llm_unavailable_or_failed")
            return None

        try:
            acts = self._normalize_acts(act_contract, spec, pattern, act_count) # this returns an acts DAG
            scene_dags = []
            next_scene_number = 1
            for act in acts:
                scene_count = act.expected_scene_count
                scene_dag = self._llm_scene_dag(
                    spec=spec,
                    pattern=pattern,
                    acts=acts,
                    act=act,
                    scene_count=scene_count,
                    first_scene_number=next_scene_number,
                )
                if scene_dag is None:
                    logger.info("planner_scene_llm_failed_using_fallback act_id=%s", act.act_id)
                    scene_dag = self._fallback_scene_dag(
                        spec=spec,
                        pattern=pattern,
                        act=act,
                        scene_count=scene_count,
                        first_scene_number=next_scene_number,
                    )
                scene_dags.append(scene_dag)
                next_scene_number += scene_count
            return self._validate_story_plan(StoryPlan(acts=acts, scene_dags=scene_dags))
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("planner_llm_plan_rejected error=%s", exc)
            return None

    def _llm_scene_dag(
        self,
        *,
        spec: StorySpec,
        pattern: StoryPattern,
        acts: list[ActNode],
        act: ActNode,
        scene_count: int,
        first_scene_number: int,
    ) -> SceneDAGPlan | None:
        """Request an LLM-generated scene DAG for a single act.

        Args:
            spec: StorySpec for context.
            pattern: StoryPattern with act names and emotion curve.
            acts: All act nodes in the plan (used for cross-reference).
            act: The specific `ActNode` to plan scenes for.
            scene_count: Number of scenes to produce for this act.
            first_scene_number: Global index of the first scene in this act.

        Returns:
            A `SceneDAGPlan` when the LLM produces a valid plan, otherwise
            `None` when no LLM client is available or normalization fails.
        """
        if self.llm_runner is None or not self.llm_runner.has_client("planner"):
            return None

        last_scene_number = first_scene_number + scene_count - 1
        contract = self.llm_runner.run_json_task(
            role="planner",
            template_name="planner_scenes_prompt.yaml",
            output_model=ActScenePlanningContract,
            payload={
                "story_spec": asdict(spec),
                "pattern": asdict(pattern),
                "act": asdict(act),
                "all_acts": [asdict(item) for item in acts],
                "hard_constraints": {
                    "act_id": act.act_id,
                    "scene_count": scene_count,
                    "scene_id_format": f"S{first_scene_number}..S{last_scene_number}",
                    "dependencies_scope": "Only reference scene IDs inside this act. Act-level dependencies are handled separately.",
                    "allowed_state_keys": [
                        "world_state",
                        "character_state",
                        "narrative_state",
                        "style_state",
                        "safety_state",
                    ],
                    "allowed_conflict": spec.allowed_conflict,
                    "forbidden_elements": spec.forbidden_elements,
                    "last_scene_state_update": {"story_phase": act.act_id},
                },
                "output_schema": {
                    "act_id": act.act_id,
                    "scenes": [
                        {
                            "scene_id": f"S{first_scene_number}",
                            "act_id": act.act_id,
                            "act": "setup (1st scene)",
                            "goal": "string for 1st scene",
                            "emotion_level": 0,
                            "depends_on": [],
                            "hard_dependencies": [],
                            "soft_dependencies": [],
                            "required_context": ["style_state", "safety_state"],
                            "constraints": [
                                "allowed_conflict=[]",
                                "forbidden_elements=[]",
                            ],
                            "state_updates": {},
                        },
                        {
                            "scene_id": f"S{first_scene_number + 1}",
                            "act_id": act.act_id,
                            "act": "development (2nd scene)",
                            "goal": "string for 2nd scene",
                            "emotion_level": 1,
                            "depends_on": [f"S{first_scene_number}"],
                            "hard_dependencies": [f"S{first_scene_number}"],
                            "soft_dependencies": [],
                            "required_context": ["world_state", "character_state", "style_state", "safety_state"],
                            "constraints": [
                                "allowed_conflict=[]",
                                "forbidden_elements=[]",
                            ],
                            "state_updates": {},
                        }
                    ],
                },
            },
        )
        if contract is None:
            return None
        try:
            return self._normalize_scene_dag(
                contract=contract,
                spec=spec,
                pattern=pattern,
                act=act,
                scene_count=scene_count,
                first_scene_number=first_scene_number,
            )
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("planner_scene_llm_plan_rejected act_id=%s error=%s", act.act_id, exc)
            return None

    def _fallback_plan(self, spec: StorySpec, pattern: StoryPattern) -> StoryPlan:
        """Create a deterministic fallback `StoryPlan` without using the LLM.

        The fallback divides the story into up to three acts and generates
        act summaries and per-act scene DAGs deterministically.
        """
        act_count = min(3, spec.target_scene_count)
        act_ids = [f"A{index + 1}" for index in range(act_count)]
        acts = [
            ActNode(
                act_id=act_id,
                act_number=index + 1,
                act_summary=self._act_summary(spec, pattern, index, act_count),
                depends_on=[] if index == 0 else [act_ids[index - 1]],
                expected_entrance_state={} if index == 0 else {"story_phase": act_ids[index - 1]},
                expected_exit_state={"story_phase": act_id},
            )
            for index, act_id in enumerate(act_ids)
        ]

        scene_dags: list[SceneDAGPlan] = []
        scene_number = 1
        for act, scene_total in zip(acts, self._scene_counts(spec.target_scene_count, act_count)):
            scene_dags.append(
                self._fallback_scene_dag(
                    spec=spec,
                    pattern=pattern,
                    act=act,
                    scene_count=scene_total,
                    first_scene_number=scene_number,
                )
            )
            scene_number += scene_total

        return self._validate_story_plan(StoryPlan(acts=acts, scene_dags=scene_dags))

    def _normalize_acts(
        self,
        contract: ActPlanningContract,
        spec: StorySpec,
        pattern: StoryPattern,
        act_count: int,
    ) -> list[ActNode]:
        """Normalize an `ActPlanningContract` into a list of `ActNode`.

        This fills missing fields, enforces unique act IDs and valid
        dependencies, and falls back to default summaries when needed.

        Normalization and dependency logic:
        - For each act, only allows dependencies on acts that have already been processed (previous acts).
        - If the LLM's requested dependencies are invalid or missing, defaults to a linear chain (each act depends on the previous one).
        - Ensures a valid, acyclic act DAG.
        """
        acts: list[ActNode] = []
        for index in range(act_count):
            # Get raw act data from contract, or empty dict if missing
            raw = contract.acts[index] if index < len(contract.acts) else {}
            act_id = f"A{index + 1}"
            previous_ids = {act.act_id for act in acts}
            
            # Only allow dependencies on previous acts
            requested_dependencies = raw.get("depends_on", []) if isinstance(raw, dict) else []
            if not isinstance(requested_dependencies, list):
                requested_dependencies = []
            dependencies = [dep for dep in requested_dependencies if dep in previous_ids]
            
            ## NOTE: An act can be independent (no dependencies) or depend on any subset of previous acts.
             
            summary = raw.get("act_summary") if isinstance(raw, dict) else ""
            acts.append(
                ActNode(
                    act_id=act_id,
                    act_number=index + 1,
                    act_summary=summary or self._act_summary(spec, pattern, index, act_count),
                    depends_on=dependencies,
                    expected_scene_count=raw.get("expected_scene_count", 1) if isinstance(raw, dict) else 1,
                    expected_entrance_state={},
                    expected_exit_state={"story_phase": act_id},
                )
            )
        # Validate uniqueness and acyclicity of the act DAG
        return self._validate_act_dag(acts)

    def _normalize_scene_dag(
        self,
        *,
        contract: ActScenePlanningContract,
        spec: StorySpec,
        pattern: StoryPattern,
        act: ActNode,
        scene_count: int,
        first_scene_number: int,
    ) -> SceneDAGPlan:
        """Normalize an `ActScenePlanningContract` into a `SceneDAGPlan`.

        Converts raw scene entries to validated `ScenePlan` objects, assigns
        global scene IDs, filters allowed context keys, and ensures
        dependencies and state updates are sensible for the act.

        Normalization and dependency logic:
        - For each scene, only allows dependencies on scenes that have already been processed (previous scenes in this act).
        - If the LLM's requested hard dependencies are invalid or missing, defaults to a linear chain (each scene depends on the previous one, except the first).
        - Soft dependencies are filtered to only include valid, non-hard dependencies.
        - Ensures a valid, acyclic scene DAG within the act.
        """
        raw_scenes = list(contract.scenes[:scene_count])
        if len(raw_scenes) < scene_count:
            raise ValueError(f"Act {act.act_id} LLM scene plan must contain {scene_count} scenes")
        allowed_context = {"world_state", "character_state", "narrative_state", "style_state", "safety_state"}
        known_ids: set[str] = set()
        normalized_scenes: list[ScenePlan] = []
        for local_index, raw_scene in enumerate(raw_scenes):
            # Validate and parse the raw scene
            scene = ScenePlan.model_validate(raw_scene)
            global_index = first_scene_number + local_index
            scene_id = f"S{global_index}"
            
            # Only allow dependencies on previous scenes in this act
            requested_hard = [dep for dep in scene.hard_dependencies or scene.depends_on if dep in known_ids]
            hard_dependencies = requested_hard
            
            # Soft dependencies: valid, not already hard dependencies
            soft_dependencies = [dep for dep in scene.soft_dependencies if dep in known_ids and dep not in hard_dependencies]
            
            # Filter required context to allowed keys
            required_context = [key for key in scene.required_context if key in allowed_context]
            act_name = scene.act or pattern.acts[min(global_index - 1, len(pattern.acts) - 1)]
            
            # If required context is empty, use default for this act/scene
            if not required_context:
                required_context = self._required_context(global_index - 1, act_name)
            state_updates = dict(scene.state_updates)
            
            # Last scene in act: update story_phase
            if local_index == scene_count - 1:
                state_updates["story_phase"] = act.act_id
            known_ids.add(scene_id)
            normalized_scenes.append(
                ScenePlan(
                    scene_id=scene_id,
                    act_id=act.act_id,
                    act=act_name,
                    goal=scene.goal
                    or self._goal_for_scene(spec, act_name, global_index, spec.target_scene_count),
                    emotion_level=max(0, min(5, int(scene.emotion_level))),
                    depends_on=hard_dependencies,
                    hard_dependencies=hard_dependencies,
                    soft_dependencies=soft_dependencies,
                    required_context=required_context,
                    constraints=self._constraints(spec),
                    state_updates=state_updates,
                )
            )
            
        # Return the normalized scene DAG for this act
        return SceneDAGPlan(act_id=act.act_id, scenes=normalized_scenes)

    def _fallback_scene_dag(
        self,
        *,
        spec: StorySpec,
        pattern: StoryPattern,
        act: ActNode,
        scene_count: int,
        first_scene_number: int,
    ) -> SceneDAGPlan:
        """Build a deterministic scene DAG for an act when LLM output is
        unavailable.

        Fills scene goals, emotion levels, dependencies, and required
        context using heuristics derived from `pattern` and `spec`.
        """
        scenes: list[ScenePlan] = []
        for local_index in range(scene_count):
            scene_number = first_scene_number + local_index
            global_index = scene_number - 1
            scene_id = f"S{scene_number}"
            act_name = pattern.acts[min(global_index, len(pattern.acts) - 1)]
            dependency = [] if local_index == 0 else [f"S{scene_number - 1}"]
            is_last_scene_in_act = local_index == scene_count - 1
            state_updates = {"story_phase": act.act_id} if is_last_scene_in_act else {}
            scenes.append(
                ScenePlan(
                    scene_id=scene_id,
                    act_id=act.act_id,
                    act=act_name,
                    goal=self._goal_for_scene(spec, act_name, scene_number, spec.target_scene_count),
                    emotion_level=pattern.emotion_curve[min(global_index, len(pattern.emotion_curve) - 1)],
                    depends_on=dependency,
                    hard_dependencies=dependency,
                    soft_dependencies=[],
                    required_context=self._required_context(global_index, act_name),
                    constraints=self._constraints(spec),
                    state_updates=state_updates,
                )
            )
        return SceneDAGPlan(act_id=act.act_id, scenes=scenes)

    # def _scene_counts(self, total_scenes: int, act_count: int) -> list[int]:
    #     """Distribute `total_scenes` across `act_count` acts as evenly as
    #     possible.

    #     Returns a list of integers with length `act_count` summing to
    #     `total_scenes`.
    #     """
    #     return [
    #         total_scenes // act_count + (1 if index < total_scenes % act_count else 0)
    #         for index in range(act_count)
    #     ]

    def _validate_story_plan(self, plan: StoryPlan) -> StoryPlan:
        """Validate the consistency of a `StoryPlan`.

        Checks that every act has a single scene DAG, each scene DAG
        references existing acts, all DAGs are internally consistent, and
        that state continuity holds across acts and scenes. Returns the
        validated plan or raises `ValueError` on validation failure.
        """
        self._validate_act_dag(plan.acts)
        act_ids = {act.act_id for act in plan.acts}
        if {dag.act_id for dag in plan.scene_dags} != act_ids:
            raise ValueError("Every act must have exactly one scene DAG")

        for dag in plan.scene_dags:
            if dag.act_id not in act_ids:
                raise ValueError(f"Scene DAG references unknown act {dag.act_id}")
            self._validate_scene_dag(dag)

        self._validate_state_continuity(plan)
        return plan

    def _validate_act_dag(self, acts: list[ActNode]) -> list[ActNode]:
        """Validate an act DAG for uniqueness and well-formed dependencies.

        Ensures act IDs are unique, dependencies reference previous acts
        and are not self-referential. Returns a topologically ordered
        list of acts or raises `ValueError` on failure.
        """
        act_by_id = {act.act_id: act for act in acts}
        if len(act_by_id) != len(acts):
            raise ValueError("Act IDs must be unique")
        for act in acts:
            for dependency in act.depends_on:
                if dependency not in act_by_id:
                    raise ValueError(f"Act {act.act_id} depends on unknown act {dependency}")
                if dependency == act.act_id:
                    raise ValueError(f"Act {act.act_id} cannot depend on itself")
        return self._topological_order(acts, lambda act: act.act_id, lambda act: act.depends_on, "Act")

    def _validate_scene_dag(self, dag: SceneDAGPlan) -> list[ScenePlan]:
        """Validate a scene DAG for uniqueness and dependency correctness.

        Ensures scene IDs are unique within the act, that each scene's
        declared act matches the DAG's act, and that dependencies reference
        existing scenes and are not self-referential. Returns a
        topologically ordered list of scenes or raises `ValueError` on
        failure.
        """
        scene_by_id = {scene.scene_id: scene for scene in dag.scenes}
        if len(scene_by_id) != len(dag.scenes):
            raise ValueError(f"Scene IDs must be unique in act {dag.act_id}")
        for scene in dag.scenes:
            if scene.act_id != dag.act_id:
                raise ValueError(f"Scene {scene.scene_id} is in act {scene.act_id}, expected {dag.act_id}")
            dependencies = [*scene.depends_on, *scene.hard_dependencies, *scene.soft_dependencies]
            for dependency in dependencies:
                if dependency not in scene_by_id:
                    raise ValueError(f"Scene {scene.scene_id} depends on unknown scene {dependency}")
                if dependency == scene.scene_id:
                    raise ValueError(f"Scene {scene.scene_id} cannot depend on itself")
        return self._topological_order(
            dag.scenes,
            lambda scene: scene.scene_id,
            lambda scene: list(dict.fromkeys([*scene.hard_dependencies, *scene.soft_dependencies, *scene.depends_on])),
            "Scene",
        )

    def _validate_state_continuity(self, plan: StoryPlan) -> None:
        """Ensure state continuity across the story using a global blackboard.

        Simulates applying each scene's `state_updates` in act and scene
        order, verifying that each act's expected entrance and exit states
        are satisfied. Attaches the final `GlobalBlackboard` to the plan as
        `plan.global_blackboard`. Raises `ValueError` on mismatches.
        """
        blackboard = GlobalBlackboard()
        dag_by_act = {dag.act_id: dag for dag in plan.scene_dags}
        for act in self._validate_act_dag(plan.acts):
            if not blackboard.satisfies(act.expected_entrance_state):
                raise ValueError(f"Act {act.act_id} entrance state is not satisfied")
            for scene in self._validate_scene_dag(dag_by_act[act.act_id]):
                blackboard.apply(scene.state_updates)
            if not blackboard.satisfies(act.expected_exit_state):
                raise ValueError(f"Act {act.act_id} exit state is not satisfied by its scene updates")

            for downstream in plan.acts:
                if act.act_id in downstream.depends_on:
                    for key, value in downstream.expected_entrance_state.items():
                        if blackboard.state.get(key) != value:
                            raise ValueError(
                                f"Act {act.act_id} exit state does not satisfy {downstream.act_id} entrance key {key}"
                            )
        plan.global_blackboard = blackboard

    def _topological_order(self, items, id_for, dependencies_for, label: str):
        """Topologically sort `items` based on dependencies.

        Args:
            items: Iterable of items to sort.
            id_for: Callable that returns the item's identifier.
            dependencies_for: Callable that returns a list of identifiers
                that the item depends on.
            label: Human-readable label used in error messages.

        Returns:
            A list of items ordered topologically. Raises `ValueError` if a
            cycle or unknown dependency is encountered.
        """
        item_by_id = {id_for(item): item for item in items}
        temporary: set[str] = set()
        permanent: set[str] = set()
        ordered = []

        def visit(item_id: str) -> None:
            if item_id in permanent:
                return
            if item_id in temporary:
                raise ValueError(f"{label} dependency graph contains a cycle")
            temporary.add(item_id)
            for dependency in dependencies_for(item_by_id[item_id]):
                if dependency not in item_by_id:
                    raise ValueError(f"{label} {item_id} depends on unknown {label.lower()} {dependency}")
                visit(dependency)
            temporary.remove(item_id)
            permanent.add(item_id)
            ordered.append(item_by_id[item_id])

        for item_id in item_by_id:
            visit(item_id)
        return ordered

    def _constraints(self, spec: StorySpec) -> list[str]:
        """Return a list of constraint strings derived from `spec`.

        These strings are used by downstream planners and prompts to
        communicate audience and content constraints.
        """
        return [
            f"age_group={spec.age_group}",
            f"moral_theme={spec.moral_theme}",
            f"max_fear={spec.fear_level}",
            f"allowed_conflict={spec.allowed_conflict}",
            f"forbidden_elements={spec.forbidden_elements}",
        ]

    def _act_summary(self, spec: StorySpec, pattern: StoryPattern, index: int, total: int) -> str:
        """Produce a human-readable default act summary.

        Chooses wording for the opening, closing, and middle acts based on
        the `spec` and `pattern`.
        """
        if index == 0:
            return f"Establish the {spec.topic} story world and protagonist."
        if index == total - 1:
            return f"Resolve the {spec.topic} story through {spec.moral_theme}."
        act_name = pattern.acts[min(index, len(pattern.acts) - 1)]
        return f"Develop the {act_name} movement of the {spec.topic} story."

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
