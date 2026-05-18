import logging

from story_engine.core.assembler import StoryAssembler
from story_engine.core.dag_builder import SceneDAGBuilder
from story_engine.core.edge_context_designer import EdgeContextDesigner
from story_engine.core.executor import StoryRun, TopologicalExecutor
from story_engine.core.planner import ActScenePlanner
from story_engine.core.propagator import StatePropagator
from story_engine.core.repair import RepairLoop
from story_engine.core.spec_builder import StorySpecBuilder
from story_engine.core.state_manager import StateManager
from story_engine.generation.scene_generator import SceneGenerator
from story_engine.generation.state_diff_extractor import StateDiffExtractor
from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.llm.router import ModelRouter
from story_engine.models.story_spec import StoryInput
from story_engine.retrieval.patterns import PatternLibrary
from story_engine.verification.semantic import SemanticVerifier
from story_engine.verification.verifier import IncrementalVerifier

logger = logging.getLogger("story_engine.engine")


class StoryEngine:
    """Main story generation engine orchestrating all components.

    Coordinates scene planning, dependency resolution, generation, verification,
    and assembly into a complete story.
    """
    def __init__(
        self,
        generator: SceneGenerator | None = None,
        model_router: ModelRouter | None = None,
        llm_runner: JSONTaskRunner | None = None,
    ) -> None:
        """Initialize the story engine with all required components.

        Args:
            generator: Optional custom SceneGenerator. If not provided,
                      a default one is created.
        """
        self.pattern_library = PatternLibrary()
        self.llm_runner = llm_runner or JSONTaskRunner(router=model_router)
        self.spec_builder = StorySpecBuilder(llm_runner=self.llm_runner)
        self.planner = ActScenePlanner(llm_runner=self.llm_runner)
        self.dag_builder = SceneDAGBuilder(edge_context_designer=EdgeContextDesigner(llm_runner=self.llm_runner))
        self.generator = generator or SceneGenerator(llm_runner=self.llm_runner)
        self.extractor = StateDiffExtractor(llm_runner=self.llm_runner)
        self.verifier = IncrementalVerifier(semantic_verifier=SemanticVerifier(llm_runner=self.llm_runner))
        self.state_manager = StateManager()
        self.executor = TopologicalExecutor(
            state_manager=self.state_manager,
            propagator=StatePropagator(),
            repair_loop=RepairLoop(self.generator, self.verifier, extractor=self.extractor),
            assembler=StoryAssembler(),
        )

    def create_story(self, story_input: StoryInput) -> StoryRun:
        """Generate a complete story from user input.

        Orchestrates the full pipeline: specification building, pattern selection,
        planning, DAG construction, and topological execution.

        Args:
            story_input: User input specifying the story requirements.

        Returns:
            A StoryRun containing the generated scenes and final state.
        """
        logger.info(
            "story_pipeline_started topic=%r age_group=%s genre=%s length=%s",
            story_input.topic,
            story_input.age_group,
            story_input.genre,
            story_input.length,
        )
        
        ## 1. Build the story specification from user input
        spec = self.spec_builder.build(story_input)
        logger.info(
            "spec_built topic=%r target_scene_count=%s vocab_level=%s fear_level=%s",
            spec.topic,
            spec.target_scene_count,
            spec.vocab_level,
            spec.fear_level,
        )
                
        ## 2. Run pattern selction from pattern library
        pattern = self.pattern_library.select(spec.genre, spec.target_scene_count)
        logger.info("pattern_selected pattern=%s acts=%s", pattern.pattern_name, len(pattern.acts))
        
        ## 3. Build hierarchical story plan from the story specification and pattern
        story_plan = self.planner.plan(spec, pattern)
        scene_plans = story_plan.flatten_scenes()
        logger.info(
            "story_plan_created acts=%s scenes=%s ids=%s",
            len(story_plan.acts),
            len(scene_plans),
            [plan.scene_id for plan in scene_plans],
        )
        
        dag = self.dag_builder.build(scene_plans)
        logger.info("dag_built nodes=%s edges=%s", len(dag.nodes), len(dag.edges))
        run = self.executor.execute(spec, dag)
        logger.info("story_pipeline_completed scenes=%s story_words=%s", len(run.scenes), len(run.story_text.split()))
        return run
import logging
