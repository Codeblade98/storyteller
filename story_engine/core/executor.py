from typing import Any

from dataclasses import dataclass
import logging

from story_engine.core.assembler import StoryAssembler
from story_engine.core.dag_builder import SceneDAG
from story_engine.core.propagator import StatePropagator
from story_engine.core.repair import RepairLoop
from story_engine.core.state_manager import StateManager
from story_engine.models.base import ModelMixin
from story_engine.models.scene import SceneOutput
from story_engine.models.story_spec import StorySpec

logger = logging.getLogger("story_engine.engine")


@dataclass
class StoryRun(ModelMixin):
    """Result of executing a complete story generation.

    Attributes:
        spec: The StorySpec used for generation.
        scenes: List of generated SceneOutput objects in order.
        final_state: Final narrative state after all scenes.
        story_text: Complete assembled story text.
        verification_log: Log entries for each scene's verification.
    """
    spec: StorySpec
    scenes: list[SceneOutput]
    final_state: dict[str, Any]
    story_text: str
    verification_log: list[dict[str, Any]]


class TopologicalExecutor:
    """Executes scenes in topological order with state management and repair.

    Processes scenes according to dependency constraints, propagates state,
    verifies outputs, and repairs scenes that fail verification.
    """
    def __init__(
        self,
        state_manager: StateManager,
        propagator: StatePropagator,
        repair_loop: RepairLoop,
        assembler: StoryAssembler,
    ) -> None:
        """Initialize the executor with required components.

        Args:
            state_manager: Manages narrative state throughout execution.
            propagator: Filters state for each scene's requirements.
            repair_loop: Handles scene generation and verification.
            assembler: Assembles final story text from scenes.
        """
        self.state_manager = state_manager
        self.propagator = propagator
        self.repair_loop = repair_loop
        self.assembler = assembler

    def execute(self, spec: StorySpec, dag: SceneDAG) -> StoryRun:
        """Execute all scenes in the DAG in topological order.

        Initializes state, generates each scene with appropriate context,
        verifies output, and accumulates results.

        Args:
            spec: Story specification for validation and configuration.
            dag: Scene dependency graph defining execution order.

        Returns:
            A StoryRun containing all results and metadata.

        Raises:
            RuntimeError: If a scene fails verification after all repair attempts.
        """
        state = self.state_manager.initial_state(spec)
        scenes: list[SceneOutput] = []
        verification_log: list[dict[str, Any]] = []
        ordered_nodes = dag.topological_order()
        logger.info("execution_started scenes=%s", len(ordered_nodes))

        for node in ordered_nodes:
            filtered_state = self.propagator.filter_for_node(state, node)
            logger.info(
                "scene_execution_started scene_id=%s act=%s required_context=%s filtered_context=%s",
                node.scene_id,
                node.plan.act,
                node.plan.required_context,
                sorted(filtered_state),
            )
            output, result = self.repair_loop.generate_valid_scene(node, spec, filtered_state, state)
            verification_log.append(
                {"scene_id": node.scene_id, "ok": result.ok, "failures": result.failures, "status": output.status}
            )
            if not result.ok:
                logger.error("scene_execution_failed scene_id=%s failures=%s", node.scene_id, result.failures)
                raise RuntimeError(f"Scene {node.scene_id} failed verification: {result.failures}")
            state = self.state_manager.apply_diff(state, output.state_diff, node.scene_id)
            scenes.append(output)
            logger.info(
                "scene_execution_committed scene_id=%s status=%s diff_keys=%s completed_scenes=%s",
                node.scene_id,
                output.status,
                sorted(output.state_diff),
                state.get("narrative_state", {}).get("completed_scenes", []),
            )

        run = StoryRun(
            spec=spec,
            scenes=scenes,
            final_state=state,
            story_text=self.assembler.assemble(scenes),
            verification_log=verification_log,
        )
        logger.info("execution_completed scenes=%s final_state_keys=%s", len(scenes), sorted(state))
        return run
