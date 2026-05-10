from typing import Any

import logging

from story_engine.generation.scene_generator import SceneGenerator
from story_engine.generation.state_diff_extractor import StateDiffExtractor
from story_engine.models.node import SceneNode
from story_engine.models.scene import SceneOutput
from story_engine.models.story_spec import StorySpec
from story_engine.verification.verifier import IncrementalVerifier, VerificationResult

logger = logging.getLogger("story_engine.engine")


class RepairLoop:
    """Attempts scene generation and repairs failed scenes through verification.

    Retries with adjusted parameters and repair instructions when verification fails.
    """
    def __init__(
        self,
        generator: SceneGenerator,
        verifier: IncrementalVerifier,
        extractor: StateDiffExtractor | None = None,
        max_attempts: int = 2,
    ) -> None:
        """Initialize the repair loop with generation and verification components.

        Args:
            generator: Scene generator to create/repair scenes.
            verifier: Verifier to check scene validity.
            max_attempts: Maximum generation attempts before failing (default 2).
        """
        self.generator = generator
        self.verifier = verifier
        self.extractor = extractor or StateDiffExtractor()
        self.max_attempts = max_attempts

    def generate_valid_scene(
        self,
        node: SceneNode,
        spec: StorySpec,
        filtered_state: dict[str, Any],
        global_state: dict[str, Any],
    ) -> tuple[SceneOutput, VerificationResult]:
        """Generate a scene and repair if verification fails.

        Attempts to generate a valid scene up to max_attempts times,
        providing repair instructions on each retry.

        Args:
            node: Scene node to generate.
            spec: Story specification for generation.
            filtered_state: State relevant to this scene.
            global_state: Complete narrative state for verification.

        Returns:
            Tuple of (SceneOutput, VerificationResult) with final attempt results.
        """
        failures: list[str] = []
        for attempt in range(self.max_attempts + 1):
            logger.info("scene_generation_attempt scene_id=%s attempt=%s failures=%s", node.scene_id, attempt, failures)
            output = self.generator.generate(
                node=node,
                spec=spec,
                filtered_state=filtered_state,
                repair_instructions=failures,
                attempt=attempt,
            )
            output.state_diff = self.extractor.extract(
                draft=output,
                node=node,
                spec=spec,
                filtered_state=filtered_state,
            )
            result = self.verifier.verify(output, spec, global_state)
            if result.ok:
                if attempt:
                    output.status = "repaired"
                logger.info(
                    "scene_generation_valid scene_id=%s attempt=%s status=%s diff_keys=%s",
                    node.scene_id,
                    attempt,
                    output.status,
                    sorted(output.state_diff),
                )
                return output, result
            failures = result.failures
            logger.warning("scene_generation_invalid scene_id=%s attempt=%s failures=%s", node.scene_id, attempt, failures)
        return output, result
