from typing import Any

from dataclasses import dataclass, field
import logging

from story_engine.models.base import ModelMixin
from story_engine.models.scene import SceneOutput
from story_engine.models.story_spec import StorySpec
from story_engine.verification.character import CharacterValidator
from story_engine.verification.chronology import ChronologyValidator
from story_engine.verification.invariants import InvariantValidator
from story_engine.verification.safety import SafetyValidator
from story_engine.verification.state_diff import StateDiffValidator
from story_engine.verification.style import StyleValidator

logger = logging.getLogger("story_engine.verification")


@dataclass
class VerificationResult(ModelMixin):
    """Result of scene verification against story constraints.

    Attributes:
        ok: Whether the scene passed all verification checks.
        failures: List of specific failures if verification failed.
    """
    ok: bool
    failures: list[str] = field(default_factory=list)


class HardConstraintVerifier:
    """Verifies hard constraints with deterministic code only.

    Runs multiple independent validators and aggregates results.
    """
    def __init__(self) -> None:
        """Initialize all verification validators."""
        self.safety = SafetyValidator()
        self.chronology = ChronologyValidator()
        self.character = CharacterValidator()
        self.style = StyleValidator()
        self.invariants = InvariantValidator()
        self.state_diff = StateDiffValidator()

    def verify(self, output: SceneOutput, spec: StorySpec, state: dict[str, Any]) -> VerificationResult:
        """Verify a scene against all story constraints.

        Args:
            output: Generated scene output to verify.
            spec: Story specification with constraints.
            state: Current narrative state for context.

        Returns:
            VerificationResult with success status and any failures.
        """
        completed = state.get("narrative_state", {}).get("completed_scenes", [])
        failures = []
        failures.extend(self.invariants.validate(output))
        failures.extend(self.state_diff.validate(output, state))
        failures.extend(self.safety.validate(output, spec))
        failures.extend(self.chronology.validate(output, completed))
        failures.extend(self.character.validate(output, state))
        failures.extend(self.style.validate(output, spec))
        if failures:
            logger.warning("hard_verification_failed scene_id=%s failures=%s", output.scene_id, failures)
        else:
            logger.info("hard_verification_passed scene_id=%s", output.scene_id)
        return VerificationResult(ok=not failures, failures=failures)


class IncrementalVerifier:
    """Runs hard deterministic checks, then optional soft semantic checks."""

    def __init__(self, semantic_verifier: Any | None = None) -> None:
        self.hard = HardConstraintVerifier()
        self.semantic_verifier = semantic_verifier

    def verify(self, output: SceneOutput, spec: StorySpec, state: dict[str, Any]) -> VerificationResult:
        hard_result = self.hard.verify(output, spec, state)
        if not hard_result.ok or self.semantic_verifier is None:
            return hard_result
        semantic_result = self.semantic_verifier.verify(output, spec, state)
        if semantic_result.ok:
            logger.info("semantic_verification_passed scene_id=%s", output.scene_id)
            return hard_result
        logger.warning("semantic_verification_failed scene_id=%s failures=%s", output.scene_id, semantic_result.failures)
        return VerificationResult(ok=False, failures=semantic_result.failures)
