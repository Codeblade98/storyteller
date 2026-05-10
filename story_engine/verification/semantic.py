from dataclasses import asdict
from typing import Any
import logging

from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.models.llm_contracts import SemanticVerificationContract
from story_engine.models.scene import SceneOutput
from story_engine.models.story_spec import StorySpec
from story_engine.verification.verifier import VerificationResult

logger = logging.getLogger("story_engine.verification")


class SemanticVerifier:
    def __init__(self, llm_runner: JSONTaskRunner | None = None) -> None:
        self.llm_runner = llm_runner

    def verify(self, output: SceneOutput, spec: StorySpec, state: dict[str, Any]) -> VerificationResult:
        uncertainty = self._deterministic_uncertainty(output, spec)
        if not uncertainty or self.llm_runner is None or not self.llm_runner.has_client("verifier"):
            logger.debug(
                "semantic_verification_skipped scene_id=%s reasons=%s has_runner=%s",
                output.scene_id,
                uncertainty,
                self.llm_runner is not None,
            )
            return VerificationResult(ok=True, failures=[])

        contract = self.llm_runner.run_json_task(
            role="verifier",
            template_name="semantic_verifier_prompt.yaml",
            output_model=SemanticVerificationContract,
            payload={
                "scene": asdict(output),
                "story_spec": asdict(spec),
                "state_snapshot": state,
                "uncertainty_reasons": uncertainty,
                "output_schema": {"ok": True, "failures": [], "confidence": 0.0},
            },
        )
        if contract is None or contract.confidence < 0.4:
            return VerificationResult(ok=True, failures=[])
        return VerificationResult(ok=contract.ok, failures=contract.failures)

    def _deterministic_uncertainty(self, output: SceneOutput, spec: StorySpec) -> list[str]:
        reasons = []
        lower = output.scene_text.lower()
        if spec.moral_theme not in lower and output.metadata.get("fear_score", 0) >= spec.fear_level:
            reasons.append("theme may be under-expressed during high emotion")
        if "felt" not in lower and "emotion" in " ".join(output.state_diff):
            reasons.append("state diff changes emotion but prose may not show it")
        return reasons
