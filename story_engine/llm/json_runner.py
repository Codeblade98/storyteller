from pathlib import Path
from typing import Any, TypeVar
import logging

from story_engine.llm.parser import JSONParser
from story_engine.llm.retry import RetryPolicy
from story_engine.llm.router import ModelRouter
from story_engine.models.llm_contracts import LLMRole

T = TypeVar("T")
logger = logging.getLogger("story_engine.llm")


class JSONTaskRunner:
    def __init__(
        self,
        router: ModelRouter | None = None,
        retry_policy: RetryPolicy | None = None,
        template_dir: Path | None = None,
    ) -> None:
        self.router = router or ModelRouter()
        self.retry_policy = retry_policy or RetryPolicy()
        self.parser = JSONParser()
        self.template_dir = template_dir or Path(__file__).parents[1] / "prompts"

    def run_json_task(
        self,
        *,
        role: LLMRole,
        template_name: str,
        payload: dict[str, Any],
        output_model: type[T],
    ) -> T | None:
        print("LLM CALLED")
        client = self.router.client_for(role)
        if client is None:
            logger.debug("llm_task_skipped role=%s template=%s reason=no_client", role, template_name)
            return None

        failures: list[str] = []
        for attempt in range(self.retry_policy.max_attempts + 1):
            prompt = self._render(template_name, payload | {"attempt": attempt, "previous_failures": failures})
            logger.info(
                "llm_task_started role=%s template=%s attempt=%s prompt_chars=%s",
                role,
                template_name,
                attempt,
                len(prompt),
            )
            try:
                raw = client.complete_json(prompt, temperature=self.retry_policy.temperature_for(attempt))
                parsed = self.parser.parse_model(raw, output_model)
                logger.info(
                    "llm_task_completed role=%s template=%s attempt=%s raw_chars=%s",
                    role,
                    template_name,
                    attempt,
                    len(raw),
                )
                print("LLM RESPONSE:", raw)
                return parsed
            except Exception as exc:
                failures = [f"{type(exc).__name__}: {exc}"]
                logger.warning(
                    "llm_task_failed role=%s template=%s attempt=%s error=%s",
                    role,
                    template_name,
                    attempt,
                    failures[0],
                )
        logger.error("llm_task_exhausted role=%s template=%s failures=%s", role, template_name, failures)
        return None

    def has_client(self, role: LLMRole) -> bool:
        return self.router.client_for(role) is not None

    def _render(self, template_name: str, payload: dict[str, Any]) -> str:
        import json

        template = (self.template_dir / template_name).read_text()
        return template.replace("{{payload_json}}", json.dumps(payload, ensure_ascii=True, indent=2))
