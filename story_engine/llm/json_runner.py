from pathlib import Path
from typing import Any, TypeVar
import json
import logging
import inspect
from datetime import datetime, timezone
from dataclasses import dataclass

from story_engine.llm.parser import JSONParser
from story_engine.llm.retry import RetryPolicy
from story_engine.llm.router import ModelRouter
from story_engine.models.llm_contracts import LLMRole

T = TypeVar("T")
logger = logging.getLogger("story_engine.llm")
response_logger = logging.getLogger("story_engine.llm_responses")


@dataclass(frozen=True)
class RenderedPrompt:
    system_prompt: str
    user_payload: str


def render_payload_markdown(payload: dict[str, Any]) -> str:
    """Render an LLM task payload as a markdown brief."""
    sections = [_render_markdown_section(key, value) for key, value in payload.items()]
    return "\n\n".join(section for section in sections if section)


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").strip().title()


def _render_markdown_section(key: str, value: Any) -> str:
    title = _humanize_key(key)
    if isinstance(value, dict):
        if not value:
            return f"## {title}\n\nNone."
        lines = [f"## {title}"]
        for item_key, item_value in value.items():
            lines.extend(_render_markdown_field(item_key, item_value))
        return "\n".join(lines)
    if isinstance(value, list):
        return f"## {title}\n\n{_render_markdown_list(value)}"
    return f"## {title}\n\n{_render_scalar(value)}"


def _render_markdown_field(key: str, value: Any) -> list[str]:
    title = _humanize_key(key)
    if isinstance(value, dict):
        if _should_render_as_json(value):
            return [f"\n### {title}", _json_block(value)]
        lines = [f"\n### {title}"]
        for item_key, item_value in value.items():
            lines.extend(_render_markdown_field(item_key, item_value))
        return lines
    if isinstance(value, list):
        return [f"\n### {title}", _render_markdown_list(value)]
    return [f"- **{title}:** {_render_scalar(value)}"]


def _render_markdown_list(value: list[Any]) -> str:
    if not value:
        return "None."
    if all(not isinstance(item, (dict, list)) for item in value):
        return "\n".join(f"- {_render_scalar(item)}" for item in value)
    return _json_block(value)


def _render_scalar(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _should_render_as_json(value: dict[str, Any]) -> bool:
    return any(isinstance(item, (dict, list)) for item in value.values())


def _json_block(value: Any) -> str:
    return f"```json\n{json.dumps(value, ensure_ascii=True, indent=2)}\n```"


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
        client = self.router.client_for(role)
        if client is None:
            logger.debug("llm_task_skipped role=%s template=%s reason=no_client", role, template_name)
            return None

        failures: list[str] = []
        for attempt in range(self.retry_policy.max_attempts + 1):
            prompt = self._render(template_name, payload | {"attempt": attempt, "previous_failures": failures})
            logger.info(
                "llm_task_started role=%s template=%s attempt=%s system_chars=%s user_payload_chars=%s",
                role,
                template_name,
                attempt,
                len(prompt.system_prompt),
                len(prompt.user_payload),
            )
            try:
                parsed = None
                raw = None
                if hasattr(client, "complete_structured"):
                    try:
                        raw, parsed = _complete_structured(
                            client,
                            prompt.user_payload,
                            temperature=self.retry_policy.temperature_for(attempt),
                            response_model=output_model,
                            system_prompt=prompt.system_prompt,
                        )
                    except Exception as exc:
                        logger.debug("llm_structured_parse_failed role=%s template=%s error=%s", role, template_name, exc)
                        parsed = None
                        raw = None
                if parsed is None:
                    raw = _complete_json(
                        client,
                        prompt.user_payload,
                        temperature=self.retry_policy.temperature_for(attempt),
                        system_prompt=prompt.system_prompt,
                    )
                    parsed = self.parser.parse_model(raw, output_model)
                logger.info(
                    "llm_task_completed role=%s template=%s attempt=%s raw_chars=%s",
                    role,
                    template_name,
                    attempt,
                    len(raw) if raw is not None else 0,
                )
                response_logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "task": template_name,
                            "response": raw,
                        },
                        ensure_ascii=True,
                    )
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

    def _render(self, template_name: str, payload: dict[str, Any]) -> RenderedPrompt:
        template = _load_prompt_template(self.template_dir / template_name)
        payload_markdown = render_payload_markdown(payload)
        user_payload = template.user_payload.replace("{{payload_markdown}}", payload_markdown)
        user_payload = user_payload.replace("{{payload_json}}", payload_markdown)
        return RenderedPrompt(system_prompt=template.system_prompt, user_payload=user_payload)


def _load_prompt_template(path: Path) -> RenderedPrompt:
    values = _parse_block_scalar_yaml(path.read_text())
    try:
        return RenderedPrompt(
            system_prompt=values["system_prompt"].strip(),
            user_payload=values["user_payload"].strip(),
        )
    except KeyError as exc:
        raise ValueError(f"{path} must define system_prompt and user_payload block scalars") from exc


def _parse_block_scalar_yaml(source: str) -> dict[str, str]:
    values: dict[str, str] = {}
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith(" ") or ":" not in line:
            index += 1
            continue

        key, marker = line.split(":", 1)
        key = key.strip()
        marker = marker.strip()
        if marker != "|":
            raise ValueError(f"Prompt YAML field {key!r} must use a literal block scalar")

        index += 1
        block_lines: list[str] = []
        while index < len(lines):
            block_line = lines[index]
            if block_line and not block_line.startswith(" "):
                break
            block_lines.append(block_line[2:] if block_line.startswith("  ") else "")
            index += 1
        values[key] = "\n".join(block_lines).strip()
    return values


def _complete_json(client: Any, prompt: str, *, temperature: float, system_prompt: str) -> str:
    method = client.complete_json
    if "system_prompt" in inspect.signature(method).parameters:
        return method(prompt, temperature=temperature, system_prompt=system_prompt)
    return method(prompt, temperature=temperature)


def _complete_structured(
    client: Any,
    prompt: str,
    *,
    temperature: float,
    response_model: type[T],
    system_prompt: str,
) -> tuple[str, T]:
    method = client.complete_structured
    if "system_prompt" in inspect.signature(method).parameters:
        return method(
            prompt,
            temperature=temperature,
            response_model=response_model,
            system_prompt=system_prompt,
        )
    return method(prompt, temperature=temperature, response_model=response_model)
