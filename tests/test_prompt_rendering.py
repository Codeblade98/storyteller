from story_engine.llm.json_runner import JSONTaskRunner, render_payload_markdown


def test_payload_renderer_creates_markdown_sections() -> None:
    rendered = render_payload_markdown(
        {
            "scene_id": "S1",
            "hard_constraints": {
                "scene_count": 2,
                "forbidden_elements": ["gore", "jump scares"],
            },
            "previous_failures": [],
        }
    )

    assert "## Scene Id\n\nS1" in rendered
    assert "## Hard Constraints" in rendered
    assert "- **Scene Count:** 2" in rendered
    assert "- gore" in rendered
    assert "## Previous Failures\n\nNone." in rendered


def test_json_task_runner_renders_markdown_payload(tmp_path) -> None:
    template = tmp_path / "sample_prompt.yaml"
    template.write_text(
        """system_prompt: |
  You are a focused test assistant.

user_payload: |
  # Task Payload

  {{payload_markdown}}
"""
    )
    runner = JSONTaskRunner(template_dir=tmp_path)

    prompt = runner._render("sample_prompt.yaml", {"story_input": {"topic": "moon garden"}})

    assert prompt.system_prompt == "You are a focused test assistant."
    assert "# Task Payload" in prompt.user_payload
    assert "## Story Input" in prompt.user_payload
    assert "- **Topic:** moon garden" in prompt.user_payload
    assert "{{payload_markdown}}" not in prompt.user_payload
