"""Tests for the story engine and state management."""

from typing import Any

from story_engine.core.engine import StoryEngine
from story_engine.core.edge_context_designer import EdgeContextDesigner
from story_engine.core.planner import ActScenePlanner
from story_engine.core.state_manager import StateManager
from story_engine.models.llm_contracts import ActPlanningContract, ActScenePlanningContract
from story_engine.models.scene import SceneOutput, ScenePlan
from story_engine.models.story_spec import StoryInput, StorySpec
from story_engine.retrieval.patterns import StoryPattern
from story_engine.verification.state_diff import StateDiffValidator


def test_story_engine_generates_valid_stateful_story() -> None:
    """Test that the story engine generates a complete valid story.

    Verifies:
    - Story has the correct number of scenes.
    - All scenes pass verification.
    - State is properly tracked throughout.
    - Story topic is reflected in output.
    """
    engine = StoryEngine()
    run = engine.create_story(
        StoryInput(topic="dragon friendship", age_group="7-9", genre="fantasy", fear_level=2, length="short")
    )

    assert len(run.scenes) == 6
    assert all(entry["ok"] for entry in run.verification_log)
    assert run.final_state["narrative_state"]["completed_scenes"] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert "Milo" in run.final_state["character_state"]
    assert "dragon friendship" in run.story_text.lower()


def test_state_manager_merges_dotted_diffs_without_overwriting_state() -> None:
    """Test that state manager properly applies dotted path diffs.

    Verifies:
    - Nested paths are correctly created and updated.
    - List appending with .add suffix works as expected.
    - Time advancement special key is handled.
    - Completed scenes are tracked correctly.
    """
    story_input = StoryInput(topic="moon garden", age_group="7-9", genre="fantasy", fear_level=1, length="short")
    spec = StoryEngine().spec_builder.build(story_input)
    manager = StateManager()
    state = manager.initial_state(spec)

    state = manager.apply_diff(
        state,
        {
            "character.Milo.emotion": "curious",
            "character.Milo.inventory.add": ["lantern"],
            "time.advance": "noon",
        },
        "S1",
    )

    assert state["character_state"]["Milo"]["emotion"] == "curious"
    assert state["character_state"]["Milo"]["inventory"] == ["lantern"]
    assert state["world_state"]["time"] == "noon"
    assert state["narrative_state"]["completed_scenes"] == ["S1"]


def test_scene_generator_does_not_commit_state_without_extractor() -> None:
    engine = StoryEngine()
    run = engine.create_story(
        StoryInput(topic="dragon friendship", age_group="7-9", genre="fantasy", fear_level=2, length="short")
    )

    assert run.scenes[0].state_diff
    assert "state_diff" not in run.scenes[0].scene_text


def test_edge_context_designer_rejects_illegal_context_keys() -> None:
    source = ScenePlan(scene_id="S1", act="setup", goal="introduce Milo")
    target = ScenePlan(
        scene_id="S2",
        act="discovery",
        goal="find a clue",
        required_context=["world_state", "full_scene_history", "unknown_state"],
    )

    edge = EdgeContextDesigner().design(source, target, "hard")

    assert edge.required_context == ["world_state"]
    assert "full_scene_history" in edge.ignored_context


def test_planner_uses_two_step_llm_calls_for_acts_then_scenes() -> None:
    class FakePlannerRunner:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def has_client(self, role: str) -> bool:
            return role == "planner"

        def run_json_task(self, *, role: str, template_name: str, payload: dict[str, Any], output_model: type) -> Any:
            self.calls.append({"template_name": template_name, "payload": payload, "output_model": output_model})
            if template_name == "planner_acts_prompt.yaml":
                return ActPlanningContract(
                    acts=[
                        {"act_id": "anything", "act_number": 99, "act_summary": "Open the mystery.", "depends_on": []},
                        {"act_id": "ignored", "act_number": 99, "act_summary": "Follow the clue.", "depends_on": ["A1"]},
                        {"act_id": "ignored", "act_number": 99, "act_summary": "Bring it home.", "depends_on": ["A2"]},
                    ]
                )

            act_id = payload["act"]["act_id"]
            scene_count = payload["hard_constraints"]["scene_count"]
            return ActScenePlanningContract(
                act_id=act_id,
                scenes=[
                    {
                        "scene_id": f"{act_id}-bad-{index}",
                        "act_id": act_id,
                        "act": "setup",
                        "goal": f"{act_id} scene {index}",
                        "emotion_level": index,
                        "depends_on": [] if index == 0 else [f"S{index}"],
                        "hard_dependencies": [] if index == 0 else [f"S{index}"],
                        "soft_dependencies": [],
                        "required_context": ["style_state", "full_scene_history"],
                        "constraints": [],
                        "state_updates": {},
                    }
                    for index in range(scene_count)
                ],
            )

    spec = StorySpec(
        topic="moon garden",
        age_group="7-9",
        genre="fantasy",
        fear_level=1,
        length="short",
        vocab_level="clear",
        sentence_complexity=2,
        allowed_conflict=[],
        forbidden_elements=["gore"],
        moral_theme="patience",
        target_scene_count=6,
    )
    pattern = StoryPattern(
        pattern_name="test",
        acts=["setup", "discovery", "challenge", "cooperation", "resolution", "resolution"],
        emotion_curve=[0, 1, 2, 3, 2, 1],
        genres=["fantasy"],
    )
    runner = FakePlannerRunner()

    plan = ActScenePlanner(llm_runner=runner).plan(spec, pattern)  # type: ignore[arg-type]

    assert [call["template_name"] for call in runner.calls] == [
        "planner_acts_prompt.yaml",
        "planner_scenes_prompt.yaml",
        "planner_scenes_prompt.yaml",
        "planner_scenes_prompt.yaml",
    ]
    first_payload = runner.calls[0]["payload"]
    assert "scene_dags" not in first_payload["output_schema"]
    assert first_payload["hard_constraints"]["target_act_count"] == 3
    assert [act.act_summary for act in plan.acts] == ["Open the mystery.", "Follow the clue.", "Bring it home."]
    assert [scene.scene_id for scene in plan.flatten_scenes()] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert all("full_scene_history" not in scene.required_context for scene in plan.flatten_scenes())
    assert [dag.scenes[-1].state_updates for dag in plan.scene_dags] == [
        {"story_phase": "A1"},
        {"story_phase": "A2"},
        {"story_phase": "A3"},
    ]


def test_state_diff_validator_blocks_unmanaged_mutation() -> None:
    failures = StateDiffValidator().validate(
        SceneOutput(
            scene_id="S1",
            scene_text="Milo smiled.",
            state_diff={"narrative_state.completed_scenes": ["S9"], "external.memory": "bad"},
        ),
        {"narrative_state": {"completed_scenes": []}},
    )

    assert any("managed path" in failure for failure in failures)
    assert any("Illegal state diff root" in failure for failure in failures)
