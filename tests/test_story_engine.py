"""Tests for the story engine and state management."""

from story_engine.core.engine import StoryEngine
from story_engine.core.edge_context_designer import EdgeContextDesigner
from story_engine.core.state_manager import StateManager
from story_engine.models.scene import SceneOutput, ScenePlan
from story_engine.models.story_spec import StoryInput
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
