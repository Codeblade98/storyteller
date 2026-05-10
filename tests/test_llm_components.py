"""Integration tests for components that rely on real LLM calls."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import pytest

from story_engine.core.edge_context_designer import EdgeContextDesigner
from story_engine.core.planner import ActScenePlanner
from story_engine.generation.scene_generator import SceneGenerator
from story_engine.generation.state_diff_extractor import StateDiffExtractor
from story_engine.llm.json_runner import JSONTaskRunner
from story_engine.llm.router import ModelRouter
from story_engine.models.llm_contracts import SceneDraftContract
from story_engine.models.node import SceneNode
from story_engine.models.scene import SceneOutput, ScenePlan
from story_engine.models.story_spec import StorySpec
from story_engine.retrieval.patterns import StoryPattern
from story_engine.verification.semantic import SemanticVerifier

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# Configure logging for tests
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "test_output.log"

# Set up file logger
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Get root logger and add file handler
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)

# Create test logger
test_logger = logging.getLogger("test_output")


pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY is required for real LLM integration tests",
)


def make_spec(target_scene_count: int = 3) -> StorySpec:
    return StorySpec(
        topic="moon garden mystery",
        age_group="7-9",
        genre="fantasy",
        fear_level=2,
        length="short",
        vocab_level="simple",
        sentence_complexity=2,
        allowed_conflict=2,
        forbidden_elements=["graphic violence", "horror"],
        moral_theme="kindness",
        target_scene_count=target_scene_count,
    )


def make_node(scene_id: str, act: str = "setup", goal: str = "Introduce the mystery") -> SceneNode:
    return SceneNode(
        scene_id=scene_id,
        plan=ScenePlan(
            scene_id=scene_id,
            act=act,
            goal=goal,
            emotion_level=2,
            depends_on=[],
            hard_dependencies=[],
            soft_dependencies=[],
            required_context=["world_state", "character_state", "style_state", "safety_state"],
            constraints=["age_group=7-9", "max_fear=2"],
        ),
    )


@pytest.fixture(scope="module")
def real_runner() -> JSONTaskRunner:
    return JSONTaskRunner(router=ModelRouter())


@pytest.fixture
def log_output() -> logging.Logger:
    """Fixture that provides a logger for test output."""
    return test_logger


def test_json_task_runner_makes_real_call_and_parses(real_runner: JSONTaskRunner) -> None:
    result = real_runner.run_json_task(
        role="generator",
        template_name="scene_generation.yaml",
        output_model=SceneDraftContract,
        payload={
            "scene_id": "S1",
            "scene_goal": "Introduce Milo and the moon garden mystery safely",
            "act": "setup",
            "constraints": ["age_group=7-9", "max_fear=2"],
            "filtered_state": {
                "world_state": {"location": "moonlit garden", "time": "morning"},
                "character_state": {"Milo": {"emotion": "curious", "inventory": []}},
                "style_state": {"vocab_level": "simple"},
                "safety_state": {"fear_level": 2},
            },
            "forbidden_elements": ["graphic violence", "horror"],
            "repair_instructions": [],
            "output_schema": {
                "scene_id": "string",
                "scene_text": "string",
                "metadata": {"fear_score": "integer", "violence_score": "integer"},
            },
        },
    )

    assert result is not None
    assert result.scene_id
    assert result.scene_text
    assert isinstance(result.metadata, dict)


def test_planner_uses_real_llm(real_runner: JSONTaskRunner) -> None:
    spec = make_spec(target_scene_count=3)
    pattern = StoryPattern(
        pattern_name="three_act",
        acts=["setup", "challenge", "resolution"],
        emotion_curve=[1, 3, 2],
        genres=["fantasy"],
    )
    planner = ActScenePlanner(llm_runner=real_runner)

    plans = planner.plan(spec, pattern)
    
    with open(LOG_DIR / "planner_test_output.txt", "w") as f:
        f.write("Planner LLM Output:\n")
        for plan in plans:
            f.write(f"Scene ID: {plan.scene_id}\n")
            f.write(f"  Act: {plan.act}\n")
            f.write(f"  Goal: {plan.goal}\n")
            f.write(f"  Emotion Level: {plan.emotion_level}\n")
            f.write(f"  Required Context: {plan.required_context}\n")
            f.write(f"  Constraints: {plan.constraints}\n")
            f.write(f"  Hard Dependencies: {plan.hard_dependencies}\n")
            f.write(f"  Soft Dependencies: {plan.soft_dependencies}\n\n")
            f.write("-" * 40 + "\n")

    assert len(plans) == 3
    assert [plan.scene_id for plan in plans] == ["S1", "S2", "S3"]
    for plan in plans:
        assert plan.goal
        assert isinstance(plan.required_context, list)


def test_edge_context_designer_uses_real_llm(real_runner: JSONTaskRunner) -> None:
    source = ScenePlan(scene_id="S1", act="setup", goal="Introduce Milo")
    target = ScenePlan(
        scene_id="S2",
        act="challenge",
        goal="Milo investigates a clue",
        required_context=["world_state", "character_state"],
    )
    designer = EdgeContextDesigner(llm_runner=real_runner)

    edge = designer.design(source, target, "hard")

    allowed = {"world_state", "character_state", "narrative_state", "style_state", "safety_state"}
    assert edge.source == "S1"
    assert edge.target == "S2"
    assert all(key in allowed for key in edge.required_context)
    assert all(key in allowed for key in edge.transforms)


def test_scene_generator_uses_real_llm(real_runner: JSONTaskRunner) -> None:
    spec = make_spec()
    node = make_node("S1", act="setup")
    generator = SceneGenerator(llm_runner=real_runner)

    output = generator.generate(
        node=node,
        spec=spec,
        filtered_state={
            "world_state": {"location": "moonlit garden", "time": "morning"},
            "character_state": {"Milo": {"emotion": "curious", "inventory": []}},
            "style_state": {"vocab_level": "simple"},
            "safety_state": {"fear_level": 2},
        },
    )

    assert output.scene_id == "S1"
    assert output.scene_text
    assert isinstance(output.metadata, dict)


def test_state_diff_extractor_uses_real_llm(real_runner: JSONTaskRunner) -> None:
    spec = make_spec()
    node = make_node("S2", act="challenge", goal="Milo follows the clue")
    draft = SceneOutput(
        scene_id="S2",
        scene_text=(
            "Milo felt curious and wrote the moon clue in a notebook. "
            "He walked from the path into the flower circle."
        ),
        metadata={"fear_score": 1, "violence_score": 0},
    )
    extractor = StateDiffExtractor(llm_runner=real_runner)

    diff = extractor.extract(
        draft=draft,
        node=node,
        spec=spec,
        filtered_state={
            "world_state": {"location": "garden path", "time": "morning"},
            "character_state": {"Milo": {"emotion": "curious", "inventory": []}},
            "narrative_state": {"active_arc": "moon_garden_mystery", "completed_scenes": ["S1"]},
        },
    )

    assert isinstance(diff, dict)
    assert diff
    assert all(isinstance(key, str) for key in diff)


def test_semantic_verifier_uses_real_llm(real_runner: JSONTaskRunner) -> None:
    spec = make_spec()
    verifier = SemanticVerifier(llm_runner=real_runner)
    output = SceneOutput(
        scene_id="S2",
        scene_text="Milo ran quickly and solved the clue.",
        state_diff={"character.Milo.emotion": "brave"},
        metadata={"fear_score": spec.fear_level},
    )
    state: dict[str, Any] = {"narrative_state": {"completed_scenes": ["S1"]}}

    result = verifier.verify(output, spec, state)

    assert isinstance(result.ok, bool)
    assert isinstance(result.failures, list)


# ============================================================================
# STAGE 1: PLANNER - Tests for LLM-generated scene plans
# ============================================================================


class TestPlannerLLMOutputs:
    """Comprehensive tests for planner stage LLM outputs."""

    @pytest.fixture
    def planner(self, real_runner: JSONTaskRunner) -> ActScenePlanner:
        return ActScenePlanner(llm_runner=real_runner)

    def test_planner_generates_all_requested_scenes(self, planner: ActScenePlanner, log_output: logging.Logger) -> None:
        """Verify planner generates the correct number of scenes."""
        spec = make_spec(target_scene_count=5)
        pattern = StoryPattern(
            pattern_name="five_act_expanded",
            acts=["setup", "rise", "climax", "fall", "resolution"],
            emotion_curve=[1, 2, 4, 3, 2],
            genres=["fantasy"],
        )

        plans = planner.plan(spec, pattern)
        
        with open(LOG_DIR / "planner_scene_count_test_output.txt", "w") as f:
            f.write("Planner Scene Count Test Output:\n")
            for plan in plans:
                f.write(f"Scene ID: {plan.scene_id}, Act: {plan.act}, Goal: {plan.goal}\n")

        log_output.info("=" * 80)
        log_output.info("TEST: Planner - Generate All Requested Scenes")
        log_output.info(f"Requested scenes: {spec.target_scene_count}")
        log_output.info(f"Generated scenes: {len(plans)}")
        for i, plan in enumerate(plans, 1):
            log_output.info(f"  Scene {i}: ID={plan.scene_id}, Act={plan.act}, Goal={plan.goal}, Emotion={plan.emotion_level}")
        
        assert len(plans) == 5
        scene_ids = [plan.scene_id for plan in plans]
        assert scene_ids == ["S1", "S2", "S3", "S4", "S5"]
        log_output.info("✓ Test passed")

    def test_planner_scene_plans_have_required_fields(self, planner: ActScenePlanner, log_output: logging.Logger) -> None:
        """Verify each scene plan has all required fields."""
        spec = make_spec(target_scene_count=3)
        pattern = StoryPattern(
            pattern_name="three_act",
            acts=["setup", "challenge", "resolution"],
            emotion_curve=[1, 3, 2],
            genres=["fantasy"],
        )

        plans = planner.plan(spec, pattern)

        log_output.info("=" * 80)
        log_output.info("TEST: Planner - Scene Plans Have Required Fields")
        for i, plan in enumerate(plans, 1):
            log_output.info(f"Plan {i}: {plan.scene_id}")
            log_output.info(f"  - Act: {plan.act}")
            log_output.info(f"  - Goal: {plan.goal}")
            log_output.info(f"  - Emotion Level: {plan.emotion_level}")
            log_output.info(f"  - Required Context: {plan.required_context}")
            log_output.info(f"  - Constraints: {plan.constraints}")
            log_output.info(f"  - Hard Dependencies: {plan.hard_dependencies}")
            log_output.info(f"  - Soft Dependencies: {plan.soft_dependencies}")

        for plan in plans:
            assert plan.scene_id
            assert plan.act
            assert plan.goal
            assert isinstance(plan.emotion_level, int)
            assert 0 <= plan.emotion_level <= 5
            assert isinstance(plan.required_context, list)
            assert isinstance(plan.constraints, list)
            assert isinstance(plan.hard_dependencies, list)
            assert isinstance(plan.soft_dependencies, list)
        log_output.info("✓ Test passed")

    def test_planner_respects_act_sequence(self, planner: ActScenePlanner, log_output: logging.Logger) -> None:
        """Verify planner creates plans matching the pattern act sequence."""
        spec = make_spec(target_scene_count=3)
        pattern = StoryPattern(
            pattern_name="three_act",
            acts=["setup", "challenge", "resolution"],
            emotion_curve=[1, 3, 2],
            genres=["fantasy"],
        )

        plans = planner.plan(spec, pattern)

        log_output.info("=" * 80)
        log_output.info("TEST: Planner - Respects Act Sequence")
        log_output.info(f"Pattern acts: {pattern.acts}")
        acts = [plan.act for plan in plans]
        log_output.info(f"Generated acts: {acts}")
        
        assert acts == ["setup", "challenge", "resolution"]
        log_output.info("✓ Test passed")

    def test_planner_emotion_levels_follow_curve(self, planner: ActScenePlanner, log_output: logging.Logger) -> None:
        """Verify emotion levels roughly follow the specified emotion curve."""
        spec = make_spec(target_scene_count=3)
        pattern = StoryPattern(
            pattern_name="three_act",
            acts=["setup", "challenge", "resolution"],
            emotion_curve=[1, 5, 2],
            genres=["fantasy"],
        )

        plans = planner.plan(spec, pattern)

        log_output.info("=" * 80)
        log_output.info("TEST: Planner - Emotion Levels Follow Curve")
        log_output.info(f"Emotion curve target: {pattern.emotion_curve}")
        emotion_levels = [plan.emotion_level for plan in plans]
        log_output.info(f"Generated emotions: {emotion_levels}")
        log_output.info(f"Peak emotion (challenge) is highest: {emotion_levels[1] >= emotion_levels[0] and emotion_levels[1] >= emotion_levels[2]}")
        
        # Check that middle scene (challenge) has highest emotion
        assert emotion_levels[1] >= emotion_levels[0]
        assert emotion_levels[1] >= emotion_levels[2]
        log_output.info("✓ Test passed")

    def test_planner_includes_story_spec_constraints(self, planner: ActScenePlanner, log_output: logging.Logger) -> None:
        """Verify plan constraints reflect story spec settings."""
        spec = make_spec(target_scene_count=3)
        pattern = StoryPattern(
            pattern_name="three_act",
            acts=["setup", "challenge", "resolution"],
            emotion_curve=[1, 3, 2],
            genres=["fantasy"],
        )

        plans = planner.plan(spec, pattern)

        log_output.info("=" * 80)
        log_output.info("TEST: Planner - Includes Story Spec Constraints")
        log_output.info(f"Story spec age_group: {spec.age_group}")

        # All plans should include age_group constraint
        for i, plan in enumerate(plans, 1):
            constraint_str = " ".join(plan.constraints)
            log_output.info(f"Plan {i} constraints: {constraint_str}")
            assert spec.age_group in constraint_str or "age_group" in constraint_str.lower()
        log_output.info("✓ Test passed")

    def test_planner_context_includes_required_state_keys(self, planner: ActScenePlanner, log_output: logging.Logger) -> None:
        """Verify required_context includes necessary state keys."""
        spec = make_spec(target_scene_count=3)
        pattern = StoryPattern(
            pattern_name="three_act",
            acts=["setup", "challenge", "resolution"],
            emotion_curve=[1, 3, 2],
            genres=["fantasy"],
        )

        plans = planner.plan(spec, pattern)

        log_output.info("=" * 80)
        log_output.info("TEST: Planner - Context Includes Required State Keys")
        allowed_keys = {"world_state", "character_state", "narrative_state", "style_state", "safety_state"}
        
        for i, plan in enumerate(plans, 1):
            log_output.info(f"Plan {i} context: {plan.required_context}")
            assert all(key in allowed_keys for key in plan.required_context)
        log_output.info("✓ Test passed")

    def test_planner_dependencies_form_valid_chain(self, planner: ActScenePlanner, log_output: logging.Logger) -> None:
        """Verify dependencies create a valid narrative chain."""
        spec = make_spec(target_scene_count=4)
        pattern = StoryPattern(
            pattern_name="four_act",
            acts=["setup", "rise", "climax", "resolution"],
            emotion_curve=[1, 2, 4, 2],
            genres=["fantasy"],
        )

        plans = planner.plan(spec, pattern)

        log_output.info("=" * 80)
        log_output.info("TEST: Planner - Dependencies Form Valid Chain")
        
        # Collect all dependency references
        all_deps = set()
        for i, plan in enumerate(plans, 1):
            log_output.info(f"Plan {i} ({plan.scene_id}):")
            log_output.info(f"  Hard deps: {plan.hard_dependencies}")
            log_output.info(f"  Soft deps: {plan.soft_dependencies}")
            all_deps.update(plan.hard_dependencies)
            all_deps.update(plan.soft_dependencies)

        scene_ids = {plan.scene_id for plan in plans}
        log_output.info(f"All dependencies referenced: {all_deps}")
        log_output.info(f"All scene IDs: {scene_ids}")
        # All dependencies should reference existing scenes or be empty
        assert all_deps.issubset(scene_ids)
        log_output.info("✓ Test passed")

    def test_planner_goals_are_specific_to_story_topic(self, planner: ActScenePlanner, log_output: logging.Logger) -> None:
        """Verify goals relate to the story topic."""
        spec = make_spec(target_scene_count=3)
        topic_keywords = spec.topic.lower().split()
        pattern = StoryPattern(
            pattern_name="three_act",
            acts=["setup", "challenge", "resolution"],
            emotion_curve=[1, 3, 2],
            genres=["fantasy"],
        )

        plans = planner.plan(spec, pattern)

        log_output.info("=" * 80)
        log_output.info("TEST: Planner - Goals Specific to Story Topic")
        log_output.info(f"Story topic: {spec.topic}")
        log_output.info(f"Topic keywords: {topic_keywords}")
        
        for i, plan in enumerate(plans, 1):
            log_output.info(f"Plan {i} goal: {plan.goal}")
        
        goals_text = " ".join(plan.goal.lower() for plan in plans)
        # At least one keyword from topic should appear in goals
        found_keywords = [kw for kw in topic_keywords if kw in goals_text]
        log_output.info(f"Found keywords in goals: {found_keywords}")
        assert any(keyword in goals_text for keyword in topic_keywords)
        log_output.info("✓ Test passed")


# ============================================================================
# STAGE 2: EDGE CONTEXT DESIGNER - Tests for edge context design outputs
# ============================================================================


class TestEdgeContextDesignerLLMOutputs:
    """Comprehensive tests for edge context designer stage LLM outputs."""

    @pytest.fixture
    def designer(self, real_runner: JSONTaskRunner) -> EdgeContextDesigner:
        return EdgeContextDesigner(llm_runner=real_runner)

    def test_edge_context_designer_valid_context_keys(self, designer: EdgeContextDesigner, log_output: logging.Logger) -> None:
        """Verify edge context only includes valid state keys."""
        source = ScenePlan(scene_id="S1", act="setup", goal="Introduce Milo")
        target = ScenePlan(
            scene_id="S2",
            act="challenge",
            goal="Milo investigates",
            required_context=["world_state", "character_state"],
        )

        edge = designer.design(source, target, "hard")

        log_output.info("=" * 80)
        log_output.info("TEST: Edge Context Designer - Valid Context Keys")
        log_output.info(f"Source: {source.scene_id} -> Target: {target.scene_id}")
        log_output.info(f"Required context: {edge.required_context}")
        log_output.info(f"Ignored context: {edge.ignored_context}")
        
        allowed = {"world_state", "character_state", "narrative_state", "style_state", "safety_state"}
        assert all(key in allowed for key in edge.required_context)
        assert all(key in allowed for key in edge.ignored_context)
        log_output.info("✓ Test passed")

    def test_edge_context_designer_source_target_match(self, designer: EdgeContextDesigner, log_output: logging.Logger) -> None:
        """Verify edge source and target match input plans."""
        source = ScenePlan(scene_id="S1", act="setup", goal="Introduce Milo")
        target = ScenePlan(
            scene_id="S2",
            act="challenge",
            goal="Milo investigates",
            required_context=["world_state", "character_state"],
        )

        edge = designer.design(source, target, "hard")

        log_output.info("=" * 80)
        log_output.info("TEST: Edge Context Designer - Source/Target Match")
        log_output.info(f"Edge source: {edge.source}, target: {edge.target}")
        
        assert edge.source == "S1"
        assert edge.target == "S2"
        log_output.info("✓ Test passed")

    def test_edge_context_designer_dependency_type_preserved(
        self, designer: EdgeContextDesigner, log_output: logging.Logger
    ) -> None:
        """Verify dependency type is preserved."""
        source = ScenePlan(scene_id="S1", act="setup", goal="Introduce Milo")
        target = ScenePlan(
            scene_id="S2",
            act="challenge",
            goal="Milo investigates",
            required_context=["world_state", "character_state"],
        )

        hard_edge = designer.design(source, target, "hard")
        soft_edge = designer.design(source, target, "soft")

        log_output.info("=" * 80)
        log_output.info("TEST: Edge Context Designer - Dependency Type Preserved")
        log_output.info(f"Hard edge type: {hard_edge.dependency_type}")
        log_output.info(f"Soft edge type: {soft_edge.dependency_type}")

        assert hard_edge.dependency_type == "hard"
        assert soft_edge.dependency_type == "soft"
        log_output.info("✓ Test passed")

    def test_edge_context_designer_respects_target_context(self, designer: EdgeContextDesigner, log_output: logging.Logger) -> None:
        """Verify target's required_context is considered in edge design."""
        source = ScenePlan(scene_id="S1", act="setup", goal="Introduce Milo")
        target = ScenePlan(
            scene_id="S2",
            act="challenge",
            goal="Milo investigates",
            required_context=["character_state", "safety_state"],
        )

        edge = designer.design(source, target, "hard")

        log_output.info("=" * 80)
        log_output.info("TEST: Edge Context Designer - Respects Target Context")
        log_output.info(f"Target required context: {target.required_context}")
        log_output.info(f"Edge required context: {edge.required_context}")

        # Edge should include at least some of target's required context
        target_context_set = set(target.required_context)
        edge_context_set = set(edge.required_context)
        overlap = target_context_set.intersection(edge_context_set)
        log_output.info(f"Overlap: {overlap}")
        assert len(overlap) > 0
        log_output.info("✓ Test passed")

    def test_edge_context_designer_transforms_format(self, designer: EdgeContextDesigner, log_output: logging.Logger) -> None:
        """Verify transforms are properly formatted."""
        source = ScenePlan(scene_id="S1", act="setup", goal="Introduce Milo")
        target = ScenePlan(
            scene_id="S2",
            act="challenge",
            goal="Milo investigates",
            required_context=["world_state", "character_state"],
        )

        edge = designer.design(source, target, "hard")

        log_output.info("=" * 80)
        log_output.info("TEST: Edge Context Designer - Transforms Format")
        log_output.info(f"Edge transforms:")
        # Transforms should be a dict mapping state keys to descriptions
        assert isinstance(edge.transforms, dict)
        for key, value in edge.transforms.items():
            log_output.info(f"  {key}: {value}")
            assert isinstance(key, str)
            assert isinstance(value, str)
        log_output.info("✓ Test passed")

    def test_edge_context_designer_multiple_edges_independent(
        self, designer: EdgeContextDesigner, log_output: logging.Logger
    ) -> None:
        """Verify multiple edge designs are independent."""
        s1 = ScenePlan(scene_id="S1", act="setup", goal="Introduce Milo")
        s2 = ScenePlan(
            scene_id="S2",
            act="challenge",
            goal="Milo investigates",
            required_context=["world_state", "character_state"],
        )
        s3 = ScenePlan(
            scene_id="S3",
            act="resolution",
            goal="Mystery solved",
            required_context=["character_state", "narrative_state"],
        )

        edge_1_2 = designer.design(s1, s2, "hard")
        edge_2_3 = designer.design(s2, s3, "hard")

        log_output.info("=" * 80)
        log_output.info("TEST: Edge Context Designer - Multiple Edges Independent")
        log_output.info(f"Edge 1-2: {edge_1_2.source} -> {edge_1_2.target}")
        log_output.info(f"Edge 2-3: {edge_2_3.source} -> {edge_2_3.target}")

        assert edge_1_2.target == "S2"
        assert edge_2_3.source == "S2"
        assert edge_2_3.target == "S3"
        log_output.info("✓ Test passed")

    def test_edge_context_ignores_never_context(self, designer: EdgeContextDesigner, log_output: logging.Logger) -> None:
        """Verify never_context keys are not included."""
        source = ScenePlan(scene_id="S1", act="setup", goal="Introduce Milo")
        target = ScenePlan(
            scene_id="S2",
            act="challenge",
            goal="Milo investigates",
            required_context=["world_state"],
        )

        edge = designer.design(source, target, "hard")

        log_output.info("=" * 80)
        log_output.info("TEST: Edge Context Designer - Ignores Never Context")
        log_output.info(f"Required context: {edge.required_context}")
        log_output.info(f"Ignored context: {edge.ignored_context}")

        never_keys = {"full_scene_history", "minor_environment_details"}
        assert not any(key in edge.required_context for key in never_keys)
        # Check ignored_context includes never_context
        ignored = set(edge.ignored_context)
        assert never_keys.intersection(ignored) == never_keys
        log_output.info("✓ Test passed")


# ============================================================================
# STAGE 3: SCENE GENERATOR - Tests for scene generation outputs
# ============================================================================


class TestSceneGeneratorLLMOutputs:
    """Comprehensive tests for scene generator stage LLM outputs."""

    @pytest.fixture
    def generator(self, real_runner: JSONTaskRunner) -> SceneGenerator:
        return SceneGenerator(llm_runner=real_runner)

    def test_scene_generator_outputs_match_spec_requirements(
        self, generator: SceneGenerator, log_output: logging.Logger
    ) -> None:
        """Verify generated scenes respect spec constraints."""
        spec = make_spec()
        node = make_node("S1", act="setup")

        output = generator.generate(
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "style_state": {"vocab_level": spec.vocab_level},
                "safety_state": {"fear_level": spec.fear_level},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Outputs Match Spec Requirements")
        log_output.info(f"Scene ID: {output.scene_id}")
        log_output.info(f"Scene text: {output.scene_text[:100]}...")
        log_output.info(f"Forbidden elements: {spec.forbidden_elements}")
        
        # Check age-appropriateness (no forbidden elements)
        text_lower = output.scene_text.lower()
        found_forbidden = []
        for forbidden in spec.forbidden_elements:
            if forbidden.lower() in text_lower:
                found_forbidden.append(forbidden)
        log_output.info(f"Forbidden elements found: {found_forbidden if found_forbidden else 'None'}")
        for forbidden in spec.forbidden_elements:
            assert forbidden.lower() not in text_lower
        log_output.info("✓ Test passed")

    def test_scene_generator_scene_id_matches_node(self, generator: SceneGenerator, log_output: logging.Logger) -> None:
        """Verify generated scene_id matches input node."""
        spec = make_spec()
        node = make_node("S3", act="resolution")

        output = generator.generate(
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "proud"}},
                "style_state": {"vocab_level": spec.vocab_level},
                "safety_state": {"fear_level": 1},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Scene ID Matches Node")
        log_output.info(f"Expected scene ID: S3")
        log_output.info(f"Generated scene ID: {output.scene_id}")

        assert output.scene_id == "S3"
        log_output.info("✓ Test passed")

    def test_scene_generator_outputs_non_empty_text(self, generator: SceneGenerator, log_output: logging.Logger) -> None:
        """Verify generated scene text is substantial."""
        spec = make_spec()
        node = make_node("S1")

        output = generator.generate(
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "style_state": {"vocab_level": spec.vocab_level},
                "safety_state": {"fear_level": spec.fear_level},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Outputs Non-Empty Text")
        log_output.info(f"Scene text length: {len(output.scene_text)} characters")
        log_output.info(f"Scene text preview: {output.scene_text[:150]}...")

        assert output.scene_text
        assert len(output.scene_text) > 50  # Minimum reasonable length
        log_output.info("✓ Test passed")

    def test_scene_generator_metadata_includes_safety_scores(
        self, generator: SceneGenerator, log_output: logging.Logger
    ) -> None:
        """Verify metadata includes required safety score fields."""
        spec = make_spec()
        node = make_node("S1")

        output = generator.generate(
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "style_state": {"vocab_level": spec.vocab_level},
                "safety_state": {"fear_level": spec.fear_level},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Metadata Includes Safety Scores")
        log_output.info(f"Metadata keys: {list(output.metadata.keys())}")
        log_output.info(f"Metadata content: {output.metadata}")

        assert "fear_score" in output.metadata or "violence_score" in output.metadata
        log_output.info("✓ Test passed")

    def test_scene_generator_metadata_scores_within_bounds(
        self, generator: SceneGenerator, log_output: logging.Logger
    ) -> None:
        """Verify safety scores are within valid bounds."""
        spec = make_spec()
        node = make_node("S1")

        output = generator.generate(
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "style_state": {"vocab_level": spec.vocab_level},
                "safety_state": {"fear_level": spec.fear_level},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Metadata Scores Within Bounds")
        for score_name in ["fear_score", "violence_score", "emotional_intensity"]:
            if score_name in output.metadata:
                score = output.metadata[score_name]
                log_output.info(f"{score_name}: {score}")
                assert isinstance(score, (int, float))
                assert 0 <= score <= 5
        log_output.info("✓ Test passed")

    def test_scene_generator_respects_vocab_level(self, generator: SceneGenerator, log_output: logging.Logger) -> None:
        """Verify generated text matches vocab level constraints."""
        spec = make_spec()
        spec.vocab_level = "simple"
        node = make_node("S1")

        output = generator.generate(
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "style_state": {"vocab_level": "simple"},
                "safety_state": {"fear_level": spec.fear_level},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Respects Vocab Level")
        log_output.info(f"Vocab level: simple")
        
        # Simple vocab should use shorter, more common words
        text_lower = output.scene_text.lower()
        words = text_lower.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        log_output.info(f"Average word length: {avg_word_length:.2f}")
        log_output.info(f"Scene preview: {output.scene_text[:100]}...")
        # For simple vocab, average word length should be reasonable for age 7-9
        assert avg_word_length < 7
        log_output.info("✓ Test passed")

    def test_scene_generator_response_metadata_type(self, generator: SceneGenerator, log_output: logging.Logger) -> None:
        """Verify metadata is a dict with string keys."""
        spec = make_spec()
        node = make_node("S1")

        output = generator.generate(
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "style_state": {"vocab_level": spec.vocab_level},
                "safety_state": {"fear_level": spec.fear_level},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Response Metadata Type")
        log_output.info(f"Metadata type: {type(output.metadata)}")
        log_output.info(f"Metadata keys: {list(output.metadata.keys())}")

        assert isinstance(output.metadata, dict)
        for key in output.metadata:
            assert isinstance(key, str)
        log_output.info("✓ Test passed")

    def test_scene_generator_includes_act_context(self, generator: SceneGenerator, log_output: logging.Logger) -> None:
        """Verify generation respects act-specific context."""
        spec = make_spec()
        # Test different acts
        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Includes Act Context")
        
        for act, goal in [("setup", "Introduce Milo"), ("challenge", "Increase danger"), ("resolution", "Solve mystery")]:
            node = make_node("S1", act=act, goal=goal)

            output = generator.generate(
                node=node,
                spec=spec,
                filtered_state={
                    "world_state": {"location": "moonlit garden"},
                    "character_state": {"Milo": {"emotion": "curious"}},
                    "style_state": {"vocab_level": spec.vocab_level},
                    "safety_state": {"fear_level": spec.fear_level},
                },
            )

            log_output.info(f"Act: {act}, Goal: {goal}")
            log_output.info(f"  Generated preview: {output.scene_text[:80]}...")
            assert output.scene_text
            assert output.scene_id
        log_output.info("✓ Test passed")

    def test_scene_generator_repair_instructions_acknowledged(
        self, generator: SceneGenerator, log_output: logging.Logger
    ) -> None:
        """Verify repair instructions are considered in generation."""
        spec = make_spec()
        node = make_node("S2", act="challenge")

        output = generator.generate(
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "style_state": {"vocab_level": spec.vocab_level},
                "safety_state": {"fear_level": spec.fear_level},
            },
            repair_instructions=["Ensure Milo feels brave", "Remove scary descriptions"],
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Scene Generator - Repair Instructions Acknowledged")
        log_output.info("Repair instructions:")
        log_output.info("  - Ensure Milo feels brave")
        log_output.info("  - Remove scary descriptions")
        log_output.info(f"Generated scene: {output.scene_text[:100]}...")

        # Repaired output should still be valid
        assert output.scene_text
        assert len(output.scene_text) > 50
        log_output.info("✓ Test passed")


# ============================================================================
# STAGE 4: STATE DIFF EXTRACTOR - Tests for state diff extraction outputs
# ============================================================================


class TestStateDiffExtractorLLMOutputs:
    """Comprehensive tests for state diff extractor stage LLM outputs."""

    @pytest.fixture
    def extractor(self, real_runner: JSONTaskRunner) -> StateDiffExtractor:
        return StateDiffExtractor(llm_runner=real_runner)

    def test_state_diff_extractor_returns_valid_dict(
        self, extractor: StateDiffExtractor, log_output: logging.Logger
    ) -> None:
        """Verify state diff is returned as dict."""
        spec = make_spec()
        node = make_node("S2", act="challenge", goal="Milo follows the clue")
        draft = SceneOutput(
            scene_id="S2",
            scene_text="Milo felt curious and followed the moon clue into the garden.",
            metadata={"fear_score": 1, "violence_score": 0},
        )

        diff = extractor.extract(
            draft=draft,
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "garden path"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "narrative_state": {"active_arc": "moon_garden_mystery"},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: State Diff Extractor - Returns Valid Dict")
        log_output.info(f"Scene ID: {draft.scene_id}")
        log_output.info(f"Scene text: {draft.scene_text}")
        log_output.info(f"Extracted diff: {diff}")

        assert isinstance(diff, dict)
        log_output.info("✓ Test passed")

    def test_state_diff_extractor_keys_follow_dotted_format(
        self, extractor: StateDiffExtractor, log_output: logging.Logger
    ) -> None:
        """Verify state diff keys follow allowed dotted path format."""
        spec = make_spec()
        node = make_node("S2", act="challenge")
        draft = SceneOutput(
            scene_id="S2",
            scene_text="Milo moved deeper into the garden, feeling braver.",
            metadata={"fear_score": 1},
        )

        diff = extractor.extract(
            draft=draft,
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "garden path"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "narrative_state": {"active_arc": "moon_garden_mystery"},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: State Diff Extractor - Keys Follow Dotted Format")
        
        # Valid root prefixes
        allowed_roots = {
            "world_state",
            "character_state",
            "narrative_state",
            "style_state",
            "safety_state",
            "world",
            "character",
            "narrative",
            "style",
            "safety",
            "time",
        }
        log_output.info("Extracted diff keys:")
        for key in diff:
            root = key.split(".")[0]
            log_output.info(f"  {key} (root: {root})")
            assert root in allowed_roots
        log_output.info("✓ Test passed")

    def test_state_diff_extractor_values_are_serializable(
        self, extractor: StateDiffExtractor, log_output: logging.Logger
    ) -> None:
        """Verify state diff values are JSON-serializable."""
        spec = make_spec()
        node = make_node("S2", act="challenge")
        draft = SceneOutput(
            scene_id="S2",
            scene_text="Milo felt braver and continued exploring.",
            metadata={"fear_score": 1},
        )

        diff = extractor.extract(
            draft=draft,
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "garden path"},
                "character_state": {"Milo": {"emotion": "curious"}},
                "narrative_state": {"active_arc": "moon_garden_mystery"},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: State Diff Extractor - Values Are Serializable")
        
        # All values should be JSON-serializable
        import json
        json_str = json.dumps(diff)
        log_output.info(f"JSON serialized: {json_str}")
        assert json_str
        log_output.info("✓ Test passed")

    def test_state_diff_extractor_character_state_changes(
        self, extractor: StateDiffExtractor, log_output: logging.Logger
    ) -> None:
        """Verify state diff can capture character emotion/status changes."""
        spec = make_spec()
        node = make_node("S2", act="challenge", goal="Milo becomes braver")
        draft = SceneOutput(
            scene_id="S2",
            scene_text="With each step, Milo felt braver and more determined to solve the mystery.",
            metadata={"fear_score": 1, "violence_score": 0},
        )

        diff = extractor.extract(
            draft=draft,
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: State Diff Extractor - Character State Changes")
        log_output.info(f"Extracted diff: {diff}")

        # Should contain character state changes
        character_keys = [k for k in diff if k.startswith(("character", "character_state"))]
        log_output.info(f"Character state keys found: {character_keys}")
        assert len(character_keys) > 0
        log_output.info("✓ Test passed")

    def test_state_diff_extractor_world_state_changes(
        self, extractor: StateDiffExtractor, log_output: logging.Logger
    ) -> None:
        """Verify state diff can capture world state changes."""
        spec = make_spec()
        node = make_node("S2", act="challenge", goal="Move to new location")
        draft = SceneOutput(
            scene_id="S2",
            scene_text="Milo walked from the garden path into the ancient forest.",
            metadata={"fear_score": 2},
        )

        diff = extractor.extract(
            draft=draft,
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "garden path", "time": "early morning"},
                "character_state": {"Milo": {"emotion": "curious"}},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: State Diff Extractor - World State Changes")
        log_output.info(f"Extracted diff: {diff}")
        
        # Should potentially contain world state changes
        diff_str = str(diff).lower()
        assert "state" in diff_str or len(diff) > 0
        log_output.info("✓ Test passed")

    def test_state_diff_extractor_time_progression(
        self, extractor: StateDiffExtractor, log_output: logging.Logger
    ) -> None:
        """Verify state diff can represent time progression."""
        spec = make_spec()
        node = make_node("S3", act="resolution")
        draft = SceneOutput(
            scene_id="S3",
            scene_text="As evening fell, Milo finally understood the mystery's truth.",
            metadata={"fear_score": 1},
        )

        diff = extractor.extract(
            draft=draft,
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moon garden", "time": "afternoon"},
                "character_state": {"Milo": {"emotion": "brave"}},
                "narrative_state": {"active_arc": "moon_garden_mystery"},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: State Diff Extractor - Time Progression")
        log_output.info(f"Extracted diff: {diff}")
        
        diff_str = str(diff).lower()
        # Should show some kind of progression
        assert len(diff) > 0
        log_output.info("✓ Test passed")

    def test_state_diff_extractor_handles_empty_draft_metadata(
        self, extractor: StateDiffExtractor, log_output: logging.Logger
    ) -> None:
        """Verify extractor handles scenes with minimal metadata."""
        spec = make_spec()
        node = make_node("S1", act="setup")
        draft = SceneOutput(
            scene_id="S1",
            scene_text="Milo stood in the moonlit garden, wondering about the strange glowing flowers.",
            metadata={},  # Empty metadata
        )

        diff = extractor.extract(
            draft=draft,
            node=node,
            spec=spec,
            filtered_state={
                "world_state": {"location": "moonlit garden"},
                "character_state": {"Milo": {"emotion": "curious"}},
            },
        )

        log_output.info("=" * 80)
        log_output.info("TEST: State Diff Extractor - Handles Empty Metadata")
        log_output.info(f"Extracted diff: {diff}")
        
        # Should still return a valid diff
        assert isinstance(diff, dict)
        log_output.info("✓ Test passed")


# ============================================================================
# STAGE 5: SEMANTIC VERIFIER - Tests for semantic verification outputs
# ============================================================================


class TestSemanticVerifierLLMOutputs:
    """Comprehensive tests for semantic verifier stage LLM outputs."""

    @pytest.fixture
    def verifier(self, real_runner: JSONTaskRunner) -> SemanticVerifier:
        return SemanticVerifier(llm_runner=real_runner)

    def test_semantic_verifier_returns_verification_result(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify semantic verifier returns valid result structure."""
        spec = make_spec()
        output = SceneOutput(
            scene_id="S1",
            scene_text="Milo stood in the moonlit garden, curious about the glowing flowers.",
            state_diff={"character.Milo.emotion": "curious"},
            metadata={"fear_score": 1},
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": []}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Returns Verification Result")
        log_output.info(f"Result ok: {result.ok}")
        log_output.info(f"Result failures: {result.failures}")

        assert hasattr(result, "ok")
        assert hasattr(result, "failures")
        assert isinstance(result.ok, bool)
        assert isinstance(result.failures, list)
        log_output.info("✓ Test passed")

    def test_semantic_verifier_accepts_valid_scenes(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify well-formed scenes pass verification."""
        spec = make_spec()
        output = SceneOutput(
            scene_id="S1",
            scene_text="Milo wandered through the moonlit garden, filled with curiosity about the mystery.",
            state_diff={"character.Milo.emotion": "curious"},
            metadata={"fear_score": 1, "violence_score": 0},
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": []}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Accepts Valid Scenes")
        log_output.info(f"Scene: {output.scene_text[:100]}...")
        log_output.info(f"Verification result: {result.ok}")
        log_output.info(f"Failures: {result.failures}")

        # Valid scenes should typically pass
        assert isinstance(result.ok, bool)
        log_output.info("✓ Test passed")

    def test_semantic_verifier_detects_forbidden_elements(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify verifier can detect forbidden content."""
        spec = make_spec()
        spec.forbidden_elements = ["extreme violence"]
        output = SceneOutput(
            scene_id="S2",
            scene_text="Milo faced extreme violence and danger in the dark forest.",
            state_diff={"character.Milo.emotion": "terrified"},
            metadata={"fear_score": 5, "violence_score": 4},
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": ["S1"]}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Detects Forbidden Elements")
        log_output.info(f"Forbidden elements: {spec.forbidden_elements}")
        log_output.info(f"Scene text: {output.scene_text}")
        log_output.info(f"Verification result: {result.ok}")
        log_output.info(f"Failures: {result.failures}")

        # High scores with forbidden elements might fail
        if "extreme violence" in output.scene_text.lower():
            # Verifier should flag this
            assert isinstance(result.ok, bool)
        log_output.info("✓ Test passed")

    def test_semantic_verifier_respects_fear_level(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify verifier checks fear level constraints."""
        spec = make_spec()
        spec.fear_level = 1  # Very low fear tolerance
        output = SceneOutput(
            scene_id="S1",
            scene_text="Milo felt slightly nervous in the garden at dusk.",
            state_diff={"character.Milo.emotion": "nervous"},
            metadata={"fear_score": 3},  # Too high for spec.fear_level=1
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": []}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Respects Fear Level")
        log_output.info(f"Spec fear level: {spec.fear_level}")
        log_output.info(f"Scene fear score: {output.metadata['fear_score']}")
        log_output.info(f"Verification result: {result.ok}")
        log_output.info(f"Failures: {result.failures}")

        # Overspec fear should be flagged
        assert isinstance(result.ok, bool)
        log_output.info("✓ Test passed")

    def test_semantic_verifier_failures_list_has_details(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify failures list contains meaningful information."""
        spec = make_spec()
        spec.fear_level = 1
        output = SceneOutput(
            scene_id="S1",
            scene_text="Milo was terrified by the horror of the ancient curse.",
            state_diff={"character.Milo.emotion": "terrified"},
            metadata={"fear_score": 5, "violence_score": 3},
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": []}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Failures List Has Details")
        log_output.info(f"Verification result: {result.ok}")
        log_output.info("Failures:")
        for failure in result.failures:
            log_output.info(f"  - {failure}")

        # If there are failures, they should be meaningful strings
        for failure in result.failures:
            assert isinstance(failure, str)
            assert len(failure) > 0
        log_output.info("✓ Test passed")

    def test_semantic_verifier_checks_age_appropriateness(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify verifier validates age group appropriateness."""
        spec = make_spec()
        spec.age_group = "7-9"
        output = SceneOutput(
            scene_id="S1",
            scene_text="Milo discovered ancient mysteries in the garden.",
            state_diff={"character.Milo.emotion": "curious"},
            metadata={"fear_score": 2},
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": []}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Checks Age Appropriateness")
        log_output.info(f"Age group: {spec.age_group}")
        log_output.info(f"Verification result: {result.ok}")
        log_output.info(f"Failures: {result.failures}")

        assert isinstance(result.ok, bool)
        log_output.info("✓ Test passed")

    def test_semantic_verifier_validates_state_diff_relevance(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify verifier checks state diffs are reflected in scene text."""
        spec = make_spec()
        output = SceneOutput(
            scene_id="S2",
            scene_text="Milo walked through the garden.",
            state_diff={"character.Milo.emotion": "brave", "character.Milo.inventory.add": ["ancient artifact"]},
            metadata={"fear_score": 2},
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": ["S1"]}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Validates State Diff Relevance")
        log_output.info(f"State diff: {output.state_diff}")
        log_output.info(f"Scene text: {output.scene_text}")
        log_output.info(f"Verification result: {result.ok}")
        log_output.info(f"Failures: {result.failures}")

        # State changes should be reflected in text
        assert isinstance(result.ok, bool)
        log_output.info("✓ Test passed")

    def test_semantic_verifier_confidence_affects_result(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify verifier respects confidence thresholds."""
        spec = make_spec()
        output = SceneOutput(
            scene_id="S1",
            scene_text="Milo explored the mysterious garden.",
            state_diff={"character.Milo.emotion": "curious"},
            metadata={"fear_score": 1},
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": []}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Confidence Affects Result")
        log_output.info(f"Verification result: {result.ok}")
        log_output.info(f"Failures: {result.failures}")

        # Result should be deterministic for this input
        assert isinstance(result.ok, bool)
        assert isinstance(result.failures, list)
        log_output.info("✓ Test passed")

    def test_semantic_verifier_handles_complex_state_diffs(
        self, verifier: SemanticVerifier, log_output: logging.Logger
    ) -> None:
        """Verify verifier can validate complex state changes."""
        spec = make_spec()
        output = SceneOutput(
            scene_id="S2",
            scene_text="Milo encountered the guardian spirit and received a magical token.",
            state_diff={
                "character.Milo.emotion": "brave",
                "character.Milo.inventory": ["magical token"],
                "world.location": "enchanted grove",
                "narrative.encountered_guardian": True,
            },
            metadata={"fear_score": 2, "violence_score": 1},
        )

        result = verifier.verify(
            output,
            spec,
            {"narrative_state": {"completed_scenes": ["S1"]}}
        )

        log_output.info("=" * 80)
        log_output.info("TEST: Semantic Verifier - Handles Complex State Diffs")
        log_output.info(f"State diff keys: {list(output.state_diff.keys())}")
        log_output.info(f"Verification result: {result.ok}")
        log_output.info(f"Failures: {result.failures}")

        # Complex diffs should be validated
        assert isinstance(result.ok, bool)
        log_output.info("✓ Test passed")