"""Scene generation and prompt building.

Handles LLM-based and deterministic scene generation with prompt
building and style selection.
"""

from story_engine.generation.scene_generator import SceneGenerator
from story_engine.generation.state_diff_extractor import StateDiffExtractor

__all__ = ["SceneGenerator", "StateDiffExtractor"]
