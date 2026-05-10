"""FastAPI endpoints for story generation service."""

import logging
import time

from fastapi import FastAPI
from fastapi import Request

from story_engine.core.engine import StoryEngine
from story_engine.core.executor import StoryRun
from story_engine.models.story_spec import StoryInput, StorySpec
from story_engine.observability import configure_logging

configure_logging()
logger = logging.getLogger("story_engine.api")
app = FastAPI(title="Novel Bot", version="0.1.0")
engine = StoryEngine()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_completed method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Dictionary with status key.
    """
    logger.debug("health_check")
    return {"status": "ok"}


@app.post("/spec", response_model=StorySpec)
def build_spec(story_input: StoryInput) -> StorySpec:
    """Generate a story specification from user input.

    Args:
        story_input: User input for story generation.

    Returns:
        A StorySpec with detailed generation parameters.
    """
    logger.info("spec_request topic=%r age_group=%s genre=%s", story_input.topic, story_input.age_group, story_input.genre)
    spec = engine.spec_builder.build(story_input)
    logger.info("spec_created topic=%r target_scene_count=%s", spec.topic, spec.target_scene_count)
    return spec


@app.post("/stories", response_model=StoryRun)
def create_story(story_input: StoryInput) -> StoryRun:
    """Generate a complete story from user input.

    Args:
        story_input: User input for story generation.

    Returns:
        A StoryRun containing generated scenes and metadata.
    """
    logger.info("story_request topic=%r age_group=%s genre=%s", story_input.topic, story_input.age_group, story_input.genre)
    run = engine.create_story(story_input)
    logger.info("story_created topic=%r scenes=%s story_words=%s", story_input.topic, len(run.scenes), len(run.story_text.split()))
    return run
