# Agentic Storytelling System MVP

A stateful narrative execution engine for children's stories.

The engine treats structured state as the source of truth. It plans a story as scenes, converts them into a dependency DAG, executes nodes in topological order, injects only selected context into each scene, verifies each node incrementally, and commits only validated state diffs.

## Run

```bash
pip install -e ".[dev]"
uvicorn story_engine.api.main:app --reload
```

Logs are written to `logs/` by default:

- `logs/app.log`: general application events
- `logs/api.log`: HTTP request and API endpoint events
- `logs/engine.log`: planning, DAG, execution, repair, state commit events
- `logs/llm.log`: role-based model calls, retries, parse failures
- `logs/verification.log`: hard and semantic verification results
- `logs/error.log`: warnings and errors across the system

You can change the log directory and level with environment variables:

```bash
STORY_ENGINE_LOG_DIR=./logs STORY_ENGINE_LOG_LEVEL=INFO uvicorn story_engine.api.main:app --reload
```

Generate a story:

```bash
curl -X POST http://127.0.0.1:8000/stories \
  -H "content-type: application/json" \
  -d '{"topic":"dragon friendship","age_group":"7-9","genre":"fantasy","fear_level":2,"length":"short"}'
```

## Test

```bash
pytest
```

## Architecture

- `story_engine/core`: orchestration, planning, DAG construction, state, repair, assembly
- `story_engine/generation`: scene generation and prompt construction
- `story_engine/verification`: deterministic validators and aggregate verification
- `story_engine/models`: typed contracts
- `story_engine/retrieval/pattern_library`: reusable narrative patterns, not stories
- `story_engine/prompts`: compact prompt templates
