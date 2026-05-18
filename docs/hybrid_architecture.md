# Hybrid Narrative Execution Architecture

This system is a symbolic execution engine with probabilistic narrative modules.
Structured state remains the source of truth. LLMs may propose semantic content,
but deterministic components validate, route, retry, and commit.

## Subsystem Classification

| Subsystem | Current Role | Ideal Class | Boundary |
| --- | --- | --- | --- |
| Story spec builder | Converts user input into age/safety controls | Deterministic | Rule-based normalization and hard age/safety limits. |
| Pattern retrieval | Selects reusable pacing structures | Deterministic | Retrieves abstract patterns only, never story text. |
| Act/scene planner | Two-step act and per-act scene planning | Hybrid | First LLM call proposes only acts, summaries, and act dependencies; each follow-up call expands one act into scenes. Code enforces scene count, IDs, dependency legality, and constraints. |
| DAG builder | Builds and validates graph | Deterministic | No LLM involvement. Graph validity is symbolic. |
| Edge context design | Currently copied from scene plan | Hybrid | LLM may propose edge filters from source/target scene names and goals; code validates against allowed state keys and falls back safely. |
| State propagation | Filters state for a node | Deterministic | Applies validated edge filters and typed state injection. |
| Scene generation | Prose and state diffs mixed together | LLM-driven for prose, deterministic/hybrid for diffs | Generator drafts scene text only. Diff extractor proposes candidate diffs. Diff validator gates commits. |
| State manager | Merges dotted diffs | Deterministic | Never lets an LLM mutate global state directly. |
| Hard verification | Safety keywords, chronology, schema, invalid transitions | Deterministic | Code-only and always run first. |
| Soft verification | Theme, emotion continuity, pacing quality | Hybrid | Deterministic precheck first; optional LLM semantic check only when useful. |
| Repair loop | Local retries | Deterministic orchestration | Bounded retries, stricter prompts, lower temperature, no whole-story regeneration. |
| Parser/sanitizer | JSON cleanup and validation | Deterministic | Never trust raw model output. |
| LLM client | Single endpoint abstraction | Hybrid infrastructure | Role-specific clients for planner, generator, extractor, verifier. |
| Storage/API/assembly | Persistence and presentation | Deterministic | No LLM decisions. |

## Boundary Questions

Symbolic information:
- Scene IDs, graph dependencies, state keys, state diffs, age limits, forbidden elements, retry counts, completed scene order.

Semantic information:
- Scene premise quality, emotional continuity, thematic fit, pacing feel, dialogue naturalness.

Requires creativity:
- Act ideas, scene goals, prose, dialogue, atmosphere, style adaptation.

Requires strict correctness:
- DAG validity, schema shape, state path legality, inventory/location transitions, chronology, safety hard limits.

Can fail probabilistically:
- LLM JSON formatting, semantic judgments, generated prose quality, extracted diffs from prose.

Must remain deterministic:
- State commits, graph execution, propagation mechanics, hard verification, retry orchestration, parsing, schema validation.

## Refactor Plan

1. Add role-based model routing so planning, generation, extraction, and verification can use different endpoints.
2. Add a reusable JSON task runner that builds prompts, calls the role model, sanitizes, parses, validates, and retries.
3. Move prompts into role-specific templates: planner, edge context, scene, diff extraction, semantic verifier, repair.
4. Make planning LLM-driven when a planner model exists, split into act planning and per-act scene planning, with deterministic fallback and deterministic normalization.
5. Make edge filters LLM-proposed when a planner model exists, with deterministic allowed-key validation and fallback.
6. Split scene generation into draft prose generation and state diff extraction.
7. Add deterministic state diff validation before any state merge.
8. Split verifiers into hard deterministic checks and optional semantic verifiers.
9. Keep tests runnable without external services by using deterministic fallbacks.
