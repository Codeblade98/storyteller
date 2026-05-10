# Story Generation Pipeline - Complete Guide

## Overview

This document outlines the entire process of the story generation pipeline, detailing the functions, files, and data flows involved.

---

## **1. Entry Point: API Layer**
**File**: [story_engine/api/main.py](story_engine/api/main.py#L39)

- **Endpoint**: `POST /stories` receives a `StoryInput` object
- **StoryInput** contains: `topic`, `age_group`, `genre`, `fear_level`, `length`
- The `StoryEngine` class orchestrates the entire pipeline

---

## **2. Phase 1: Specification Building**
**File**: [story_engine/core/spec_builder.py](story_engine/core/spec_builder.py)

**Function**: `StorySpecBuilder.build(story_input)` → `StorySpec`

**What it does:**
- Converts high-level user input into detailed generation constraints
- Determines vocabulary level and sentence complexity based on age group
- Infers moral theme from topic
- Sets forbidden elements (death, gore, violence, cruel punishment)
- Calculates target scene count (6 for short, 8 for medium)

**Output**: `StorySpec` object with all constraints

---

## **3. Phase 2: Pattern Selection**
**File**: [story_engine/retrieval/patterns.py](story_engine/retrieval/patterns.py)

**Function**: `PatternLibrary.select(genre, target_scene_count)` → `StoryPattern`

**What it does:**
- Loads story patterns from [pattern_library/discovery_journey.json](story_engine/retrieval/pattern_library/discovery_journey.json)
- Selects a pattern matching the genre and scene count
- Pattern defines narrative structure (acts like setup, challenge, climax, resolution)

**Output**: `StoryPattern` with acts and emotional curve

---

## **4. Phase 3: Scene Planning**
**File**: [story_engine/core/planner.py](story_engine/core/planner.py)

**Function**: `ActScenePlanner.plan(spec, pattern)` → `List[ScenePlan]`

**What it does:**
- **LLM path** (if available): Calls LLM with planner role to generate detailed scene plans
- **Fallback path**: Creates sequential plans with linear dependencies (S1→S2→S3...)
- Each `ScenePlan` includes:
  - `scene_id` (S1, S2, etc.)
  - `goal` (what should happen)
  - `act` (narrative structure)
  - `hard_dependencies` (must complete first)
  - `soft_dependencies` (should ideally complete first)
  - `required_context` (which state keys needed)
  - `constraints` (age-appropriate, avoid forbidden elements)

**Output**: Ordered list of `ScenePlan` objects

---

## **5. Phase 4: DAG Construction**
**File**: [story_engine/core/dag_builder.py](story_engine/core/dag_builder.py)

**Function**: `SceneDAGBuilder.build(scene_plans)` → `SceneDAG`

**What it does:**
- Converts linear scene plans into nodes
- Adds edges (dependencies) between nodes
- Calls `EdgeContextDesigner` to determine what state context flows through each edge
- Creates a directed acyclic graph (DAG) of scenes and their dependencies

**Output**: `SceneDAG` with:
- `nodes`: Dict of SceneNode objects
- `edges`: List of EdgeContext objects with required_context and ignored_context

---

## **6. Phase 5: Topological Execution** (Main Loop)
**File**: [story_engine/core/executor.py](story_engine/core/executor.py)

**Function**: `TopologicalExecutor.execute(spec, dag)` → `StoryRun`

**What it does:**
- Orders nodes via `dag.topological_order()` (uses NetworkX or fallback Kahn's algorithm)
- **For each scene node:**

### **6a. State Filtering**
**File**: [story_engine/core/propagator.py](story_engine/core/propagator.py)

`StatePropagator.filter_for_node(state, node)` → `filtered_state`
- Extracts only the state keys the scene needs (from node.plan.required_context + edges)
- Ignores context marked as ignored_context in edges

### **6b. Generation & Repair Loop** 
**File**: [story_engine/core/repair.py](story_engine/core/repair.py)

`RepairLoop.generate_valid_scene(node, spec, filtered_state, global_state)`

**Inner loop (up to max_attempts=2):**

#### **6b-i. Scene Generation**
**File**: [story_engine/generation/scene_generator.py](story_engine/generation/scene_generator.py)

`SceneGenerator.generate(node, spec, filtered_state, repair_instructions, attempt)`
- **LLM path**: Calls LLM with generator role using scene_generation.yaml template
  - Passes scene goal, act, constraints, filtered_state
  - Temperature increases with attempt number (retry with higher temperature)
- **Fallback**: Deterministic placeholder generation
- **Output**: `SceneOutput` with scene_text and metadata

#### **6b-ii. State Diff Extraction**
**File**: [story_engine/generation/state_diff_extractor.py](story_engine/generation/state_diff_extractor.py)

`StateDiffExtractor.extract(draft, node, spec, filtered_state)` → `state_diff`
- **LLM path**: Calls LLM with extractor role to parse what state changed
  - Template: diff_extraction_prompt.yaml
  - Requires confidence ≥ 0.35 to use LLM result
- **Fallback**: Deterministic extraction based on scene type
- **Output**: Dict with state changes (e.g., `{"character.Milo.emotion": "curious"}`)

#### **6b-iii. Verification**
**File**: [story_engine/verification/verifier.py](story_engine/verification/verifier.py)

`IncrementalVerifier.verify(output, spec, state)` → `VerificationResult`

**Hard constraints** (deterministic):
- `SafetyValidator`: Checks for forbidden elements
- `StyleValidator`: Validates vocabulary/sentence complexity
- `CharacterValidator`: Checks character consistency
- `ChronologyValidator`: Ensures timeline makes sense
- `InvariantValidator`: Checks structural invariants
- `StateDiffValidator`: Validates state changes are valid

**Soft constraints** (optional semantic):
- `SemanticVerifier`: LLM-based checks for narrative coherence

If verification **fails** and **attempts remain**: Loop back with repair_instructions
If verification **passes**: Continue to state application

### **6c. State Application**
**File**: [story_engine/core/state_manager.py](story_engine/core/state_manager.py)

`StateManager.apply_diff(state, state_diff, scene_id)` → `new_state`
- Applies the state_diff to the global state
- Updates completed_scenes list
- Accumulates all changes for next scenes

---

## **7. Phase 6: Assembly**
**File**: [story_engine/core/assembler.py](story_engine/core/assembler.py)

`StoryAssembler.assemble(scenes)` → `story_text`
- Joins all scene texts with double newlines
- Creates final narrative string

---

## **8. Return to API**
**File**: [story_engine/api/main.py](story_engine/api/main.py#L68)

Returns `StoryRun` containing:
- `spec`: StorySpec used
- `scenes`: List of SceneOutput objects
- `final_state`: Complete narrative state after all scenes
- `story_text`: Complete assembled story
- `verification_log`: Success/failure logs for each scene

---

## Key Data Flows

| Stage | Input | Processing | Output |
|-------|-------|-----------|--------|
| Specification | StoryInput | Age-based rules | StorySpec |
| Planning | StorySpec + Pattern | LLM or deterministic | List[ScenePlan] |
| DAG | ScenePlans | Graph construction | SceneDAG |
| Per-Scene | (All previous) | Generate → Extract → Verify → Apply | state updated |
| Assembly | All scenes | Join text | Final story |

---

## Fallback Mechanisms

The system has **three levels of fallback**:

1. **Spec Builder**: Always deterministic (no fallback needed)
2. **Planner**: LLM → Deterministic sequential plans
3. **Generator**: LLM direct → LLM with role routing → Deterministic placeholder
4. **State Diff Extractor**: LLM → Deterministic based on scene type
5. **Verifier**: Hard deterministic checks + optional semantic LLM checks
6. **DAG Ordering**: NetworkX → Kahn's algorithm fallback

If any stage fails completely, `RuntimeError` is raised with the details.