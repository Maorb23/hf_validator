# Phase 1 — CLI Readiness, Selection, and Feedback

**Implementation pass:** 1. **Prerequisite:** Read `phase0_introduction.md` and inspect the actual CLI, recommender, artifacts, persistence, and tests.

## Goal

Make the existing `hf-model-watch` recommendation workflow usable by an external conversational agent **without rebuilding functionality that already works**. The deliverable is an automation-safe recommendation → clarification (if needed) → selection → feedback → revised recommendation path. **Do not integrate OpenClaw or inference optimization in this phase.**

## 1. Inspect before editing

Identify and document the current implementations of:

- The real command/flags producing structured recommendation JSON and their established exit-code behavior.
- Missing-constraint questions and human-facing interactive clarification; how a request resumes after a clarification.
- Existing run IDs, artifact storage, candidate identity, selection/persistence, feedback, and run resumption, if any.
- Model evidence (`verified`, `inferred`, `unknown`), eligibility checks, recommendation ranking, and relevant tests.

If a capability exists, reuse it. If an existing CLI flag or method already supports a required operation, use that interface; do not introduce a duplicate just to match examples below.

## 2. Preserve the discovery contract

Keep the existing fields, their names/types and meanings:

```text
status
 task_request
 constraints
 recommendations
 selected_model_id
 action
 action_status
```

(The indentation above is illustrative; fields remain the existing top-level JSON keys.) Preserve existing statuses, especially `insufficient_evidence`; do not convert it into `needs_input` or pretend there are recommendations. Keep `action: "Optimize Inference + Benchmark"` and `action_status: "placeholder"` non-executable. Do not invent results or change the current ranking/evidence engine.

Add fields such as `run_id`, `parent_run_id`, or structured `questions` **only if they are absent and needed**. Document additive response fields and any required backward-compatibility considerations. Prefer established request, response, and validation types to new duplicate schemas.

## 3. Structured clarification and constraint integrity

Ensure the existing non-interactive recommendation flow can distinguish:

- **`needs_input`:** Required information is genuinely missing or ambiguous; return machine-readable questions with field names, prompt, type, and valid choices/ranges when applicable.
- **`insufficient_evidence`:** Inputs are sufficiently specified but supported recommendations cannot be established. Return the actual empty/limited candidate set and evidence gap, not made-up suggestions.
- **Execution/validation failure:** Return stable structured error information; do not disguise it as a routine clarification.

Preserve previously collected constraints when the user answers a follow-up. Do not require an agent to parse interactive terminal prose or silently drop user-supplied constraints.

**Concrete regression to investigate:** The provided request asks for a *maximum of 2,000 generated tokens*, while the observed `constraints.max_generation_tokens` is `null`. Reproduce against the current CLI, identify whether extraction/mapping/validation is responsible, and fix the existing component if reproducible. Add a regression test. If the intended semantics are genuinely ambiguous, request clarification; do not invent a default or conflate generated-token limits with context length.

## 4. Run identity and durable selection

If the application lacks a durable run ID, introduce one and persist the original request, validated constraints, discovery/evidence reference, and relevant timestamps. Use existing artifact/storage conventions before adding a new database. Ensure a later command can refer to a known run without rerunning discovery unexpectedly.

Expose a **non-interactive selection operation** if absent. Illustrative CLI shape; use actual project conventions:

```bash
hf-model-watch select --run-id rec_001 --model-id org/model --json
```

Selection must:

- Resolve the correct current recommendation run and confirm the chosen model belongs to its candidate set.
- Reject selections when there are no candidates or when the run is stale/invalid.
- Persist the selected model ID and exact revision/evidence reference if available.
- Return a machine-readable result, and handle duplicate or conflicting submissions safely.

Do not ask an agent to edit `discovery.json` or any selection artifact directly.

## 5. Feedback and revised discovery

If absent, expose a non-interactive feedback operation using existing CLI patterns, for example:

```bash
hf-model-watch feedback add \
  --run-id rec_001 --model-id org/model \
  --category out_of_memory --details 'OOM at 32k context' --json
```

Persist feedback ID, associated run/model, source, timestamp, category, original user observation, and optional provided workload/hardware context. Store reported claims with `feedback_status: reported` (or the current equivalent), **not** as independently verified model capability/evidence. Do not manufacture missing telemetry.

Support creating a revised recommendation from a prior run when new or corrected constraints justify it. Record `parent_run_id` and the triggering feedback/explicit revision reason if not already available. Preserve the original run and bound automatic retry attempts; do not enter a repeated search loop for unchanged inputs or inconclusive results.

Do not automatically equate 'slow' with 'wrong model': latency/throughput feedback may be a runtime-configuration issue for the later external optimization API. Model-quality/capability feedback may justify revised discovery after appropriate clarification.

## 6. Automation-safe I/O

Use structured validated request/response data. If the existing CLI supports non-interactive input, keep it. Add a request-file option only if the real integration requires it; do not require an unnecessary new CLI command syntax.

Keep machine JSON on stdout and diagnostics on stderr; document ordinary workflow outcomes versus actual command failures. Enforce bounded input size and safe run/artifact path handling. Keep the interactive user flow and other existing CLI consumers functioning.

## Phase 1 tests and acceptance criteria

- Existing discovery and interactive CLI tests remain green; the original JSON fields retain their meanings.
- Structured recommendation returns real recommendations **or** a valid `needs_input` / `insufficient_evidence` state without fabricating results.
- The 2,000-generated-token case is reproduced and either correctly extracted or explicitly clarified, with a regression test.
- A separate non-interactive caller can clarify a run, select a candidate from that run, persist feedback, and request a traceable revision (when warranted).
- Invalid/stale model IDs, missing runs, duplicate conflicting actions, and malformed input fail predictably.
- The optimization placeholder remains non-executable. No OpenClaw or optimizer/API implementation is added.

**Deliverable:** Committed Phase 1 changes plus test results and a short inventory of what was reused vs. implemented. Stop before Phase 2.
