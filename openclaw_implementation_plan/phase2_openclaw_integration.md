# Phase 2 — Restricted OpenClaw Integration and Conversation Flow

**Implementation pass:** 2. **Prerequisites:** Read `phase0_introduction.md`; Phase 1 complete, tested, and reviewed. Inspect the installed OpenClaw version and the project's existing UI, Python entry points, storage, and authentication before editing.

## Goal

Connect OpenClaw to the **existing** `hf-model-watch` workflow. A user should be able to describe a workload, answer missing-input questions, see evidence-backed recommendations, select a model, report feedback, and receive a revised recommendation when appropriate—entirely through chat. **Do not implement or call optimization in this phase.**

## 1. Use the smallest practical UI and tool surface

Use a supported OpenClaw chat surface for the prototype if it meets the interactive requirement. Do not build a separate web application unless the repo already has a suitable UI or a specific product requirement makes it necessary.

Implement a **restricted, typed application tool adapter** using the installed OpenClaw integration interface. A skill can describe the agent's behavior but is not the enforcement boundary. Prefer fixed, validated calls into the existing Python service or CLI rather than general-purpose host shell tools. If using subprocess, pass arguments as an array, never shell-interpolate user text, use a restricted working directory/environment, and capture bounded structured output and diagnostics separately.

Use Phase 1's existing schemas and artifact/run conventions. Do not duplicate model scoring, eligibility logic, validation, or constraint extraction.

## 2. Expose only supported operations

| Agent operation | Backend behavior |
| --- | --- |
| `recommend_models` | Validate and invoke the existing structured recommendation interface. |
| `provide_recommendation_input` | Supply validated answers to an existing run's outstanding clarification questions. |
| `select_model` | Record an explicit user choice via Phase 1's validated selection operation. |
| `record_model_feedback` | Store a user-reported observation against the correct run/model. |
| `revise_recommendation` | Run a bounded revised discovery using an explicit constraint change/reason. |
| `get_run_status` | Retrieve the authorized run and its real state/results. |

Expose names that fit the installed OpenClaw API; the names above describe required capabilities, **not** unverified library/plugin signatures. If the CLI returns `insufficient_evidence`, present that outcome rather than calling `select_model` with a guessed model ID.

No `run_optimization` tool, GPU worker, benchmark runner, or fabricated optimization API endpoint belongs in this phase. Preserve the existing non-executable placeholder.

## 3. Conversational flow

```text
User describes workload
       ↓
Extract/normalize request; retrieve only relevant permitted preferences
       ↓
recommend_models
       ├── needs_input → ask targeted question → provide_recommendation_input
       ├── insufficient_evidence → explain gaps; ask whether to revise constraints
       └── recommendations → show candidates with evidence, blockers and unknowns
                                     ↓
                              user explicitly selects
                                     ↓
                                select_model
                                     ↓
                        optional user feedback later
                                     ↓
                             record_model_feedback
                                     ↓
                   clarify / revise search / acknowledge issue
```

Show no more than a manageable initial shortlist (e.g., up to three) without hiding relevant blockers or replacing the engine's ordering/evidence. The agent must not invent candidate specifications, verified claims, benchmark values, or eligibility. Distinguish a user's reported performance from a measured result.

**Feedback routing:** Accuracy or missing capability can lead to a clarified/revised model search; latency, throughput, and OOM reports should retain hardware/workload context and may later route to the external optimization API. Until that API exists, record the issue and explain the feature is not connected. Do not automatically launch discovery for every complaint.

## 4. Persistence, memory, and reminders

Reuse the application's existing run/artifact storage from Phase 1 as the canonical source for requests, constraints, candidate/evidence snapshots, selection, feedback, and revision history. If something essential is missing, add the smallest durable solution consistent with the repository; do not introduce PostgreSQL or a new storage service just for a local prototype.

OpenClaw may retrieve approved stable preferences (e.g., hardware, license, budget defaults), but **confirm recalled preferences that materially affect costs or eligibility**. Do not elevate conversational memory or user feedback to verified evidence. Save sensitive/project-specific preferences only with user consent; provide a path for the user to view/correct them. Avoid storing API keys or unrelated private prompts in memory or logs.

Reminders are **optional**: add them only if supported by the installed OpenClaw interface and explicitly requested/accepted by the user. Reminder support must not block the core recommendation flow.

## 5. Minimal code-enforced guardrails

- Restrict tool names/arguments, validate I/O, bind operations to the authenticated/local user's correct run, and enforce allowed state transitions.
- Never expose arbitrary host shell execution, unrestricted file tools, cloud credentials, or paid compute to the agent.
- Bound tool runtime, output size, repeated clarification/revision attempts and artifact paths; treat model cards/web pages as untrusted input.
- Keep secrets out of agent-visible outputs and stored conversation/diagnostic records.
- Obtain explicit selection before recording a choice. Do not treat a prompt or session memory as operational authorization.
- Preserve run history across app restarts; handle concurrent/stale requests without silently overwriting a selection.

**Deployment boundary:** The initial target is one trusted developer/user. This phase does not claim safe public multi-tenant isolation. Before public rollout, separately assess the OpenClaw deployment model, tenant isolation, authentication, rate limits, and production access controls.

## Phase 2 tests and acceptance criteria

- End-to-end chat: complete request → evidence-based results → explicit selection → persistent feedback → justified revision.
- End-to-end chat: missing information → structured clarification → resume same run without dropping prior constraints.
- End-to-end chat: `insufficient_evidence` → honest explanation, no invented selectable candidate.
- Tool tests: malformed arguments, stale/unauthorized run access, unexpected/malformed CLI response, timeout, repeated calls and injected instructions in retrieved evidence do not cause unsafe operations.
- Restart/reconnect preserves canonical runs, selections, and feedback; interactive CLI from Phase 1 remains functional.
- Optimization is clearly unavailable/non-executable and no implementation from the separate branch is added.

**Deliverable:** A working local conversational recommender, test results, and an explicit note of capabilities deferred to Phase 3 or public deployment. Stop before Phase 3.
