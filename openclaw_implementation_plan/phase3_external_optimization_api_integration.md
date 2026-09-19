# Phase 3 — Connect the Ready External Inference-Optimization API

**Implementation pass:** 3 (later). **Prerequisites:** Read `phase0_introduction.md`; Phase 2 working; the **separate optimization branch** provides a ready, accessible, documented API and agreed contract.

## Goal and non-negotiable boundary

Add **only a thin integration** from the conversational application to the existing inference optimization API. The external optimization system remains authoritative for optimization, inference configuration search, GPU execution, job management, benchmarks, and measurements.

**Do not implement, refactor, copy, or merge optimizer logic into this branch.** Do not build GPU workers, scheduling/queue infrastructure, benchmark runners, search algorithms, quantization routines, or optimizer-specific CLI commands here. Do not guess endpoints or invent job/status/result semantics.

If the API is not ready, **do not implement this phase**. Keep the existing `action: "Optimize Inference + Benchmark"` / `action_status: "placeholder"` non-executable, retain all Phase 1–2 recommendation/feedback behavior, and document the missing contract rather than scaffolding fake results.

## 1. Inspect and document the actual API contract

Before writing the client, read the real API specification or code from the other branch and determine:

- Transport/base URL, authentication, deployment access, and existing client SDK if any.
- Required model identity/revision and how it accepts hardware, workload, quality, budget, and other constraints.
- Synchronous response vs. asynchronous job submission; actual returned identifiers and available status/result/cancel operations, **if supported**.
- Actual response/error schemas, units, timeout behavior, request limits, cost/resource exposure, and idempotency/retry semantics.
- Whether the API or caller is responsible for approval, cost caps, and authorization; do not assume either automatically handles the other's responsibilities.

Only expose a capability if the real API supports it. Record the exact source of the contract in the implementation notes.

## 2. Implement a small validated client and approved call path

Reuse the existing app's selected model, run ID, confirmed workload, and user identity. Translate these to the API's **actual** request schema; validate required inputs and report missing fields to the user.

Before invoking a resource-intensive or paid operation, present the actual request and known cost/resource implications and obtain explicit user approval. Bind approval to the user and specific run/model/request, and prevent duplicate submissions. Do not claim a guaranteed price or duration based on unverified estimates.

Call the existing API using the documented authentication mechanism. Persist the API response/reference and link it to the original recommendation run, selected model/revision and approved request. If the API responds synchronously, show that result. If it is asynchronous, use **only documented** status/result/cancellation operations; do not build your own job manager to replace missing functionality.

Expose an OpenClaw tool only for the supported client operations, with the same restricted adapter and application-side authorization from Phase 2. An agent must not fabricate an approval or choose a different model/configuration after approval.

## 3. Present results and receive feedback

Present only actual returned optimizer configuration, measurements, units, hardware/workload context, caveats, and error states. Save the original response and provenance; distinguish external measured results from user-reported observations. Do not mark a reported claim reproduced unless the returned measurement supports that particular claim under matching conditions.

Accept subsequent feedback through the existing Phase 1–2 mechanism. It may lead to user-approved further use of **supported** optimization API operations or a revised model recommendation. Do not let the agent silently start new compute jobs.

## Phase 3 tests and acceptance criteria

- Client contract tests against the real API schema or a mock derived from that documented schema; no made-up endpoints/statuses.
- An explicitly selected model and approved request reach the real API and return/persist an attributable response.
- Unsupported operations, unavailable API, invalid inputs, unauthorized/unapproved calls, errors, and duplicate submissions are handled safely.
- Both sync and async behavior are implemented **only as applicable to the actual contract**; test only supported paths.
- Phase 1–2 discovery, chat, selection, and feedback continue to work independently when optimization is unavailable.
- There are **no changes to optimization algorithms, GPU workers, benchmark implementation, or the separate optimization branch**.

**Deliverable:** Thin documented API adapter, chat presentation of actual results, persistent linkage to recommendation runs, and focused integration-test results.
