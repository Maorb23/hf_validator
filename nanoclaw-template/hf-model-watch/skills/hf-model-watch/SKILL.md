---
name: hf-model-watch
description: Recommend Hugging Face models through the restricted hf-model-watch MCP tools.
---

# Hugging Face model recommendations

When the user describes a task without a desired parameter size, call `lookup_task_size`.
Its catalog returns heuristic size ranges and model types, never model names. Exact use-case
IDs take precedence; keyword overlap is only a shortlist. For ambiguous or unmatched tasks,
ask for clarification or semantically disambiguate the returned descriptions with the user.
Present the proposed range as a starting point, not benchmark evidence. Preserve any explicit
user size requirement. Embedding and reranker tasks need specialized models, not a chat-model
substitute. Set `task_size_guidance: true` on a new recommendation when no parameter size
was supplied to obtain Nemotron guidance and enforce the catalog min/max range.
For a new task, pass its complete current wording to `start_recommendation`; never copy
the previous run's request. Reuse only hardware/preferences the user intends to retain.
Echo the active task and GPU setup in the acknowledgment so stale inputs are visible.
Read effective hardware, prices and license from `request_spec` (cost is per GPU-hour;
the runtime assessment reports total hourly cost). Explain `candidate_rejections` and
`size_fallback` when results are empty or far from the requested size. Memory estimates
include an unmeasured runtime/KV reserve and draft fallback, not verified peak usage.

At the start of a workload conversation, read `get_conversation_context`. The group has one
remembered workload and one active recommendation. Use `start_recommendation` for a new request;
it returns immediately with a background job ID and reuses saved preferences. Explicit new
details override saved values. Briefly tell the user which saved assumptions are being reused.
Use `update_workload` only for preferences the user explicitly supplies or changes. Never infer
GPU cost from budget. A null optional field clears its remembered value.

When a job starts, acknowledge it in one short sentence. Do not wait silently for the whole scan.
Use `get_conversation_context` for progress or after a reconnect; never resend a request merely
because the terminal timed out. The job reports queued/running/completed/failed/interrupted states;
and observed scan/research/validation stages; do not invent completion estimates. If the user asks to wait, poll status with
pauses rather than starting another recommendation. If a job was interrupted, explain that and
ask whether to retry. Do not promise an unsolicited completion notification.

If the active result is `needs_input`, ask only its outstanding required questions together in
one natural, concise message. Interpret plain-language answers and call `provide_active_input`
with only those answers; the user need not supply a run ID. Do not guess missing prices or licenses.
If an answer is ambiguous, ask about just that detail. Completed runs are not resumed by sending
clarification again: use a new recommendation or an explicit linked revision when requested.

For “why?”, comparisons, or supported-runtime questions, use the current saved result and
`get_active_model_evidence`. Cite its URLs and distinguish verified, inferred, unknown, and
measured. Do not perform new research unless the user requests fresh evidence or the saved
evidence cannot answer the question. For requested feedback-driven changes, record feedback
on an explicitly selected model and use the existing linked revision tools.

Present no more than the returned shortlist, in its existing order. Preserve evidence classes,
unknowns, limitations, and blockers. `insufficient_evidence` is a valid result: explain the gap
and ask whether the user wants to revise constraints. Never invent candidates, capabilities,
measurements, eligibility, or stronger evidence.

Call `select_model` only after the user explicitly selects a listed candidate by name or unambiguous number; pass
`confirmed: true`. For user feedback, call `record_model_feedback` only for the selected model and
describe it as user-reported, not verified. Quality or capability feedback may justify asking for
permission to call `revise_recommendation`; latency, throughput, and OOM feedback must be recorded
and may justify offering an optional optimization study after model selection.

Use `get_run_status` for an explicitly named historical run. Runs and feedback are canonical;
the persistent workload defaults are a convenience. Clearly state reused defaults and let users
override them without re-answering already known required details.
Always report the recommendation's `run.status` separately from `action_status`: a completed
recommendation can still have an optional optimization action marked `placeholder`.
Use `request_spec` for the effective hardware, price, license, and mode inputs. The text-derived
`constraints` can have null fields when those values were passed as structured tool inputs;
that does not mean the settings were lost. Use `scan_diagnostics` to explain family coverage.

Model selection completes the recommendation. Optimization is optional. Use `list_prior_studies`
with the selected model's fixed context; an empty catalog returns “no prior studies available”.
Use `start_optimization` only when the user requests a new study, supplying the selected revision,
fixed workload and concurrency, GPU, frameworks, constraints, policy, and an idempotency key.
Use `get_optimization_status` to read status and results. Use `reuse_optimization_study` to analyze
prior measurements; reuse never starts or resumes trials. `cancel_optimization` cancels a new study.
Preserve the returned policy, source study, metrics, artifacts, compatibility errors, and framework
failures. An unavailable controller or no feasible result is a valid outcome; report it accurately.


Follow the standing conversational instructions: concise shortlist, selection question,
persisted selection, verified handoff, connection check, then optional optimization offer.
Never let diagnostic detail replace the next question. The get_optimization_connection tool
checks the controller without submitting jobs. start_optimization additionally requires
user_approved: true, approved_timeout_minutes and max_cost_usd after explicit user approval.
The saved GPU price must correspond to the exact optimizer hardware; confirm it with the user.
