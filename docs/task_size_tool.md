# Task-to-size guidance tool

`lookup_task_size` returns the user's heuristic parameter-size ranges for 20 task
categories. The packaged catalog deliberately excludes all example model names.
It is available as a NanoClaw MCP tool and a Nemotron function tool in the explicit
recommender research loop. It performs no web requests, starts no job, and saves
no workload state.

Examples:

```json
{"task": "draft an email"}
{"use_case_id": "long_document_summarization"}
```

The first returns `email_rewriting`, 2–4 billion parameters. The second returns
4–9 billion parameters with the original section-level summarization caveat.
Embedding and reranking entries retain their specialized model types.

Matching uses Unicode normalization/casefold and punctuation-to-space conversion.
Exact IDs win; complete descriptive aliases can identify a use case. Multiple
aliases across categories or keyword overlap return `needs_clarification` with
no selected size. The agent can interpret the candidates semantically or ask the
user. There is no embedding retrieval service and no model-name matching.

These are supplied heuristics, not measured capability or deployment guarantees.
Explicit user size requirements take precedence. Hardware, license, context,
quality, and modality checks still need to happen during discovery/evaluation.

Test a live Nemotron tool call without scanning models:

```bash
PYTHONPATH=src python -m hf_model_watch task-size "draft an email"
```

The existing Nebius environment/configuration supplies credentials and the model;
no credential is printed. The command requires one complete `lookup_task_size`
call for the original task. The trusted catalog supplies the range, rather than
accepting generated numbers or an unconfirmed use-case ID from the model.
Ambiguity returns `needs_clarification` and candidate descriptions.

Use that guidance in discovery with the opt-in flag:

```bash
PYTHONPATH=src python -m hf_model_watch recommend "draft an email" \
  --task-size-guidance --gpu-memory-gb 24 --gpu-count 1 \
  --gpu-hourly-cost-usd 0.5 --budget-usd-per-hour 0.7 \
  --license any --framework-mode sweep --speculative-decoding off
```

Discovery enforces the exact catalog min/max range before candidate research.
The run saves `task_sizing`, `size_range`, and `task_size_guidance.json`.
Explicit size requests and explicit model-ID flows skip task-size inference.
Ambiguous tasks stop before scanning; inspect `task-size` output and restate the
task with an explicit size. Embedding/reranker tasks stop rather than recommending
chat models. API and truncated/invalid tool-call failures do not silently remove
the size restriction. The default CLI flow remains available without the flag.
The sizing client uses the configured timeout with no automatic retries.

Validation: 160 focused tests passed before the tool-choice correction; 13 tool
tests passed after it. Forced tool selection timed out. Switching to
`tool_choice="auto"` produced a successful live Nebius `lookup_task_size` call
for `draft an email`, returning `email_rewriting`, 2–4B, with finish reason
`tool_calls`. Expected tool name and unchanged task validation remain enforced.

The combined memory, GPU, research scheduling, metadata, state and reporting
revision also exposes `task_size_guidance` through the NanoClaw adapter. See
`reports/pipeline_revision.md` for deployment, validation, and estimate limitations.
