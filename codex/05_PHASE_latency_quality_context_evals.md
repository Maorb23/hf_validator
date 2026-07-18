# Codex Phase 5 — latency, quality, long-context drift, and token-normalized metrics

Implement only Phase 5.

Assume Phases 1–4 are complete and tests pass.

Goal: implement deterministic latency benchmarking, quality benchmarking, long-context quality drift benchmarking, and token-normalized usage metrics.

Do not implement fine-tuning, Nebius, or Prefect yet.

Create/modify:

```text
src/hf_model_watch/evals/prompts.py
src/hf_model_watch/evals/latency_bench.py
src/hf_model_watch/evals/quality_eval.py
src/hf_model_watch/evals/context_drift.py
src/hf_model_watch/evals/judge.py
src/hf_model_watch/evals/metrics.py

tests/test_latency_metrics.py
tests/test_quality_metrics.py
tests/test_context_drift.py
tests/test_token_usage_metrics.py
```

## Latency benchmark

Use realistic task categories:

* short chat answer
* summarization
* structured JSON extraction
* classification-by-generation
* RAG-style answer from supplied context
* Text-to-SQL prompt
* code generation
* long-context rewrite/synthesis

Measure:

* cold load time if available
* warmup generation time
* first-token latency when possible
* end-to-end latency
* decode tokens/sec
* prompt tokens
* generated tokens
* batch size
* max_new_tokens
* GPU peak memory if available
* backend
* dtype
* quantization
* failure category

Default generation config:

```yaml
temperature: 0.0
top_p: 1.0
do_sample: false
max_new_tokens_short: 128
max_new_tokens_long: 512
```

## Quality benchmark

Implement a small fixed suite:

* JSON schema adherence
* concise factual QA from supplied context
* Text-to-SQL mini eval
* financial analyst-style answer
* instruction following
* refusal/caveat behavior for investment-advice-like prompts

Use rule-based metrics first:

* exact JSON parse pass
* schema compliance
* SQL syntax pass
* simple exact/substring checks
* length/format checks
* caveat presence
* no paid API judge

`judge.py` may include an optional local model judge hook, disabled by default.

Every quality result should include:

* task_type
* prompt_id
* prompt_family
* difficulty_bucket
* quality_score
* success boolean
* failure_category if applicable

## Long-context quality drift

Implement synthetic tests:

* short context
* medium context
* long context
* near claimed context limit, capped by config

Test types:

* needle retrieval
* summarization consistency
* answer from early/middle/late context
* contradiction detection
* instruction retention

Record:

```text
context_tokens
claimed_context
actual_context_tested
success_rate
drift_score
failure_mode
```

## Token usage and token-normalized metrics

In `latency_bench.py`, `quality_eval.py`, and `metrics.py`, record token-normalized latency and quality metrics.

For every generation request, store:

* provider
* router_name
* model_id
* backend
* task_type
* prompt_id
* prompt_family
* difficulty_bucket
* prompt_tokens_provider
* completion_tokens_provider
* total_tokens_provider
* prompt_tokens_estimated
* completion_tokens_estimated
* total_tokens_estimated
* cached_prompt_tokens, if available
* reasoning_tokens, if available
* ttft_ms
* total_latency_ms
* decode_latency_ms
* router_overhead_ms, if measurable
* queue_time_ms, if provider exposes it
* decode_tokens_per_second
* ms_per_output_token
* ttft_per_input_token
* ttft_per_1k_input_tokens
* latency_per_1k_input_tokens
* latency_per_1k_output_tokens
* decode_latency_per_1k_output_tokens
* quality_score, when available
* quality_per_1k_tokens, when quality_score is available
* cost_estimate_usd, if pricing config is available
* usage_source: `provider_reported`, `estimated`, or `missing`

Definitions:

```text
effective_input_tokens =
  prompt_tokens_provider if available
  else prompt_tokens_estimated

effective_output_tokens =
  completion_tokens_provider if available
  else completion_tokens_estimated

effective_total_tokens =
  total_tokens_provider if available
  else total_tokens_estimated
```

Latency formulas:

```text
decode_latency_ms =
  max(total_latency_ms - ttft_ms, 0)

ttft_per_input_token =
  ttft_ms / max(effective_input_tokens, 1)

ttft_per_1k_input_tokens =
  1000 * ttft_ms / max(effective_input_tokens, 1)

latency_per_1k_input_tokens =
  1000 * total_latency_ms / max(effective_input_tokens, 1)

latency_per_1k_output_tokens =
  1000 * total_latency_ms / max(effective_output_tokens, 1)

decode_latency_per_1k_output_tokens =
  1000 * decode_latency_ms / max(effective_output_tokens, 1)

decode_tokens_per_second =
  effective_output_tokens / max(decode_latency_ms / 1000, small_epsilon)

ms_per_output_token =
  decode_latency_ms / max(effective_output_tokens, 1)
```

Quality-per-token formula:

```text
quality_per_1k_tokens =
  1000 * quality_score / max(effective_total_tokens, 1)
```

For task-level aggregation, calculate `quality_per_1k_tokens` per request and also aggregate by:

* model_id
* backend
* provider
* router_name
* task_type
* prompt_family
* difficulty_bucket

Do not compare raw `quality_per_1k_tokens` across unrelated task types without grouping by task type.

For streaming APIs:

* record `ttft_ms` as the time from request start until the first generated token/chunk
* request usage metadata when supported
* if final provider usage is returned, use it and mark `usage_source = "provider_reported"`
* if final usage is missing due to stream interruption or provider limitation, estimate token usage locally and mark `usage_source = "estimated"`
* if neither provider nor estimated usage is available, mark `usage_source = "missing"`

For OpenAI-compatible router libraries:

* store provider-reported usage when available
* also store local estimated usage for debugging
* support cached input tokens and reasoning tokens when the provider exposes them
* never assume all providers use the same tokenization
* mark provider usage as the billing/source-of-truth value when present

## Artifacts

Add these fields to:

```text
artifacts/<run_id>/latency_results.jsonl
artifacts/<run_id>/quality_results.jsonl
artifacts/<run_id>/context_drift_results.jsonl
```

Update DB summaries to include:

* avg_ttft_per_1k_input_tokens
* avg_latency_per_1k_input_tokens
* avg_latency_per_1k_output_tokens
* avg_decode_latency_per_1k_output_tokens
* avg_quality_per_1k_tokens
* total_prompt_tokens
* total_completion_tokens
* total_tokens
* total_cached_prompt_tokens, if available
* total_reasoning_tokens, if available
* total_cost_estimate_usd
* successful_task_count
* failed_task_count

## CLI wiring

Wire `run-local` so after the Phase 4 smoke test it runs:

* latency benchmark
* quality benchmark
* context drift benchmark

Example:

```bash
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config
```

## Tests

Add or update tests for:

* latency metrics calculation
* JSON schema quality metric
* SQL syntax metric
* context drift scoring
* token usage effective-token fallback
* `usage_source = provider_reported`
* `usage_source = estimated`
* `usage_source = missing`
* `ttft_per_1k_input_tokens`
* `latency_per_1k_input_tokens`
* `latency_per_1k_output_tokens`
* `decode_latency_per_1k_output_tokens`
* `quality_per_1k_tokens`
* graceful failure category if generation fails

## Success criteria

Run:

```bash
uv run ruff check .
uv run pytest
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config
uv run hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
```

Must create:

```text
artifacts/<run_id>/latency_results.jsonl
artifacts/<run_id>/quality_results.jsonl
artifacts/<run_id>/context_drift_results.jsonl
```

The JSONL records must include token-normalized metrics where applicable:

```text
ttft_per_1k_input_tokens
latency_per_1k_input_tokens
latency_per_1k_output_tokens
decode_latency_per_1k_output_tokens
quality_per_1k_tokens
usage_source
```

Stop after Phase 5.

Report:

* files created/modified
* commands run
* test results
* generated artifacts
* any TODOs
* whether Phase 5 success criteria are met
