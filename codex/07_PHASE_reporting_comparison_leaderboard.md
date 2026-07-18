# Codex Phase 7 — reporting, comparison, leaderboard

Implement only Phase 7.

Assume Phases 1–6 are complete and tests pass.

Goal: implement robust Markdown/HTML reports, model comparison, token-efficiency reporting, and leaderboard logic.

Create/modify:

```text
src/hf_model_watch/reporting/leaderboard.py
src/hf_model_watch/reporting/compare.py
src/hf_model_watch/reporting/markdown_report.py
src/hf_model_watch/reporting/html_report.py

tests/test_leaderboard.py
tests/test_compare.py
tests/test_token_efficiency_reporting.py
```

## Leaderboard

Leaderboard columns:

* rank
* model_id
* revision short SHA
* family
* size band
* scan score
* eligibility status
* selected backend
* base quality score
* adapted quality score
* adaptation lift
* normalized adaptation lift
* latency tokens/sec
* first-token latency / TTFT
* context drift score
* peak VRAM
* cost estimate
* failure category
* artifact URI

## Token-efficiency leaderboard columns

Add these columns:

* total_prompt_tokens
* total_completion_tokens
* total_tokens
* total_cached_prompt_tokens, if available
* total_reasoning_tokens, if available
* total_cost_estimate_usd
* quality_per_1k_tokens
* cost_per_successful_task
* tokens_per_successful_task
* ttft_per_1k_input_tokens
* latency_per_1k_input_tokens
* latency_per_1k_output_tokens
* decode_latency_per_1k_output_tokens
* adaptation_lift_per_1M_training_tokens
* normalized_lift_per_1M_training_tokens
* usage_source_summary

Definitions:

```text
cost_per_successful_task =
  total_cost_estimate_usd / max(successful_task_count, 1)
```

```text
tokens_per_successful_task =
  total_tokens / max(successful_task_count, 1)
```

```text
quality_per_1k_tokens =
  1000 * quality_score / max(total_tokens, 1)
```

If `quality_per_1k_tokens` already exists from Phase 5, use the stored value. If only raw `quality_score` and token totals exist, compute it in reporting.

If token usage is missing, show `null` or `N/A`; do not invent values.

## Comparison report

Comparison report rules:

* compare current model to previously tested models only
* prefer same family comparison when possible
* prefer same size band comparison when possible
* mark as inconclusive when insufficient prior data exists
* answer: `newer_better`, `similar`, `mixed`, `newer_worse`, or `inconclusive`
* show evidence from stored metrics
* do not hallucinate predecessor results
* do not compare against models that were scanned but never tested

Comparison logic:

```text
If no prior tested models in same family/size band:
  comparison_status = inconclusive

If current model improves quality, latency, and context drift beyond configured thresholds:
  comparison_status = newer_better

If differences are small and within configured thresholds:
  comparison_status = similar

If quality improves but latency/cost/token efficiency worsens:
  comparison_status = mixed

If quality or context drift is meaningfully worse:
  comparison_status = newer_worse
```

Allow thresholds and weights in config.

Suggested comparison priorities:

1. quality_score
2. context_drift_score
3. latency metrics
4. token efficiency
5. cost efficiency
6. adaptation lift

A model should not be marked `newer_better` only because it uses fewer tokens. It should be marked better only if quality is comparable or better.

Suggested default thresholds:

```yaml
comparison:
  min_quality_delta: 0.02
  min_context_drift_delta: 0.02
  min_latency_delta_pct: 0.05
  min_token_efficiency_delta_pct: 0.05
  quality_comparable_margin: 0.01
```

## Reports to create

Create:

```text
reports/leaderboard.md
reports/leaderboard.html
reports/runs/<run_id>.md
reports/runs/<run_id>.html
reports/comparisons/<model_slug>.md
```

Each run report should include:

* summary
* model metadata
* runtime resolution
* claim verification
* latency results
* quality results
* context drift
* token usage
* token-normalized latency
* token efficiency
* PEFT config
* adaptation lift
* adaptation efficiency
* failures
* artifacts

Add these report sections:

```text
## Token Usage

- Input tokens
- Output tokens
- Total tokens
- Cached input tokens, if available
- Reasoning tokens, if available
- Usage source: provider_reported / estimated / missing

## Token-Normalized Latency

- TTFT per 1k input tokens
- Total latency per 1k input tokens
- Total latency per 1k output tokens
- Decode latency per 1k output tokens
- Decode tokens/sec

## Token Efficiency

- Quality per 1k tokens
- Tokens per successful task
- Cost per successful task

## Adaptation Efficiency

- Base quality score
- Adapted quality score
- Raw adaptation lift
- Normalized adaptation lift
- Training tokens
- Adaptation lift per 1M training tokens
- Normalized lift per 1M training tokens
```

For comparisons, show a compact table:

```text
Metric | Current model | Best prior same-family model | Winner
```

Important:

* Compare token-efficiency metrics only within the same task type or same benchmark suite.
* Mark comparison as `inconclusive` when token usage is estimated for one model but provider-reported for another and the difference is small.
* Store whether usage was provider-reported, estimated, missing, or mixed.
* If Phase 6 adaptation metrics are unavailable for a run, show `N/A` instead of failing.

## CLI

Commands:

```bash
hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
hf-watch compare --model-id org/model --db results/hf_model_watch.sqlite
```

The `compare` command should write:

```text
reports/comparisons/<model_slug>.md
```

and also print a concise summary to stdout.

## Tests

Add or update tests:

* leaderboard sorts correctly
* failed runs appear with failure category
* comparison does not compare against untested models
* comparison returns `inconclusive` when no prior tested model exists
* comparison returns `similar` when differences are below threshold
* token-efficiency metrics are calculated correctly
* missing token metrics render as `N/A`
* estimated vs provider-reported usage is surfaced
* HTML and Markdown reports are generated
* comparison report file is generated

## Success criteria

Run:

```bash
uv run ruff check .
uv run pytest
uv run hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
uv run hf-watch compare --model-id sshleifer/tiny-gpt2 --db results/hf_model_watch.sqlite
```

Must create:

```text
reports/leaderboard.md
reports/leaderboard.html
reports/comparisons/sshleifer-tiny-gpt2.md
```

Stop after Phase 7.

Report:

* files created/modified
* commands run
* test results
* generated reports
* any TODOs
* whether Phase 7 success criteria are met
