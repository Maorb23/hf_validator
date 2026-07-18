# Codex Phase 6 — PEFT config resolver and adaptation suite

Implement only Phase 6.

Assume Phases 1–5 are complete and tests pass.

Goal: implement PEFT config resolution, LoRA/QLoRA smoke tests, small adaptation tasks, and base-vs-adapted scoring.

Do not implement Nebius or Prefect yet.

Create/modify:

```text
src/hf_model_watch/finetune/datasets.py
src/hf_model_watch/finetune/peft_config_resolver.py
src/hf_model_watch/finetune/train_peft.py
src/hf_model_watch/finetune/eval_adaptation.py

tests/test_peft_config_resolver.py
tests/test_adaptation_metrics.py
```

Adaptation principle:

Measure improvement, not only final score.

For each task:

```text
base_score = model before adaptation
adapted_score = model after adaptation
adaptation_lift = adapted_score - base_score
```

Implement three adaptation tasks:

## Task A: structured JSON extraction

Use a small deterministic local synthetic dataset.

Goal:

* extract structured fields
* obey schema
* avoid extra text

Metrics:

* JSON parse rate
* schema pass rate
* field-level F1

## Task B: Text-to-SQL mini adaptation

Use a small local fixture dataset.

Goal:

* adapt to SQL output format
* measure syntax and simple correctness

Metrics:

* SQL syntax pass
* exact query match where applicable
* execution accuracy when SQLite fixture exists

Important: do not skip adaptation if baseline SQL is weak. The lift is informative.

## Task C: financial analyst-style response adaptation

Primary intended dataset: FinTextQA or configurable finance QA dataset.

For local/offline tests, include a tiny synthetic finance-style fixture.

Goal:

* clear answer
* concise reasoning
* caveats/assumptions
* no hallucinated certainty
* evidence-based wording when context is supplied

Metrics:

* format compliance
* caveat presence/quality
* factual grounding against supplied context
* concise reasoning score
* optional local judge hook, disabled by default

Do not produce investment recommendations. The style is educational/analytical, not financial advice.

Add configurable support for harder finance holdout evals later, such as FinQA, TAT-QA, or ConvFinQA, but do not train on holdout data by default.

PEFT config resolver:

Implement `peft_config_resolver.py`.

Inputs:

* model size
* architecture
* available VRAM
* task
* desired budget
* context length
* quantization mode

Outputs:

* method
* optimizer
* scheduler
* rank
* alpha
* dropout
* target modules
* batch size
* gradient accumulation
* max sequence length
* learning rate
* max steps
* reason

Default policy:

```yaml
default_method: qlora
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
lora_dropout: 0.05
target_modules: all-linear
max_grad_norm: 1.0
weight_decay: 0.0
scheduler: cosine
warmup_ratio: 0.03
```

Model size policy:

```text
<= 7B:
  method: LoRA or QLoRA
  r: 16
  alpha: 32
  learning_rate: 2e-4
  effective_batch_size: 32

7B-15B:
  method: QLoRA
  r: 16
  alpha: 32
  learning_rate: 2e-4
  effective_batch_size: 16

15B-30B:
  method: QLoRA
  r: 8 or 16
  alpha: 16 or 32
  learning_rate: 1e-4 or 2e-4
  effective_batch_size: 8-16
```

Optimizer policy:

```text
For QLoRA:
  optimizer: paged_adamw_8bit
  fallback: paged_adamw_32bit

For BF16 LoRA:
  optimizer: adamw_torch_fused
  fallback: adamw_torch
```

Scheduler policy:

```text
scheduler: cosine
warmup_steps: max(5, int(warmup_ratio * max_steps))
```

Training defaults:

```yaml
bf16: true
gradient_checkpointing: true
max_steps: 100
eval_steps: 25
save_strategy: final_only
seed: 42
```

Do not hardcode only `q_proj` and `v_proj`.

Use:

```text
Primary target_modules: all-linear
Fallback: discover linear module names automatically
Fallback: architecture-specific known map
Fail: no trainable adapter modules found
```

Mandatory PEFT smoke test before real training:

1. Load model.
2. Attach adapter.
3. Verify trainable parameter count > 0.
4. Run one forward pass.
5. Run one backward pass.
6. Run one optimizer step.
7. Save adapter.
8. Reload adapter.
9. Run one generation.

Failure category:

```text
finetune_smoke_failed
```
## Adaptation lift per training token

In `train_peft.py`, `eval_adaptation.py`, and `metrics.py`, record training-token efficiency for every PEFT adaptation run.

For each adaptation task, store:

* model_id
* run_id
* task_type
* peft_method
* base_quality_score
* adapted_quality_score
* adaptation_lift
* normalized_adaptation_lift
* training_examples
* training_prompt_tokens
* training_completion_tokens
* training_total_tokens
* train_steps
* effective_batch_size
* max_seq_length
* optimizer
* scheduler
* learning_rate
* lora_r
* lora_alpha
* trainable_params
* total_params
* trainable_param_ratio
* adaptation_lift_per_1k_training_tokens
* adaptation_lift_per_1M_training_tokens
* normalized_lift_per_1M_training_tokens

Definitions:

```text
adaptation_lift =
  adapted_quality_score - base_quality_score
```

```text
normalized_adaptation_lift =
  adaptation_lift / max(1 - base_quality_score, epsilon)
```

Use normalized lift because a model with high baseline quality has less headroom to improve.

Training-token formulas:

```text
training_total_tokens =
  training_prompt_tokens + training_completion_tokens
```

```text
adaptation_lift_per_1k_training_tokens =
  1000 * adaptation_lift / max(training_total_tokens, 1)
```

```text
adaptation_lift_per_1M_training_tokens =
  1_000_000 * adaptation_lift / max(training_total_tokens, 1)
```

```text
normalized_lift_per_1M_training_tokens =
  1_000_000 * normalized_adaptation_lift / max(training_total_tokens, 1)
```

Also compute task-specific versions:

* json_adaptation_lift_per_1M_training_tokens
* sql_adaptation_lift_per_1M_training_tokens
* finance_adaptation_lift_per_1M_training_tokens

Important:

* Store raw lift and normalized lift.
* Store per-task lift and overall weighted lift.
* Do not treat low lift as failure when the base model is already strong.
* Mark saturated baselines clearly when `base_quality_score >= 0.90`.
* Use fixed train-token accounting so models are comparable.
* If token counting fails, record `training_total_tokens = null` and do not compute lift-per-token metrics.

Add these fields to:

```text
artifacts/<run_id>/finetune_config.json
artifacts/<run_id>/adaptation_results.json
artifacts/<run_id>/adapter_eval.json
```

Update DB summaries to include:

* base_quality_score
* adapted_quality_score
* adaptation_lift
* normalized_adaptation_lift
* training_total_tokens
* adaptation_lift_per_1M_training_tokens
* normalized_lift_per_1M_training_tokens


Record:

```json
{
  "peft_method": "qlora",
  "optimizer": "paged_adamw_8bit",
  "scheduler": "cosine",
  "r": 16,
  "alpha": 32,
  "dropout": 0.05,
  "target_modules": "all-linear",
  "trainable_params": 123,
  "total_params": 123456,
  "trainable_param_ratio": 0.001,
  "learning_rate": 0.0002,
  "max_steps": 100,
  "effective_batch_size": 16,
  "reason": "model estimated >10B params; using QLoRA to fit H100 and keep comparisons fair"
}
```

Wire `run-local`:

* default local tiny model should not necessarily run full PEFT if impossible
* support `--skip-finetune`
* support `--finetune-smoke-only`
* when PEFT cannot run locally, record typed skip/failure cleanly
* GPU path should be realistic

Artifacts:

```text
artifacts/<run_id>/finetune_config.json
artifacts/<run_id>/adaptation_results.json
artifacts/<run_id>/adapter_eval.json
```

Tests:

* PEFT resolver chooses sane config for 3B, 8B, 20B mock models
* target module fallback works on mock modules
* adaptation lift calculation works
* local synthetic datasets load
* fine-tune unavailable path records typed result, not crash

Success criteria:

```bash
uv run ruff check .
uv run pytest
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config --finetune-smoke-only
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config --skip-finetune
uv run hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
```

Stop after Phase 6. Report files created/modified, commands run, test results, and any TODOs.
