# Codex Phase 4 — runtime resolver, metadata probe, tiny local model smoke test

Implement only Phase 4.

Assume Phases 1–3 are complete and tests pass.

Goal: implement model metadata verification, runtime resolution, memory estimation, Transformers backend loading, and a tiny local generation smoke test.

Do not implement full latency benchmark, quality eval, context drift, fine-tuning, Nebius, or Prefect yet.

Create/modify:

```text
src/hf_model_watch/runtime/runtime_resolver.py
src/hf_model_watch/runtime/memory_estimator.py
src/hf_model_watch/runtime/backend_transformers.py
src/hf_model_watch/runtime/backend_vllm.py
src/hf_model_watch/runtime/smoke_tests.py
src/hf_model_watch/hub/metadata.py

tests/test_memory_estimator.py
tests/test_runtime_resolver.py
```

Implement metadata claim verification:

Collect and compare:

* architecture
* parameter count estimate
* tokenizer load success
* chat template presence
* max context from config
* license
* gated status
* safetensors availability
* shard count and file size
* Transformers load support
* vLLM support attempt/result
* whether remote code was required
* whether BF16 load worked
* whether 4-bit load worked, if attempted

Output:

```text
artifacts/<run_id>/metadata/claims_vs_observed.json
```

Implement `memory_estimator.py`:

Estimate memory for:

* BF16 inference
* KV cache for configured context and batch
* QLoRA 4-bit training
* LoRA adapter size
* safety margin

Config:

```yaml
gpu_type: H100
gpu_memory_gb: 80
memory_safety_margin: 0.85
max_params_billion: 30
max_download_gb: 150
```

Implement `runtime_resolver.py`.

Given model metadata and config, decide:

* backend: `vllm` or `transformers`
* dtype: BF16/FP16/FP32
* attention implementation: flash/sdpa/eager when applicable
* quantization: none / 4bit
* max context for tests
* whether model fits H100
* whether to fail fast
* whether `trust_remote_code` would be required and whether allowed
* selected load arguments
* fallback ladder

Default fallback ladder:

```text
Attempt 1: vLLM inference, only if architecture appears supported and vLLM is installed
Attempt 2: Transformers BF16 with sdpa/flash attention
Attempt 3: Transformers BF16 eager attention
Attempt 4: 4-bit quantized load for metadata/QLoRA path only
Attempt 5: fail fast with typed failure category
```

For this phase, vLLM can be a safe optional stub that reports unavailable if not installed.

Implement `backend_transformers.py`:

* load tokenizer
* load model
* default `trust_remote_code=false`
* safe dtype selection
* run deterministic generation
* support tiny local smoke model

Implement `smoke_tests.py`:

Smoke test should:

1. Load tokenizer.
2. Load model.
3. Generate one short answer.
4. Record prompt tokens, generated tokens, elapsed time.
5. Save raw generation artifact.
6. Update manifest and DB.

Wire `run-local` so this command actually loads and tests tiny model:

```bash
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config
```

Artifacts:

```text
artifacts/<run_id>/metadata/claims_vs_observed.json
artifacts/<run_id>/raw_generations.jsonl
artifacts/<run_id>/runtime_result.json
```

Tests:

* memory estimator rejects obviously too-large model
* runtime resolver chooses Transformers when vLLM unavailable
* runtime resolver blocks trust_remote_code when disallowed
* tiny model local smoke path succeeds, unless environment lacks internet; in that case mock the backend in tests

Success criteria:

```bash
uv run ruff check .
uv run pytest
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config
uv run hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
```

Must create:

```text
artifacts/<run_id>/metadata/claims_vs_observed.json
artifacts/<run_id>/raw_generations.jsonl
artifacts/<run_id>/runtime_result.json
```

Stop after Phase 4. Report files created/modified, commands run, test results, and any TODOs.
