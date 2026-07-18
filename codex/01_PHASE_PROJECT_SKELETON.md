# Codex Phase 1 — project skeleton, configs, tracing, CLI

Implement only Phase 1.

Goal: create the project skeleton, config system, tracing system, CLI shell, basic error categories, and tests. Do not implement actual Hugging Face scanning, model loading, fine-tuning, Nebius, or Prefect yet.

Create this structure:

```text
hf-model-watch/
  pyproject.toml
  README.md
  .gitignore
  .env.example

  config/
    scan.yaml
    eval_tasks.yaml
    finetune.yaml
    serverless.yaml
    storage.yaml
    scoring.yaml

  scripts/
    run_model_worker_job.sh
    run_prefect_controller_job.sh
    examples/
      README.md

  src/hf_model_watch/
    __init__.py
    cli.py
    tracing.py
    config.py
    utils.py
    errors.py

    hub/
      __init__.py

    runtime/
      __init__.py

    evals/
      __init__.py

    finetune/
      __init__.py

    storage/
      __init__.py

    orchestration/
      __init__.py

    reporting/
      __init__.py

  tests/
    test_config.py
    test_tracing.py
    test_errors.py
```

Implement:

1. `pyproject.toml`

   * Python 3.11+
   * package name `hf-model-watch`
   * CLI entrypoint `hf-watch`
   * pytest and ruff config

2. `tracing.py`

   * function `emit_trace_event(event_type, stage, status, message, metadata=None, metrics=None, artifacts=None)`
   * prints exactly one line starting with `TRACE_EVENT `
   * JSON must be valid
   * include timestamp in ISO format

3. `errors.py`

   * define typed failure categories as constants or enum:

```text
not_causal_lm
too_large_for_h100
missing_tokenizer
missing_config
missing_weights
missing_safetensors
gated_no_access
license_blocked
already_tested_revision
requires_trust_remote_code
unsupported_transformers_architecture
unsupported_vllm_architecture
download_failed
download_too_large
load_oom
load_failed
generation_failed
latency_benchmark_failed
quality_eval_failed
context_test_failed
finetune_smoke_failed
finetune_oom
finetune_failed
adapter_save_reload_failed
artifact_upload_failed
controller_auth_failed
nebius_job_failed
unknown_failure
```

4. `config.py`

   * load YAML config files from `config/`
   * merge into one config object or dict
   * validate required sections
   * support env override for common values where simple

5. `cli.py`

   * Typer CLI with placeholder commands that emit trace events and do not crash:

```bash
hf-watch scan --config config/scan.yaml
hf-watch score-candidates --input artifacts/scan/latest_models.jsonl
hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config
hf-watch worker --payload-json payload.json
hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
hf-watch compare --model-id org/model --db results/hf_model_watch.sqlite
hf-watch prefect-flow --config-dir config
```

For Phase 1, these commands may return “not implemented yet” trace events, but the CLI must exist and run.

6. Config files

   * add sane default YAML files.
   * `scan.yaml` should include:

```yaml
lookback_days: 2
max_models_to_scan: 200
max_candidates_to_run: 5
max_params_billion: 30
allow_gated: false
allow_trust_remote_code: false
require_safetensors: true
allowed_licenses:
  - apache-2.0
  - mit
  - llama
  - other
blocked_licenses:
  - unknown-restrictive
max_download_gb: 150
```

* `serverless.yaml` should include:

```yaml
gpu_type: H100
gpu_count: 1
timeout_minutes: 180
max_parallel_jobs: 2
```

7. Shell scripts

   * create minimal placeholder scripts with `set -euo pipefail`
   * emit a basic `TRACE_EVENT`
   * do not implement full Nebius yet

8. README

   * describe the project and phase status
   * include local setup commands

Tests:

* `test_tracing.py`: confirms `TRACE_EVENT` line is valid JSON.
* `test_errors.py`: confirms required failure categories exist.
* `test_config.py`: confirms configs load.

Success criteria:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run hf-watch --help
uv run hf-watch scan --config config/scan.yaml
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config
```

Stop after Phase 1. Report files created/modified, commands run, test results, and any TODOs.
