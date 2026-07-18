# hf-model-watch

`hf-model-watch` is a Python 3.11+ pipeline for discovering, filtering, testing,
adapting, comparing, and reporting on Hugging Face text-generation/chat causal
language models. It is built for local development first, then Nebius serverless GPU
worker execution with S3-compatible artifact storage.

The project favors evidence over claims: reports and comparisons use metrics stored in
the local results database and artifacts, not model-card promises.

## Architecture

The pipeline is split into small stages:

1. Hugging Face scan writes candidate metadata to JSONL.
2. Candidate filters block unsupported, gated, unsafe, oversized, or already-tested models.
3. Scoring ranks eligible candidates.
4. Local or worker runs resolve runtime backend, verify metadata claims, generate outputs,
   run latency/quality/context checks, and optionally run PEFT adaptation smoke tests.
5. SQLite stores runs, failures, artifacts, metrics, adaptation results, and comparisons.
6. Reporting creates Markdown/HTML leaderboards, per-run reports, and comparison reports.
7. Worker jobs run the same pipeline in `/workspace/output/<run_id>/` and upload artifacts
   to S3-compatible storage when configured.
8. The controller flow selects candidates, submits Nebius jobs, polls them, ingests manifest
   references, and regenerates reports.

Every CLI/job stage emits `TRACE_EVENT` JSON lines. Model-level failures are recorded with a
typed category; infrastructure failures may exit nonzero.

## Local Setup

```bash
uv sync
uv run ruff check .
uv run pytest
uv run hf-watch --help
```

On Windows, set a project-local uv cache to avoid user-profile cache issues:

```powershell
$env:UV_CACHE_DIR=(Resolve-Path .).Path + '\.uv-cache'
```

## Scanner And Scoring

```bash
uv run hf-watch scan --config config/scan.yaml
uv run hf-watch score-candidates --input artifacts/scan/latest_models.jsonl
```

Scanner output:

```text
artifacts/scan/latest_models.jsonl
artifacts/scan/scored_candidates.jsonl
```

## Local Tiny Model Run

```bash
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config --skip-finetune
```

Expected artifacts:

```text
artifacts/<run_id>/run_manifest.json
artifacts/<run_id>/metadata/claims_vs_observed.json
artifacts/<run_id>/runtime_result.json
artifacts/<run_id>/raw_generations.jsonl
artifacts/<run_id>/latency_results.jsonl
artifacts/<run_id>/quality_results.jsonl
artifacts/<run_id>/context_drift_results.jsonl
artifacts/<run_id>/trace_events.jsonl
```

Each run manifest includes model id, revision, config snapshot, package versions, CUDA
availability, GPU name when available, selected backend/dtype/quantization, random seed,
and artifact paths.

## PEFT Adaptation

Default fine-tune config lives in `config/finetune.yaml`. Fixture datasets are enabled by
default. Real datasets can be configured per task with:

```yaml
datasets:
  json_extraction:
    source: huggingface
    hf_dataset_name: org/dataset
    train_split: train
    eval_split: validation
    column_mapping:
      prompt_column: prompt
      completion_column: completion
```

Supported dataset sources are `fixture`, `huggingface`, `local_jsonl`, and `disabled`.
`financial_analyst_style` is the canonical finance task name; `finance_analysis` remains
accepted as a backward-compatible alias.

`peft` is optional. Without it, smoke-only runs record a typed unavailable result instead of
claiming adaptation happened.

## Worker Dry Run

Local dry-run does not require AWS or Nebius credentials:

```bash
DRY_RUN=1 bash scripts/run_model_worker_job.sh
```

On this Windows machine, plain `bash` may resolve to WSL. If WSL has no distro installed,
use Git Bash:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' -lc "DRY_RUN=1 bash scripts/run_model_worker_job.sh"
```

Dry-run artifacts are written under `.worker_workspace/output/<run_id>/` unless overridden.

## Prefect Dry Run

```bash
uv run hf-watch prefect-flow --config-dir config --dry-run
DRY_RUN=1 bash scripts/run_prefect_controller_job.sh
```

The dry-run flow selects a mock `sshleifer/tiny-gpt2` candidate, constructs a worker payload,
submits a dry-run Nebius job object, polls it as completed, records manifest references, and
regenerates the leaderboard.

## Nebius Service Account Setup

`scripts/run_prefect_controller_job.sh` supports these auth modes for non-dry-run execution:

```text
NEBIUS_CONFIG_YAML_B64
NEBIUS_SERVICE_ACCOUNT_KEY_B64
SA_ID + PUBLIC_KEY_ID + NEBIUS_PRIVATE_KEY_B64
SA_ID + PUBLIC_KEY_ID + NEBIUS_PRIVATE_KEY_PATH
```

Required non-dry-run environment:

```bash
NB_PROJECT_ID=<nebius-project-id>
NB_PROFILE_NAME=hf-model-watch-controller
```

The script installs/checks the Nebius CLI, configures the selected profile, then runs:

```bash
uv run hf-watch prefect-flow --config-dir config
```

## S3/Object Storage

Configure S3-compatible storage in `config/storage.yaml` or the worker payload:

```yaml
s3:
  enabled: true
  bucket: my-bucket
  prefix: hf-model-watch
  endpoint_url: https://s3.example.com
  dry_run: false
```

Worker artifacts upload to:

```text
s3://<bucket>/<prefix>/<run_id>/
```

Dry-run mode records the URI and emits upload trace events without contacting object storage.

## Result Storage

The implemented default store is SQLite:

```text
results/hf_model_watch.sqlite
```

Stored records include scans, candidate scores, runs, failures, artifacts, runtime results,
latency, quality, context drift, fine-tune runs, adaptation results, and comparisons.

## Reports

```bash
uv run hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
uv run hf-watch compare --model-id sshleifer/tiny-gpt2 --db results/hf_model_watch.sqlite
```

Generated reports:

```text
reports/leaderboard.md
reports/leaderboard.html
reports/runs/<run_id>.md
reports/runs/<run_id>.html
reports/comparisons/<model_slug>.md
```

Comparisons only use previously tested models from the result database.

## Full Local Dry Run

```bash
uv sync
uv run ruff check .
uv run pytest
uv run hf-watch scan --config config/scan.yaml
uv run hf-watch score-candidates --input artifacts/scan/latest_models.jsonl
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config --skip-finetune
uv run hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
uv run hf-watch compare --model-id sshleifer/tiny-gpt2 --db results/hf_model_watch.sqlite
uv run hf-watch prefect-flow --config-dir config --dry-run
DRY_RUN=1 bash scripts/run_model_worker_job.sh
DRY_RUN=1 bash scripts/run_prefect_controller_job.sh
```

## Full Prefect Run:

### Terminal 1:
```bash
export PREFECT_HOME=/home/maorb/.prefect-hf-model-watch
uv run --no-sync prefect server start
```

### Terminal 2:
```bash
set -a
source .env
set +a
uv run --no-sync python scripts/serve_prefect_controller.py --real-run --cron "0 7,19 * * *"
```

### Terminal 3:
```bash
uv run --no-sync prefect deployment run 'hf-model-watch-controller/hf-model-watch-controller-local'
```
## Nebius Commands:
access-permit create requires a group ID as --parent-id, not a user account ID
List groups with nebius iam group list --parent-id <tenant_id>
List registries with nebius registry list --parent-id <project_id>
List images/tags with nebius registry image list --parent-id <registry_id> (not repository list — that flag doesn't exist)
Docker pull from cr.eu-north1.nebius.cloud/<registry_id>/<image>:<tag> after running nebius registry configure-helper for auth
## Known Limitations

- Postgres storage is a documented optional integration point; SQLite is the implemented store.
- vLLM is optional and only selected when compatibility checks pass; Transformers is mandatory.
- Rule-based quality/context evaluations are smoke tests, not human-grade benchmark claims.
- PEFT training is a smoke path unless optional fine-tune dependencies and real datasets are installed.
- Hugging Face scanning depends on network access and current Hub API availability.
- On Windows, use Git Bash when `bash` resolves to WSL without an installed distro.
