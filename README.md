# hf-model-watch

`hf-model-watch` is a Python pipeline for discovering, filtering, evaluating, adapting, and
comparing Hugging Face causal language models. It supports lightweight local validation and a
remote execution path built around Nebius GPU jobs, Prefect orchestration, SQLite, and
S3-compatible artifact storage.

The project favors observed evidence over model-card claims. Runtime metadata, generated
outputs, latency, quality checks, context behavior, PEFT results, failures, and reproducibility
details are stored as artifacts and database records that can be inspected after each run.

> **Project status:** experimental. The included evaluations and fine-tuning defaults are
> designed for model intake and smoke testing, not for safety certification or production
> benchmark claims. Review model licenses, dataset licenses, generated outputs, and cloud costs
> before running untrusted or large models.

## What it does

- Discovers recently updated Hugging Face text-generation models.
- Blocks gated, unsupported, oversized, unsafe, or policy-incompatible candidates.
- Scores and ranks eligible candidates using configurable heuristics.
- Resolves a Transformers or optional vLLM runtime from observed capabilities.
- Measures latency, basic task quality, and context behavior.
- Runs independent LoRA or QLoRA adaptations for JSON extraction, text-to-SQL, and financial
  analyst-style tasks.
- Records typed failures instead of silently dropping unsuccessful models.
- Produces Markdown and HTML leaderboards, run reports, and model comparisons.
- Runs locally or through Prefect-managed Nebius jobs with optional S3-compatible storage.

## Requirements

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.
- Network access for Hugging Face scanning, model downloads, and configured remote services.
- Enough disk and memory for the selected model.
- Linux and an NVIDIA CUDA GPU for the checked-in QLoRA/automatic fine-tuning path.

A CPU is sufficient for the tiny-model smoke run shown below. Real fine-tuning requires the
optional fine-tuning dependencies and compatible GPU drivers. Sentence-completion semantic
scoring requires the `quality` extra.

## Quick start

From a local checkout:

```bash
uv sync
cp .env.example .env
uv run hf-watch --help
```

To run the custom sentence-completion benchmark, install its embedding dependency:

```bash
uv sync --extra quality
```

Run the test suite and linter:

```bash
uv run pytest
uv run ruff check .
```

Run a small local model without fine-tuning:

```bash
uv run hf-watch run-local \
  --model-id sshleifer/tiny-gpt2 \
  --config-dir config \
  --skip-finetune
```

This downloads a Hugging Face model and writes a new run under `artifacts/`. It does not submit
a Nebius job or upload artifacts.

## Typical workflow

### 1. Discover candidates

```bash
uv run hf-watch scan --config config/scan.yaml
uv run hf-watch score-candidates --input artifacts/scan/latest_models.jsonl
```

Outputs:

```text
artifacts/scan/latest_models.jsonl
artifacts/scan/scored_candidates.jsonl
```

The scanner is controlled by `config/scan.yaml`. Review its license allowlist, model-size cap,
gated-model policy, `trust_remote_code` policy, and download-size cap before scanning.

### 2. Validate one model locally

```bash
uv run hf-watch run-local \
  --model-id <hugging-face-model-id> \
  --config-dir config \
  --skip-finetune
```

Remove `--skip-finetune` only after installing the fine-tuning dependencies and reviewing the
fine-tuning configuration. Use `--finetune-smoke-only` to perform a single-step PEFT capability
check instead of the configured training run.

### 3. Generate reports

```bash
uv run hf-watch leaderboard \
  --db results/hf_model_watch.sqlite \
  --out reports/leaderboard.md

uv run hf-watch compare \
  --model-id <hugging-face-model-id> \
  --db results/hf_model_watch.sqlite
```

Comparisons use previously recorded runs in the SQLite database; they do not rerun a model.

## Command reference

| Command | Purpose |
| --- | --- |
| `scan` | Discover Hugging Face candidates and write scan metadata. |
| `lookup` | Fetch exact model metadata or search a model family without downloading weights. |
| `score-candidates` | Score a candidate JSONL file and write ranked candidates. |
| `run-local` | Run metadata checks, inference evaluations, and optional PEFT adaptation. |
| `worker` | Execute a model job from a JSON payload and optionally upload its artifacts. |
| `leaderboard` | Build a Markdown leaderboard from the SQLite database. |
| `compare` | Generate a report comparing recorded runs for one model. |
| `ingest-results` | Import downloaded run directories into SQLite and rebuild reports. |
| `sync-nebius-results` | Download reporting artifacts from configured S3 storage and import them. |
| `prefect-flow` | Select candidates and orchestrate Nebius worker jobs. |
| `prefect-sync-results` | Run result synchronization through the Prefect flow. |

Use `uv run hf-watch <command> --help` for command-specific options.

Examples: `hf-watch lookup --model-id Qwen/Qwen3-4B`, `hf-watch lookup --family qwen3`,
or `hf-watch lookup --query qween3 --fuzzy`. Lookup records include the exact model ID,
revision, and `lookup_match` evidence; use the returned exact ID for later validation.

## Configuration

The CLI loads YAML files from `config/`:

| File | Controls |
| --- | --- |
| `config/scan.yaml` | Discovery limits, licenses, gating, size limits, and remote-code policy. |
| `config/scoring.yaml` | Candidate ranking weights and scoring behavior. |
| `config/eval_tasks.yaml` | Latency, quality, and context evaluation prompts. |
| `config/finetune.yaml` | Datasets, LoRA/QLoRA policy, batching, evaluation, and W&B settings. |
| `config/serverless.yaml` | Nebius image, GPU platform, job limits, and concurrency. |
| `config/storage.yaml` | Local artifacts, SQLite, reports, and S3-compatible storage. |

Copy `.env.example` to `.env` for environment-based overrides and credentials. Do not commit
tokens, cloud keys, private keys, base64-encoded service-account material, or populated payloads.

Common environment variables include:

```text
HF_TOKEN
HF_HOME
HF_MODEL_WATCH_ALLOW_GATED
HF_MODEL_WATCH_ALLOW_TRUST_REMOTE_CODE
HF_MODEL_WATCH_MAX_MODELS_TO_SCAN
HF_MODEL_WATCH_MAX_CANDIDATES_TO_RUN
HF_MODEL_WATCH_MAX_PARAMS_BILLION
HF_MODEL_WATCH_STORAGE_URI
WANDB_API_KEY
WANDB_PROJECT
```

`trust_remote_code` and gated-model access are disabled by default. Enable either only after
reviewing the model repository and its license.

### Custom quality benchmark

The checked-in quality tasks are disabled until you replace the placeholder JSONL rows:

- `data/quality/math.jsonl` contains math questions and canonical final answers.
- `data/quality/sentence_completion.jsonl` contains online-discourse queries and two or more
  acceptable completions.
- `data/quality/README.md` documents every field and the source, licensing, privacy, and
  decontamination checks to perform before enabling the tasks.

After filling the files, set `quality_benchmark.math.enabled` and
`quality_benchmark.sentence_completion.enabled` to `true` in `config/eval_tasks.yaml`.
Math reports exact/tolerance-aware accuracy plus mean, median, p90, and p95 answer latency.
Sentence completion reports the maximum cosine similarity against acceptable references using
`Qwen/Qwen3-Embedding-0.6B` by default. It also teacher-forces each reference through the
evaluated model and reports cross-entropy, perplexity, negative log-likelihood, and
tokenizer-normalized bits per UTF-8 byte; lower loss values are better.

The embedding model and revision are configurable. Pin `embedding_revision` for reproducible
published results. Keep benchmark rows out of fine-tuning data and retain source/license
metadata for externally collected discourse.

Latency output intentionally keeps raw total latency, decode latency, TTFT, and decode
tokens/second. It does not emit per-input-token or per-output-token latency ratios, because
those ratios can be misleading across prompts and generation lengths.

## Run artifacts

A normal local run writes files similar to:

```text
artifacts/<run_id>/
├── run_manifest.json
├── metadata/claims_vs_observed.json
├── runtime_result.json
├── raw_generations.jsonl
├── latency_results.jsonl
├── quality_results.jsonl
├── context_drift_results.jsonl
├── trace_events.jsonl
├── finetune_config.json
├── finetune_plan.json
├── model_profiles.json
├── adaptation_results.json
├── adapter_eval.json
└── finetune/<task>/
```

Fine-tuning files are still written when adaptation is skipped or unavailable so that the
recorded status remains explicit. Successful task runs also write adapter directories and
per-attempt logs/results.

The run manifest captures the model revision, configuration snapshot, package versions, CUDA
availability, observed GPU, backend, dtype, quantization, seed, and artifact references.

The default application database is:

```text
results/hf_model_watch.sqlite
```

Every CLI and job stage emits machine-readable lines prefixed with `TRACE_EVENT `. Model-level
failures are stored with typed failure categories; infrastructure failures may exit nonzero.

## PEFT adaptation

Install the optional dependencies before using LoRA or QLoRA:

```bash
uv sync --extra finetune
```

The default `method: auto` policy selects BF16 LoRA when its conservative memory estimate fits
the configured GPU budget and otherwise selects NF4 QLoRA. Dataset and model profiles influence
sequence length, microbatch size, gradient accumulation, rank, dropout, and learning rate.
Resolved values and their reasons are stored in `finetune_config.json`.

The checked-in configuration runs one independent adapter per task:

| Task | Default dataset | Train cap | Evaluation cap |
| --- | --- | ---: | ---: |
| JSON extraction | `paraloq/json_data_extraction` | 500 | 100 |
| Text-to-SQL | `b-mc2/sql-create-context` | 500 | 100 |
| Financial analyst style | `bevaya/FinQA` | 500 | 100 |

These are cost-conscious smoke defaults, not statistically justified training sizes. Dataset
sources may be changed to `fixture`, `huggingface`, `local_jsonl`, or `disabled`.

Example dataset configuration:

```yaml
datasets:
  json_extraction:
    source: huggingface
    hf_dataset_name: org/dataset
    train_split: train
    eval_split: validation
    max_train_examples: 500
    max_eval_examples: 100
    column_mapping:
      prompt_column: prompt
      completion_column: completion
      schema_column: schema
```

For local JSONL data, set `source: local_jsonl`, `local_jsonl_path`, and optionally
`local_eval_jsonl_path`. The financial task's legacy name, `finance_analysis`, remains accepted
as an alias for `financial_analyst_style`.

### JSON extraction evaluation

JSON evaluation budgets are based on tokenizer tokens rather than whitespace-delimited words.
The defaults reserve at least 32 new tokens, target twice the expected token length, and cap the
generation at 256 tokens. The scorer extracts the first complete JSON object or array and
compares the parsed value with the expected value, so key order does not affect exact-match
accuracy.

These settings are configurable in `config/finetune.yaml`:

```yaml
json_eval_generation_token_multiplier: 2.0
json_eval_min_new_tokens: 32
json_eval_max_new_tokens: 256
```

A completed optimizer run is not proof of useful adaptation. Inspect `base_scores`,
`adapted_scores`, losses, evaluation history, and attempt logs before accepting an adapter.

### Monitoring and retries

W&B monitoring is enabled by default. Set `wandb_enabled: false` when W&B is not configured or
an untracked run is required. Each task and fallback attempt gets a separate W&B run.

Training executes in a child process so a failed attempt releases its CUDA context. Automatic
LoRA may retry once as QLoRA; automatic batch sizing may retry with a smaller microbatch after
an OOM or memory-threshold failure. Explicit method and batch choices are not silently changed.

## Dry runs

Run the worker bootstrap without Nebius or S3 credentials:

```bash
DRY_RUN=1 bash scripts/run_model_worker_job.sh
```

Run the controller flow without submitting Nebius jobs:

```bash
uv run hf-watch prefect-flow --config-dir config --dry-run
DRY_RUN=1 bash scripts/run_prefect_controller_job.sh
```

Worker dry-run files are written below `.worker_workspace/output/` unless overridden.

On Windows, use Git Bash if `bash` resolves to WSL and no WSL distribution is installed:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' -lc "DRY_RUN=1 bash scripts/run_model_worker_job.sh"
```

## Nebius and Prefect

Before a real remote run:

1. Build and publish the included `Dockerfile` to a registry accessible by Nebius.
2. Replace the sample image and project values in `config/serverless.yaml` or environment
   variables.
3. Configure a Nebius service account with only the permissions required to submit and inspect
   jobs.
4. Configure S3-compatible artifact storage and test its credentials separately.
5. Run the controller dry-run before enabling real submissions.

The controller bootstrap supports these authentication inputs:

```text
NEBIUS_CONFIG_YAML_B64
NEBIUS_SERVICE_ACCOUNT_KEY_B64
SA_ID + PUBLIC_KEY_ID + NEBIUS_PRIVATE_KEY_B64
SA_ID + PUBLIC_KEY_ID + NEBIUS_PRIVATE_KEY_PATH
```

Required values for non-dry-run controller execution include `NB_PROJECT_ID` and a configured
profile name such as `NB_PROFILE_NAME=hf-model-watch-controller`.

Start a local Prefect server:

```bash
export PREFECT_HOME="$PWD/.prefect-hf-model-watch"
uv run --no-sync prefect server start
```

In another terminal, register the controller deployment:

```bash
set -a
source .env
set +a
uv run --no-sync python scripts/serve_prefect_controller.py \
  --real-run \
  --cron "0 7,19 * * *"
```

Trigger it manually:

```bash
uv run --no-sync prefect deployment run \
  'hf-model-watch-controller/hf-model-watch-controller-local'
```

The result-sync deployment can be served separately:

```bash
uv run --no-sync python scripts/serve_prefect_result_sync.py --cron "*/10 * * * *"
```

## S3-compatible artifact storage

Configure `config/storage.yaml` or provide an equivalent worker-payload override:

```yaml
s3:
  enabled: true
  bucket: my-bucket
  prefix: hf-model-watch
  endpoint_url: https://s3.example.com
  dry_run: false
```

Worker artifacts are uploaded to:

```text
s3://<bucket>/<prefix>/<run_id>/
```

Dry-run uploads record the destination URI and emit trace events without contacting object
storage.

Download reporting artifacts and import them into SQLite:

```bash
uv run --no-sync hf-watch sync-nebius-results \
  --config-dir config \
  --cache-dir outputs/synced-nebius
```

Alternatively, import an existing directory of downloaded runs:

```bash
uv run --no-sync hf-watch ingest-results \
  --input outputs \
  --db results/hf_model_watch.sqlite \
  --out reports/leaderboard.md
```

Both operations are idempotent unless `--force` is supplied. S3 synchronization downloads
reporting JSON files, not model or adapter weights.

## Architecture

```text
Hugging Face Hub
      │
      ▼
scan → policy filters → candidate scoring
      │
      ▼
local runner or Prefect/Nebius worker
      │
      ├── metadata verification
      ├── runtime resolution and inference
      ├── latency / custom quality / context evaluation
      └── optional per-task LoRA or QLoRA
      │
      ▼
artifacts + SQLite + optional S3
      │
      ▼
leaderboards, run reports, and comparisons
```

## Safety and operational guidance

- Keep `allow_trust_remote_code: false` unless model repository code has been reviewed.
- Treat model output, dataset content, and downloaded artifacts as untrusted data.
- Keep model-size, download-size, job-timeout, and concurrency limits appropriate for your
  hardware and cloud budget.
- Do not interpret the included rule-based quality checks as factuality, security, bias, or
  alignment certification.
- Do not expose the SQLite database, generated outputs, W&B runs, or artifact buckets if they
  contain sensitive prompts or model responses.
- Pin model revisions and container images for reproducible or regulated workflows.

## Known limitations

- SQLite is the implemented application store; the Postgres module is an integration point,
  not the default production path.
- Transformers is the required runtime. vLLM is optional and selected only when compatibility
  checks pass.
- The supplied quality data files are placeholders. Published benchmark claims require a
  sufficiently large, reviewed, licensed, versioned, and contamination-checked dataset.
- PEFT completion means the training process finished; adapter usefulness must be judged from
  task metrics and generated outputs.
- JSON scoring accepts the first complete object or array embedded in a response; it does not
  currently report strict JSON-only adherence as a separate metric.
- Generation-based PEFT evaluation processes examples sequentially and may add noticeable time
  before and after training, especially for long schemas or contexts.
- Hugging Face scanning depends on current Hub metadata and network availability.
- Real Nebius and object-storage execution requires external credentials, permissions, images,
  quotas, and billing configuration that this repository cannot validate locally.

## Development

Install all development and fine-tuning dependencies:

```bash
uv sync --extra finetune
```

Run validation:

```bash
uv run pytest
uv run ruff check .
```

When changing artifact schemas, CLI behavior, configuration fields, or report formats, update
the relevant tests and this README in the same change.
# Model recommendations

For the recommender notebook and interactive CLI, `requirements_cpu.txt` creates
a lightweight environment without Torch, CUDA, benchmark, or training packages:

```bash
uv venv ~/.venvs/nemotron_runner
source ~/.venvs/nemotron_runner/bin/activate
uv pip install -r requirements_cpu.txt
python -m ipykernel install --user --name nemotron-runner --display-name "Nemotron runner (WSL)"
```

Start Jupyter from the repository root and choose that kernel. Run the CLI from
the repository root with `PYTHONPATH=src python -m hf_model_watch`; no project install
or GPU packages are needed. This CPU-only environment is not suitable for local model
inference or benchmarks.

`hf-watch recommend --model-id <id>` uses a bounded Nemotron tool loop to collect pinned Hugging Face
metadata and model cards. Full cards, a JSON recommendation, and structured tool events
are saved under `artifacts/recommendations/`. Set `NEBIUS_API_KEY` and optionally
`TAVILY_API_KEY`; the router endpoint and model are configurable in
`config/recommender.yaml`. Confirm the configured model ID is offered by your Nebius
account, or pass `--nemotron-model` with an available 120B or 550B model ID.
For a stage-by-stage diagnostic, run `notebooks/recommender_quickstart.ipynb`: its
HF, Tavily, and Nemotron cells save and reuse the same evidence. The key-status cell
checks the environment seen by WSL, which may differ from a Windows notebook kernel.
For fields not established by the Hub lookup, Tavily runs bounded, field-specific
searches (exact model, config, parameters, chat template, license, weight size,
parent model, runtime, and speculative decoding). Defaults are advanced search,
eight results per query, and eight searches per recommendation. A final Nemotron
pass receives the pinned HF scan and attributed Tavily evidence and returns
per-field `verified`, `inferred`, or `unknown` assessments; parent-model evidence
cannot verify a target model's license or chat template. Search failures and
missing Tavily credentials are recorded as trace events without exposing keys.

```bash
PYTHONPATH=src python -m hf_model_watch recommend "Score social posts quickly" --gpu-memory-gb 80 \
  --budget-usd-per-hour 5 --gpu-hourly-cost-usd 3 --license apache-2.0 \
  --framework-mode fast --model-id Qwen/Qwen3-4B --json
```

```bash
maorb@MaorLaptop:/mnt/c/Users/maorb/work/hf_validator$ PYTHONPATH=src python -m hf_model_watch recommend
     --nemotron-model nvidia/nemotron-3-super-120b-a12b     "Recommend similar models to openbmb/MiniCPM5-2B, prefer speculative
  decoding"  --max-prompt-chars 120000 --validation-max-tokens 6000
```

```bash
PYTHONPATH=src python -m hf_model_watch recommend--model-id Qwen/Qwen3-4B     --nemotron-model nvidia/nemotron-3-super-120b-a12b
  "Want to run the model fast to get solid result for log analysis, speculative decoding"  --max-prompt-chars 120000 --validation-max-tokens 6000
  --explore-family
```



When `--model-id` is omitted, the command discovers models from the original task
request. It runs a recent Hub scan, a same-family search when a reference model is
present, and a Gemma search. The existing Hub eligibility filter and score determine
the candidate pool. Tavily searches each candidate against the original request and
extracts up to two relevant source pages. Follow-up searches come from the
features named in the request and focus on the exact candidate. A direct statement
on its Hugging Face page can verify a feature; a credible source that names the
candidate can support an inference. Speculative decoding requires an exact
model, draft, and runtime pairing before it is marked verified. The terminal
then presents up to three recommendations and prompts for a selection. The selected model is saved in
`discovery.json`; **Optimize Inference + Benchmark** is a placeholder action only.

With `(nemotron_runner)` active at the repository root, start the interactive
terminal flow with a plain request:

```bash
PYTHONPATH=src python -m hf_model_watch recommend "Find a new model like Qwen3.5-4B with optional speculative decoding head"
```

This runs directly from the checkout and does not install the project or its GPU
dependencies. The command asks for the missing GPU memory, budget, GPU hourly cost,
license, and framework mode, then prompts you to select a recommended model. If those
values are already in the request or provided as options, it reuses them.

Omit inputs for terminal prompts. With `--json`, missing values are returned in
`needs_input` and no network request is made. Hourly GPU cost must be provided because
memory capacity cannot establish a price. Missing license, runtime, task fit, or cost
evidence appears as uncertainty or `insufficient_evidence`.

Benchmarking is a separate action:

```bash
hf-watch benchmark-recommendation artifacts/recommendations/<run>/recommendation.json
hf-watch benchmark-recommendation artifacts/recommendations/<run>/recommendation.json --confirm --dry-run
```

The first command reports preflight without submitting a job. `--confirm` authorizes
submission; `--dry-run` exercises the existing job path without cloud submission.
The confirmed worker executes the selected fixture at the pinned revision. It records
latency, output throughput, peak GPU memory, fixture score, and failures in
`benchmark_results.json` and SQLite. These small fixtures are smoke checks; their
scores are not broad task quality estimates. Transformers uses the configured worker
image. For vLLM, SGLang, or TRT-LLM, configure a matching image in
`config/recommender.yaml` with the corresponding server executable installed; preflight
blocks an unconfigured framework. The worker starts its local server from the pinned
Hugging Face snapshot and calls its completion endpoint. Starter task fixtures live in
`src/hf_model_watch/recommender/fixtures/`.

After synchronizing completed run folders, run
`hf-watch benchmark-report <recommendation.json> --results-dir <synced-run-root>`.
It writes `framework_comparison.json` beside the recommendation and records comparison
rows in SQLite. Runs awaiting results are listed explicitly.
