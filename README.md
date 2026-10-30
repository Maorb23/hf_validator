# hf-model-watch

`hf_validator` contains **hf-model-watch**, a toolkit for discovering Hugging Face
models, validating their behavior, and optimizing inference on Nebius GPUs. Use
it from the local CLI or the Infiro web application.

## Run jobs on Nebius

Choose the execution path for your task:

| Task | Entry point | What launches GPU jobs |
| --- | --- | --- |
| Optimize through the website | [Local web app](#run-the-web-app-locally), then prepare and approve a study | The separate `platform_worker`, using the optimizer controller and Nebius SDK |
| Optimize from the CLI | [Optimization commands](#3-inference-optimization) | The `llm-optimizer` controller processing your approved study |
| Scan and validate models remotely | [Validation orchestrator](#run-the-validation-orchestrator-locally) | The local controller calling the Nebius CLI |

Running a controller locally can launch paid remote GPUs. Recommendation and
request preparation do not allocate GPUs. Dry runs exercise the workflow without
measuring real inference performance.

Live execution requires a Nebius project, GPU quota, service-account job permissions,
a private S3-compatible artifact bucket, and compatible worker images. Gated models
also require an HF token and acceptance of their access terms.

**Inference access and GPU-job access are separate.** `NEBIUS_API_KEY` authenticates
the hosted recommendation model. GPU jobs require IAM/service-account credentials.
For web jobs, save credentials in [Connections](#configure-connectors). CLI jobs
use their controller environment and execution configuration.

### Run the validation orchestrator locally

Use Python 3.11+ on Linux or WSL and install `uv`. Run commands from the repository
root. This installs the full project, including GPU/runtime dependencies;
CPU-only recommendation and web setup are documented below.

```bash
uv sync
# Fresh installation only; preserve an existing .env:
cp .env.example .env
```

Configure these before a live run:

| Location | Settings |
| --- | --- |
| `.env` | `NB_PROJECT_ID`, service-account authentication, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `HF_TOKEN` |
| `config/serverless.yaml` | Worker image, GPU platform, matching preset, GPU count, timeout |
| `config/storage.yaml` | Your S3 bucket, prefix, endpoint, `enabled: true`, and `dry_run: false` |
| `config/scan.yaml` | Scan window, eligibility filters, maximum candidates |
| `config/finetune.yaml` | `enabled: false` to skip adaptation; `wandb_enabled: false` if not using W&B |

Build and publish the root `Dockerfile` to a registry accessible by your project.
Set that image in `config/serverless.yaml` or `HF_MODEL_WATCH_NEBIUS_IMAGE`. Replace
the checked-in deployment's image and bucket with your own. Keep the YAML bucket
and prefix consistent with `AWS_S3_BUCKET` and `AWS_S3_PREFIX` if set: result
synchronization reads `config/storage.yaml`.

The bootstrap supports these authentication alternatives:

- `NB_PROJECT_ID`, `SA_ID`, `PUBLIC_KEY_ID`, and `NEBIUS_PRIVATE_KEY_PATH` or `NEBIUS_PRIVATE_KEY_B64`.
- `NB_PROJECT_ID` and `NEBIUS_SERVICE_ACCOUNT_KEY_B64`.
- `NB_PROJECT_ID` and `NEBIUS_CONFIG_YAML_B64` containing an existing CLI configuration.

The direct CLI command needs an installed, authenticated Nebius CLI. The bootstrap
can install it and create a service-account profile. Use the default profile name
`hf-model-watch-controller` and ensure it is the active CLI profile; job commands
do not pass a profile flag. Never commit populated environment files or keys.

First, run the controller with a fixture candidate and simulated jobs:

```bash
uv run hf-watch prefect-flow --config-dir config --dry-run
```

This writes local reporting state without contacting Nebius or scanning the Hub.
It does not verify live authentication, GPU capacity, or image compatibility.

For a live run, change the following `.env` settings:

```dotenv
DRY_RUN=false
HF_MODEL_WATCH_DRY_RUN=false
HF_MODEL_WATCH_WORKER_DRY_RUN=false
HF_MODEL_WATCH_OBJECT_STORE_DRY_RUN=false
```

Load the environment and run the bootstrap on an already provisioned host:

```bash
set -a
source .env
set +a
HF_MODEL_WATCH_SKIP_SYSTEM_DEPS=1 bash scripts/run_prefect_controller_job.sh
```

The bootstrap runs `uv sync`. Without `HF_MODEL_WATCH_SKIP_SYSTEM_DEPS=1`, it also
attempts system package installation on hosts with `apt-get`.

If the environment and Nebius CLI authentication are already configured, run directly:

```bash
uv run hf-watch prefect-flow --config-dir config
```

**Both live commands can allocate paid GPUs.** The flow scans, scores, selects
candidates, submits jobs, follows logs, and imports completed results. It currently
submits and monitors one candidate at a time; `max_parallel_jobs` does not create
parallel submissions in this flow. Fine-tuning is enabled in the checked-in config.

Keep the terminal running. Trace events include run and job IDs. Inspect a printed
job ID with:

```bash
nebius ai job get YOUR_JOB_ID --format json
nebius ai job logs YOUR_JOB_ID --follow
```

Artifacts are uploaded to `s3://<bucket>/<prefix>/<run_id>/`. Refresh local reports:

```bash
uv run hf-watch sync-nebius-results --config-dir config
uv run hf-watch leaderboard \
  --db results/hf_model_watch.sqlite --out reports/leaderboard.md
```

Defaults are `outputs/synced-nebius/` for downloaded reporting artifacts,
`results/hf_model_watch.sqlite` for the database, and `reports/leaderboard.md`
for the leaderboard. Stopping the local process does not cancel remote jobs;
manage validation jobs through Nebius. Use the study cancel command below for
optimizer jobs.

For optional scheduling, see [Nebius and Prefect](instructions.md#nebius-and-prefect).
For worker bootstrap checks, see [dry runs](instructions.md#dry-runs).

## The three core modules

| Module | Purpose | Main code under `src/hf_model_watch/` |
| --- | --- | --- |
| Recommendation | Discover models for a workload or assess a specific model | `recommender/`, `hub/` |
| Validation and evaluation | Run inference checks, measure latency/quality/context behavior, optionally evaluate PEFT adaptation | `runtime/`, `evals/`, `finetune/` |
| Inference optimization | Prepare and approve studies and compare framework configurations | `optimization/`, `orchestration/optimization_adapter.py` |

The web application in `web/` adds accounts, connections, datasets, configuration
review, approved optimization studies, and results through `/app`.

### 1. Recommendation

Use this lightweight CPU environment for local recommendation:

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements_cpu.txt
# Fresh installation only:
cp .env.example .env
```

Configure `NEBIUS_API_KEY`, `NEBIUS_API_BASE_URL`, and `TAVILY_API_KEY` in `.env`;
add `HF_TOKEN` when needed. The CLI loads the root `.env`. Confirm the hosted model
configured in `config/recommender.yaml` is available on your inference endpoint.

Discover models from a workload description:

```bash
PYTHONPATH=src python -m hf_model_watch recommend \
  "Recommend a 4B model for coding, prefer speculative decoding" \
  --nemotron-model nvidia/nemotron-3-super-120b-a12b
```

Assess a specific model:

```bash
PYTHONPATH=src python -m hf_model_watch recommend \
  "Assess this model for coding" --model-id Qwen/Qwen3-4B
```

The interactive flow asks for missing workload, hardware, framework, storage, and
budget choices, then lets you select a model. Add `--json` for noninteractive output,
including clarification questions when inputs are missing. Runs are saved under
`artifacts/recommendations/<run-id>/`.

Selection pins the revision and attempts optimization preparation, which also needs
the optimizer package and execution settings below. Prepared files include
`optimization-prepared/intent.json`, `request.json`, `research.json`, and the pinned
model card. Inspect preparation's status; missing inputs do not produce a runnable study.

Evidence retains source URLs and verified, inferred, or unknown status. A web
mention does not establish runtime support, and unknown speculative-head compatibility
does not enable a head. See [preferences](docs/recommender_preferences.md) and the
[quickstart notebook](notebooks/recommender_quickstart.ipynb).

### 2. Validation and evaluation

Use the [validation orchestrator](#run-the-validation-orchestrator-locally) for
remote evaluation. For a small local smoke test in the full `uv sync` environment:

```bash
uv run hf-watch run-local \
  --model-id sshleifer/tiny-gpt2 --config-dir config --skip-finetune
uv run hf-watch leaderboard \
  --db results/hf_model_watch.sqlite --out reports/leaderboard.md
```

This downloads weights and runs local checks, writing artifacts and SQLite results.
Larger models need sufficient hardware. Prompts live in `config/eval_tasks.yaml`;
adaptation settings live in `config/finetune.yaml`. Fine-tuning requires
`uv sync --extra finetune` and compatible hardware. See the
[detailed guide](instructions.md) for quality datasets, PEFT, and reporting.

### 3. Inference optimization

hf-model-watch prepares and reviews requests. **llm-optimizer owns Optuna, GPU jobs,
checkpoints, and result storage.** Study frameworks include vLLM, SGLang, and
TensorRT-LLM, subject to compatible images and model/runtime support.

Install the updated sibling optimizer checkout with remote extras in the CLI environment:

```bash
python -m pip install -e "../bento_extend/llm-optimizer[remote]"
export HF_MODEL_WATCH_OPTIMIZER_DATABASE="$PWD/optimizer-controller.sqlite"
```

Use the same database for preparation, submission, and the controller. If you already
have a managed controller, use its existing database. Configure execution defaults
with `HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON` or `config/optimizer.json`: your
bucket, compatible framework images, matching GPU preset, timeout, and trial budget.
The validation worker image is separate from optimizer framework images.
See [optimization configuration](docs/optimization.md) for schemas and overrides.

After selecting a recommendation, replace the saved run ID:

```bash
RUN_ID=run_your_saved_id
PYTHONPATH=src python -m hf_model_watch optimization prepare --run-id "$RUN_ID"
```

Preparation defaults to a dry run. Inspect its status, summary, and saved request.
Supply missing choices through `--request-file choices.json`. To prepare for live
execution, save this override as `live-choices.json`:

```json
{"execution":{"dry_run":false}}
```

Regenerate and review the request:

```bash
PYTHONPATH=src python -m hf_model_watch optimization prepare \
  --run-id "$RUN_ID" --request-file live-choices.json
```

Live execution requires controller credentials, storage, and updated worker images.
Installing code locally does not update remote images. Check the
[worker requirements](docs/optimization.md) before using image defaults.
Preemptible requests use `NEBIUS_SPOT_PRICING_POLICY_ID` when configured;
otherwise they follow spot price.

After reviewing the regenerated files, explicitly approve and queue the study:

```bash
PYTHONPATH=src python -m hf_model_watch optimization start \
  --run-id "$RUN_ID" \
  --request-file "artifacts/recommendations/$RUN_ID/optimization-prepared/intent.json" \
  --user-approved
```

Submission queues work. For a foreground controller, load the root `.env` and run:

```bash
set -a
source .env
set +a
python -m llm_optimizer.remote \
  --database "$HF_MODEL_WATCH_OPTIMIZER_DATABASE" --concurrency 3 serve
```

**This launches paid GPU jobs when the queued request has `dry_run: false`.** Keep
the controller running. Three slots permit three simultaneous framework jobs.
Benchmark request concurrency is a separate setting. GPU price is per GPU-hour;
the hourly budget covers the combined allocation. Three one-GPU jobs at $1/GPU-hour
require $3/hour while running together.

On the previously provisioned host, the managed controller is
`hf-model-watch-optimizer.service`. Check it before starting another controller:

```bash
systemctl --user status hf-model-watch-optimizer.service
journalctl --user -u hf-model-watch-optimizer.service -n 50 --no-pager
```

This host-specific service is not installed by cloning the repository. Use its
configured database and environment; restart it after environment changes.
Run one controller per database. Stopping it does not cancel running GPUs.

Inspect or cancel a CLI study:

```bash
PYTHONPATH=src python -m hf_model_watch optimization status --run-id "$RUN_ID"
PYTHONPATH=src python -m hf_model_watch optimization cancel --run-id "$RUN_ID"
```

Status reads controller state. See [the optimization guide](docs/optimization.md)
for result storage, analysis, comparability rules, and advanced search settings.

## Run the web app locally

Use Python 3.11+, Node/npm compatible with `web/package.json`, and two terminals.
The web environment is CPU-only; GPU execution happens in remote workers.

### First-time setup

From the repository root:

```bash
python3 -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -r web/requirements-railway.txt
python web/scripts/platform_env.py \
  --output "$HOME/.config/infiro/local.env" \
  --data-dir "$HOME/.local/share/infiro"
npm --prefix web ci
npm --prefix web run build
```

The requirements install the packaged optimizer snapshot. For development against
a sibling checkout, install a compatible optimizer in the environment used by both
Django and the worker; see [platform startup](docs/authenticated-platform.md#local-startup).

The helper creates persistent session/encryption keys, database/artifact paths,
and `INFIRO_LIVE_ENABLED=0`. **Create the file once and reuse it.** Keep it private
and back up keys separately from the database; replacement keys cannot decrypt
existing connections.

### Start the server and worker

**Terminal 1 — website**, from the repository root:

```bash
source .venv-web/bin/activate
export PYTHONPATH="$PWD/src:$PWD/web"
export INFIRO_ENV_FILE="$HOME/.config/infiro/local.env"
python web/manage.py migrate --noinput
python web/manage.py runserver 127.0.0.1:8000 --noreload
```

**Terminal 2 — worker**, from the same repository root:

```bash
source .venv-web/bin/activate
export PYTHONPATH="$PWD/src:$PWD/web"
export INFIRO_ENV_FILE="$HOME/.config/infiro/local.env"
python web/manage.py platform_worker
```

Open **http://127.0.0.1:8000/app/register**, or `/app/login` for an existing account.
Use the same hostname consistently. Leave both terminals running; stop each with
Ctrl+C. The worker handles queued validation, research, preparation, and approved
study operations. Web studies use the application database and do not need the
separate CLI `serve` process or CLI controller database.

For later starts, reuse the environment file and repeat the two terminal commands.
After frontend edits, rebuild with `npm --prefix web run build` and restart Django.
Restart both processes after application environment changes. With `DATABASE_URL`
unset, this setup uses local SQLite, independent of hosted installations.

### Configure connectors

For **CLI usage**, edit the root `.env` copied from `.env.example`. For **web usage**,
open **http://127.0.0.1:8000/app/connections**, save the needed connections, and
validate them. The website encrypts per-user credentials. Web operations receive
credentials from saved connections; root `.env` credentials are not imported.

| Connector | CLI environment/configuration | Website fields |
| --- | --- | --- |
| Nebius inference | `NEBIUS_API_KEY`, `NEBIUS_API_BASE_URL`; optional `NEMOTRON_MODEL` | Token Factory API key and HTTPS inference endpoint |
| Nebius jobs | `NB_PROJECT_ID` and controller authentication described above | Project ID, service-account ID, public-key ID, private key (PEM/base64 PEM); alternatively a temporary IAM token |
| Hugging Face | `HF_TOKEN` | HF read token; accept required gated model/dataset terms in HF |
| Object storage | AWS access key/secret and endpoint; validation bucket/prefix in `config/storage.yaml`, optimizer bucket in execution settings | Permanent access key ID, secret key, bucket, region, HTTPS endpoint, worker secret reference |
| Tavily | `TAVILY_API_KEY` | Research API key |

The web jobs connection uses the Nebius SDK and refreshes tokens in service-account
mode; it does not read a CLI profile. Manual IAM tokens expire. Validation checks
provider authentication/read access, such as model listing, job listing, HF identity,
and S3 `HeadBucket`; it does not establish GPU quota, job creation, or object writes.

Application settings belong in the file selected by `INFIRO_ENV_FILE`, loaded by
both processes:

| Setting | Purpose |
| --- | --- |
| `DJANGO_SECRET_KEY`, `INFIRO_CREDENTIAL_KEYS` | Persistent sessions and encrypted connections; generated by the helper |
| `INFIRO_DATABASE`, `INFIRO_PRIVATE_ROOT` | Local database and private artifact paths |
| `DATABASE_URL` | Optional PostgreSQL connection; overrides the local database |
| `INFIRO_LIVE_ENABLED` | `0` for dry-run-only operation; `1` enables explicitly approved live studies |
| `INFIRO_SERVING_IMAGES`, `INFIRO_IMAGE_ALIASES` | Optional JSON overrides for checked-in image/digest and alias mappings |
| `INFIRO_WANDB` | Optional JSON with `project`, optional `entity`, and immutable `secret_version_id` containing `WANDB_API_KEY` |

For live workers, provision Nebius SecretStash versions with storage credentials
and, when connected, `HF_TOKEN`. Enter references on the corresponding Connections
forms. The jobs service account needs job permissions, registry access, secret
metadata reads, and payload-read access (`mysterybox.payload-viewer`). Preparation
verifies references and pins immutable versions. Workers receive secret references
rather than inline credentials. See [the platform guide](docs/authenticated-platform.md)
for live provider requirements and image compatibility.

### Run a web study

1. Save and validate your connections.
2. Discover or select a model, then add and validate an HF dataset with its relative
   JSON file. Supported formats are ShareGPT, Alpaca, and text `messages`.
3. Choose GPU resources, frameworks, workload sizes, concurrency, trials, and timeout.
4. Prepare and review the pinned model, frozen workload, images, cost assumptions,
   and connection references.
5. Explicitly approve the study. The worker processes it; monitor progress, trials,
   diagnostics, and result downloads on the study page.

Start with a dry run. For an offline demonstration after registering, use the same
environment exports as the server:

```bash
python web/manage.py platform_demo --username YOUR_USERNAME
```

Open its printed review URL and approve the dry run. It uses labeled fixtures,
makes no provider calls, allocates no GPU, and produces no measured inference metrics.

For live execution, verify images, worker compatibility, quota, storage, and secret
permissions. Set `INFIRO_LIVE_ENABLED=1`, restart both processes, and prepare a fresh
review. Approval authorizes remote allocation. Cancel studies through the app;
closing the browser or stopping Django does not cancel provider jobs.

Inspect saved framework log tails with the same environment:

```bash
python web/manage.py study_logs --latest
```

## Further documentation

- [Website setup and frontend development](web/README.md)
- [Private platform and provider requirements](docs/authenticated-platform.md)
- [Study diagnostics and W&B](docs/platform-study-diagnostics.md)
- [Optimization configuration and analysis](docs/optimization.md)
- [Custom HF optimization workloads](docs/optimization-hf-datasets.md)
- [Detailed validation and reporting guide](instructions.md)
- [Railway deployment](docs/web/railway.md)

CLI help:

```bash
PYTHONPATH=src python -m hf_model_watch --help
PYTHONPATH=src python -m hf_model_watch recommend --help
PYTHONPATH=src python -m hf_model_watch prefect-flow --help
PYTHONPATH=src python -m hf_model_watch optimization --help
```
