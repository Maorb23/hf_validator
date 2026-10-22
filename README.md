# hf-model-watch

Discover Hugging Face models, validate them, and prepare inference optimization studies.
Use the CLI directly or access recommendation and optimization through NanoClaw.

| Module | Purpose | Main code |
| --- | --- | --- |
| Recommendation | Find models for a workload using Hub metadata, Tavily evidence, and Nemotron. | `recommender/`, `hub/` |
| Validation and evaluation | Run a model, measure latency/quality/context behavior, and optionally evaluate PEFT adaptation. | `runtime/`, `evals/`, `finetune/` |
| Inference optimization | Prepare requests, submit approved studies to llm-optimizer, and compare results. | `optimization/`, `orchestration/optimization_adapter.py` |

Code paths above are relative to `src/hf_model_watch/`. The previous detailed README is
preserved in [instructions.md](instructions.md).

## Setup

Use Python 3.11+ on Linux or WSL. For recommendation and optimization preparation, a CPU
is enough; this setup avoids installing local GPU/training dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_cpu.txt
cp .env.example .env
```

In `.env`, configure `NEBIUS_API_KEY` for Nemotron, `TAVILY_API_KEY` for web evidence,
and optionally `HF_TOKEN`. Never commit populated credentials.

Automatic optimization preparation also requires the updated llm-optimizer package in
this environment. If its checkout is at `../bento_extend/llm-optimizer`:

```bash
python -m pip install -e "../bento_extend/llm-optimizer[remote]"
```

Run commands from this repository's root. `PYTHONPATH=src python -m hf_model_watch`
works without installing the full project; an installed environment also provides `hf-watch`.

## 1. Recommend a model

```bash
PYTHONPATH=src python -m hf_model_watch recommend \
  "Recommend a 4B model for coding, prefer speculative decoding" \
  --nemotron-model nvidia/nemotron-3-super-120b-a12b
```

The CLI asks for missing workload, GPU, framework, storage, and budget choices. Select a
model from the shortlist; preparation then saves its exact revision and optimizer request.
Use `--model-id Qwen/Qwen3-4B` for an explicit-model assessment, or `--json` for noninteractive
output with clarification questions. Evidence distinguishes verified, inferred, and unknown claims.

The GPU price prompt is **per GPU-hour**; the budget is the **total hourly limit**.
For three concurrent one-GPU framework jobs at $1 each, allow at least $3/hour.
Benchmark concurrency (for example 32 requests) is a separate setting.

Run files are stored under `artifacts/recommendations/<run-id>/`. The
`optimization-prepared/` directory contains `intent.json`, `request.json`, `research.json`,
and the pinned model card. Unknown speculative-head compatibility never enables a head.

More: [preferences](docs/recommender_preferences.md),
[quickstart notebook](notebooks/recommender_quickstart.ipynb).

## 2. Validate and evaluate a model

Local inference needs the full project environment and sufficient hardware. Install with
`uv sync`, then start with a tiny model and skip fine-tuning:

```bash
uv run hf-watch run-local \
  --model-id sshleifer/tiny-gpt2 --config-dir config --skip-finetune

uv run hf-watch leaderboard \
  --db results/hf_model_watch.sqlite --out reports/leaderboard.md
```

This downloads model weights and runs local checks. Results are saved under `artifacts/`
and recorded in SQLite. Evaluation prompts live in `config/eval_tasks.yaml`; optional
adaptation is configured in `config/finetune.yaml`. Fine-tuning needs the `finetune` extra
and compatible GPU hardware. See the [detailed guide](instructions.md) for remote validation,
quality datasets, reporting, and PEFT setup.

## 3. Optimize inference

hf-model-watch prepares and reviews the request; **llm-optimizer runs Optuna and owns
GPU jobs, checkpoints, and result storage**. The expanded profile covers vLLM, SGLang,
and TRT-LLM. Client workload stays fixed; the default objectives maximize throughput
and minimize mean end-to-end latency. Concurrency 32 uses the batched search regime.

Preparation defaults to a dry run. Preemptible jobs use your configured
`NEBIUS_SPOT_PRICING_POLICY_ID`, or follow spot price when no policy ID is configured.
Review the printed summary and request before submitting.

To prepare a live request for an existing selection, replace the run ID below:

```bash
RUN_ID=run_your_saved_id
export HF_MODEL_WATCH_OPTIMIZER_DATABASE="$PWD/optimizer-controller.sqlite"
echo '{"execution":{"dry_run":false}}' > /tmp/hf-live.json

PYTHONPATH=src python -m hf_model_watch optimization prepare \
  --run-id "$RUN_ID" --request-file /tmp/hf-live.json
```

**Before launching:** configure Nebius authentication and S3 bucket credentials. GPU
images must contain the updated optimizer worker, including TRT config-file materialization
and compressed-payload support. Installing locally does not update container images;
the older default image tags have not been verified to include these changes. Rebuild
and configure updated images before preparing a live request.

After reviewing the regenerated request, explicitly approve and queue it:

```bash
PYTHONPATH=src python -m hf_model_watch optimization start \
  --run-id "$RUN_ID" \
  --request-file "artifacts/recommendations/$RUN_ID/optimization-prepared/intent.json" \
  --user-approved
```

Then start the controller using the same database. **This launches live GPU jobs** when
`dry_run` is false; `--concurrency 3` allows three framework jobs at once:

```bash
python -m dotenv run -- python -m llm_optimizer.remote \
  --database "$HF_MODEL_WATCH_OPTIMIZER_DATABASE" --concurrency 3 serve
```

Keep the controller running. See [optimization.md](docs/optimization.md) for overrides,
status/results, approval checks, search settings, and worker requirements. Regenerate
through `optimization prepare` rather than hand-editing an approved request.

## NanoClaw and further help

NanoClaw provides a conversational interface to the same recommendation and optimization
flows, with saved workload preferences and explicit submission approval. Setup:
[docs/nanoclaw.md](docs/nanoclaw.md). Full original project instructions:
[instructions.md](instructions.md).

```bash
PYTHONPATH=src python -m hf_model_watch --help
PYTHONPATH=src python -m hf_model_watch optimization --help
```
