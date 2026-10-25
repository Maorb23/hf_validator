# Optional inference optimization

Model selection completes a recommendation. The optional `optimization` package adds version 1
intents, references, normalized results, compatibility checks, and deterministic selection.
CLI and explicit-model flows retain opt-in execution. NanoClaw automatically saves preparation
metadata and validated files after selection; it requires explicit approval before submission.

Install the sibling `llm-optimizer` checkout with its `remote` extras in the same environment.
Set `HF_MODEL_WATCH_OPTIMIZER_DATABASE` to the existing controller SQLite database, and
`HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON` to controller execution defaults, for example:

```json
{
  "images": {"vllm": "your-optimizer-image", "sglang": "your-sglang-image"},
  "bucket": "your-results-bucket",
  "dry_run": true,
  "optuna": {"trials": 30},
  "server_args": {"vllm": ["max_num_seqs=[1,2,4,8]"]}
}
```

Images must contain the updated optimizer worker. Use the existing controller reconciliation
process to tick this database. The adapter calls submit, status, results, and cancel; the optimizer
controller owns Nebius, checkpoints, and S3. Calling submit queues a study; it does not tick the
controller. A live execution configuration explicitly sets `dry_run: false`. Credentials stay in
the controller environment. For multiple GPUs, supply a matching controller `preset`.

For a selected recommendation, save a new-study intent as `intent.json`:

```json
{
  "schema_version": 1,
  "mode": "new",
  "idempotency_key": "run-example-study-1",
  "context": {
    "model_id": "org/model",
    "revision": "SELECTED_MODEL_COMMIT_SHA",
    "gpu_type": "gpu-h100-sxm",
    "gpu_count": 1,
    "workload": {"input_tokens": 128, "output_tokens": 32, "num_prompts": 100, "seed": 1},
    "concurrency": 4,
    "constraints": []
  },
  "frameworks": ["vllm"],
  "selection_policy": {"name": "balanced"}
}
```

Use the model's immutable selected commit for revision. Explicit selection and revision checks
run before submission. Repeating the same intent returns the existing controller study. The
controller also persists idempotency keys, covering a crash between submission and reference save.
Reusing a key for a different execution request is rejected by the controller.

```bash
PYTHONPATH=src python -m hf_model_watch optimization start --run-id run_example --request-file intent.json
PYTHONPATH=src python -m hf_model_watch optimization status --run-id run_example
PYTHONPATH=src python -m hf_model_watch optimization list-prior-studies --run-id run_example --request-file context.json
PYTHONPATH=src python -m hf_model_watch optimization reuse --run-id run_example --request-file reuse.json
PYTHONPATH=src python -m hf_model_watch optimization cancel --run-id run_example
```

`context.json` contains just the context object. `reuse.json` contains `mode: "reuse"`,
`study_reference`, `context`, and `selection_policy`. The initial `FixtureStudyCatalog` is empty,
returning “no prior studies available”; applications can inject typed fixture summaries. A known
controller run can be reused directly by reference through a configured adapter. Reuse reads
measurements and records the analysis; it never submits, ticks, cancels, or resumes jobs. Continuing
trials remains an explicit optimizer `--continue` operation, outside the reuse tools.

NanoClaw exposes `start_optimization`, `get_optimization_status`, `list_prior_studies`,
`reuse_optimization_study`, and `cancel_optimization`, with the same validated contracts.

Concurrency 1 uses mean E2E latency and mean TTFT in milliseconds. Higher concurrency uses
output-token throughput in tokens/second and mean E2E latency. Workload size, prompt count, seed,
concurrency, GPU type/count, model/revision, objectives, and measurement constraints must match
before measurements are compared. Legacy manifests without context are marked incomparable.
Selection constraints can further narrow feasible results without changing measurements.

`latency_first`, `ttft_first`, and `throughput_first` compare the named objective first, then the
other objectives. A policy requiring an absent objective returns a reason. `balanced` minimizes
the weighted mean of normalized objective loss; `knee_point` minimizes the maximum weighted
normalized loss (a minimax regret definition of the knee). `constrained` applies supplied bounds
then uses balanced selection. Normalization uses feasible candidate minima/maxima. Ties use
framework name then configuration ID. Results include the full unconstrained Pareto front,
bests for each objective, the selected configuration or a no-feasible-result reason, importance
for each objective and framework, errors, and artifact references.

The optimizer accepts `optuna.objectives` as a list in remote requests and comma-separated metrics
in `--optuna-objective`. It creates a genuine Optuna study with one direction per metric. Scalar
requests retain their scalar behavior and existing result fields. New summary fields preserve all
complete trial vectors and compute parameter importance separately for each target with a seeded
fANOVA evaluator and an [explicit target for each objective](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.importance.get_param_importances.html). Importance is explicitly unavailable when measurements are insufficient or the
optional `scikit-learn` dependency is missing. Install that dependency to calculate importance.
Successful measurements remain analyzable when another framework or later trial fails.

### Scalar studies with pruning

The default objectives remain the concurrency-specific Pareto pair. Set
`context.objectives` explicitly to one supported metric, for example
`["mean_e2e_latency_ms"]`, for a scalar study. Execution defaults may then include:

```json
{"optuna": {"trials": 30, "sampler": "TPESampler", "pruner": "MedianPruner",
 "pruner_kwargs": {"n_startup_trials": 10, "n_min_trials": 3},
 "min_completed_requests": 64}}
```

Pruning is restricted to unconstrained scalar studies. Checkpoints report cumulative
metrics at 25%, 50%, and 100% of completed requests, delayed by the observation
minimum. Pruned trials cancel pending requests and clean up their server. They
are excluded from recommendation candidates. Worker images must contain the
updated optimizer. Search results still require independent full-budget validation
before being treated as validated deployment recommendations.

### NanoClaw connection and spending checks

`get_optimization_connection` verifies imports and an existing controller schema with a
read-only SQLite connection. It never creates a database, submits a study, or ticks the
controller. Connected is distinct from execution-ready, and the tool reports dry-run mode.
The operator may put `config/optimizer.json` in the read-only installed runtime:

```json
{"database":"/workspace/extra/hf-model-watch/optimizer-controller/optimizer-controller.sqlite",
 "execution":{"dry_run":true,"timeout_minutes":60,"optuna":{"trials":16}}}
```

The database directory must be explicitly allowlisted and mounted; no cloud credentials
are needed for the connection probe or queuing. An external controller environment owns
reconciliation and cloud credentials. Never start reconciliation merely to test connectivity.
Worker images and live execution defaults must be verified separately before live submission.

NanoClaw's `start_optimization` requires `user_approved: true`, `approved_gpu_type`,
`approved_gpu_hourly_cost_usd`, `approved_timeout_minutes`, and `max_cost_usd`. The MCP server
loads the exact persisted intent for the run rather than asking the model to reconstruct it.
These values must reflect actual user approval. The gate verifies the selected
revision, GPU count and price against the saved request, bounds potential concurrent
framework cost by the hourly budget, and bounds all framework job durations by the approved
study cost. The cost ceiling uses the user-supplied GPU price and the controller's job timeout;
it is not a provider billing guarantee and excludes non-GPU charges. A new hardware price
requires updated user inputs, not silently replacing the saved recommendation assumptions.

### Automatic preparation after selection

CLI selection (interactive or `hf-watch select`) and NanoClaw `select_model` now
prepare `optimization-prepared/intent.json`, `request.json`, and `research.json`
inside the recommendation run. Preparation does not submit or launch GPU work.
Python validates the intent and the llm-optimizer remote contract. Nemotron inspects
the pinned model card and sourced evidence; speculative heads require verified exact
target revision and configured runtime image support plus an official launch example.
Verified launch flags become serving arguments; inferred or unknown support is omitted.
The research file preserves source URLs and confidence labels.

Configure `HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON` (or `config/optimizer.json` as
above) with images, bucket, preset, timeout and Optuna defaults. The bucket must be supplied explicitly. Framework images and known GPU presets use
the defaults below unless overridden. Saved workload and GPU preferences are
reused. Incomplete preparation returns `needs_input`, `insufficient_evidence`, or
`preparation_failed`; it never implies a runnable request. NanoClaw can supply missing
choices to `prepare_optimization`; infrastructure choices go in its `execution` object.
For CLI retries, put only missing choices in a JSON file, for example:

```json
{"input_tokens":128,"execution":{"bucket":"operator-bucket","preset":"1gpu-16vcpu-96gb"}}
```

```bash
hf-watch optimization prepare --run-id RUN --request-file choices.json
```

A prepared result includes paths, the configuration summary and an exact next command.
Review these before running that command with `--user-approved`. NanoClaw continues
to require its explicit user approval and cost fields. Optuna execution remains entirely
in llm-optimizer. Runtime images must include that optimizer's current worker contract.

Discovery runs HF/Tavily/Nemotron research from model requirements without requiring
optimization settings. Optional GPU memory and budget constraints are enforced when
provided; otherwise deployment suitability stays unknown. License defaults to `any`
and framework discovery to `sweep`. No GPU price is invented for discovery.

After selection, preparation collects missing GPU platform, input/output token lengths,
concurrency, frameworks, pricing, budget and artifact settings. Saved preferences and
execution defaults satisfy known choices. Supply overrides with `--gpu-type`,
`--input-tokens`, `--output-tokens`, `--concurrency`, `--frameworks`, `--optimizer-bucket`,
`--optimizer-preset`, and `--optimizer-images`, or through `prepare_optimization`.
Preparation can return `needs_input` while the recommendation remains complete.
Explicit-model inspection retains its existing intake. Preparation never authorizes
submission or GPU work.


Default worker images use registry `cr.eu-north1.nebius.cloud/e00p24e8mwqab11v1e`,
repositories `vllm`, `sglang`, and `trtllm`, with tag `sf261025-1`.
These short image references fit Nebius's 64-character limit. Selecting those
frameworks requires no image JSON. Explicit `--optimizer-images` or operator
configuration overrides these defaults. Infiro's image digests and aliases remain
configured separately in `config/platform-images.json` and
`config/platform-image-aliases.json`; its explicit images take precedence over
these CLI defaults. Existing saved requests retain their original images.

| Platform | 1 GPU default | 8 GPU default |
|---|---|---|
| `gpu-h100-sxm` | `1gpu-16vcpu-200gb` | `8gpu-128vcpu-1600gb` |
| `gpu-h200-sxm` | `1gpu-16vcpu-200gb` | `8gpu-128vcpu-1600gb` |
| `gpu-l40s-a` | `1gpu-8vcpu-32gb` | Requires explicit preset |
| `gpu-l40s-d` | `1gpu-16vcpu-96gb` | Requires explicit preset |

Other GPU counts/platforms require a preset. CLI users can press Enter to accept a
known default or enter a custom preset; `--optimizer-preset` and configured presets
are preserved. NanoClaw uses the same defaults without extra clarification questions.


### Preemptible default and saved spot pricing

New hf-model-watch requests default to `"preemptible": true` and
`"follows_spot_price": true`. The configuration summary displays both values.
If `NEBIUS_SPOT_PRICING_POLICY_ID` is configured during preparation, its value is
saved as `spot_pricing_policy_id` and `follows_spot_price` becomes false. Explicit
execution settings override these defaults:

```json
{"preemptible":true,"spot_pricing_policy_id":"YOUR_POLICY_ID","follows_spot_price":false}
```

A request must use either a policy ID or follow spot pricing. Set
`{"preemptible":false}` to opt out when no pricing fields are configured; remove any
spot policy and set `follows_spot_price` to false when overriding an existing spot
configuration. Generated JSON pins the reviewed policy, so later controller environment
changes cannot silently replace it. Example requests explicitly follow spot pricing.
The updated llm-optimizer controller contract is required; scheduler pricing fields
are omitted from worker payloads for compatibility with the existing images.
Preparation still never launches jobs; submission requires explicit approval.

### GPU price and concurrent framework budgets

The default GPU hourly price estimate is $1 per GPU-hour. The CLI shows this as
an editable default; JSON/NanoClaw and preparation reuse the same default when
no price is saved or supplied. Explicit prices and saved workload preferences take
precedence. This is a user-provided planning estimate, not a live provider quote.

For three frameworks on one GPU each, $1 per GPU-hour means a potential $3/hour
GPU cost, which fits a $5/hour budget. The existing optimizer controller supports
concurrent jobs; run it with `--concurrency 3` to allow all three simultaneously.
Benchmark request concurrency (for example 128) is separate from GPU job concurrency.
Total study cost still includes every framework's full timeout.

An existing run with a saved $5 GPU price retains that value until explicitly
updated. Run `hf-watch optimization prepare --run-id RUN_ID --gpu-hourly-cost-usd 1`
to correct the estimate and prepare the selected model without rerunning discovery.
A choices JSON file with `{"gpu_hourly_cost_usd":1}` is also supported.

Terminal numeric answers are validated and saved as numbers. Preparation also normalizes
numeric strings in older saved runs. At the price prompt, press Enter to accept `$1`;
typing `5` explicitly selects `$5` per GPU-hour, independently of the total hourly budget.

### Expanded optimizer search profile

Preparation now calls `llm_optimizer.search_profiles.expanded_server_args` for vLLM,
SGLang and TRT-LLM; hf-model-watch does not implement or execute the search itself.
The profile shares `EXPANDED_CHOICES` with the optimizer's experiment lab. Its full
remote search uses the requested domains, while the lab's existing capped experiment
grids remain unchanged:

- Prefill regime (concurrency 1): token budgets 1024–8192, step 1024; serving capacity 1.
- Batched regime (any concurrency greater than 1, including 32): token budgets
  1024–32768, step 256; capacities 8–128, step 8; framework-specific memory fractions.
- Expanded choices include scheduling, attention, caching, KV dtype and graph settings
  for the relevant framework. Synthetic graph knobs are rendered as native settings.

New SGLang requests from CLI and web preparation share decode graph choices
`["disabled", "full"]` and scheduling policies `["fcfs", "lpm"]`. Prefill graph
choices remain `["disabled", "breakable"]`, and attention backends remain
`["triton", "fa3", "flashinfer"]`. Existing saved requests and studies retain
their reviewed arguments. Restart long-lived web/preparation processes after
deploying the updated optimizer package. Worker images include a copy of the
profile, but execute the request's saved arguments without regenerating its
search space; this choices-only update requires no worker image publication or
GPU runtime rebuild.

Nemotron research does not set context-length overrides such as `max_model_len`;
runtime defaults apply unless the user explicitly configures a value. Research
settings are restricted to framework-specific argument names, repeated proposals
are deduplicated, and conflicting proposals are omitted. Explicit global and
per-framework settings take precedence over research. These rules apply to new
preparations; saved requests and existing approvals are not rewritten.

The benchmark's actual concurrency, input/output lengths, prompt count and seed remain
fixed in `client_args`. The selected objectives are unchanged. Default Optuna settings
are 30 trials with `TPESampler`; configured global/per-framework settings take precedence.
Explicit serving settings override overlapping profile dimensions. A fixed TRT YAML
`config` overrides the coupled generated batch/config dimension.

SGLang uses its existing valid cache/scheduler pair menu. TRT-LLM nested KV-cache,
scheduler and CUDA-graph settings are serialized into `server_config_files`, coupled
with batch size so graph captures never exceed the selected capacity. The worker
materializes these files locally and passes actual `--config` paths to the server.
TRT token budgets smaller than the input length are omitted because chunking-off
trials must fit a full prompt. Oversized inputs with no valid budget fail preparation.
Large worker payloads use compressed transport to avoid command-line size limits.

**Deploy updated optimizer workers before launching this profile.** The older default
image tags have not been verified to contain `server_config_files` materialization or
compressed transport support. Rebuild the serving images with the updated optimizer,
then configure their image references explicitly. Preparation reports these worker
requirements and a concise list of search dimensions; all actual domains are saved
in the request JSON. No model-card evidence is replaced by a search-profile assumption.

Existing runs can regenerate their request with
`PYTHONPATH=src python -m hf_model_watch optimization prepare --run-id RUN_ID`.
This invalidates earlier approvals and still does not submit or launch a GPU job.

### Managed local controller

This NanoClaw installation uses a managed user service rather than a manually started
`serve` terminal. Approved submissions wait for controller readiness; a stopped service
is activated automatically through a startup signal in the shared database directory.
Read-only probes and preparation do not activate it. Other installations retain manual
controller behavior unless a `.controller-managed.json` registration exists beside the
database. Both `tick` and `serve` acquire the same database lease, preventing overlapping
reconciliation. See [managed controller operations](managed-optimizer-controller.md).
