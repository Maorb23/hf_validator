# Optional inference optimization

Model selection completes a recommendation. The optional `optimization` package adds version 1
intents, references, normalized results, compatibility checks, and deterministic selection.
No optimization field is written until an optimization action is requested.

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
