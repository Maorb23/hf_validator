# Gemma notebook and remote worker comparison

Inspected on 2026-10-30. Reference notebook:
`../bento_extend/llm-optimizer/docs/optuna_inference_optimization_lab_v2.ipynb`.
Earlier comparisons used the root-level updated notebook instead; this v2 file
and the execution artifacts beside it establish actual successful Gemma runs.

## Confirmed successful reference executions

All three reference executions below use `google/gemma-4-12B-it` at commit
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, report `status=complete`, and account
for 1000 completed requests with zero failed requests. Paths are relative to the
sibling optimizer's `experiment_results/` directory:

- SGLang: `sglang-phase-b-pruning-gemma4-12b/B-sglang-c128-e2e-gp-warmstart-20261022T180209567084-B-sglang-c128-measured-attempts/a13c0b6b75f448a6a238c3a01f961727/execution.json`
- vLLM: `vllm-phase-b-pruning-gemma4-12b/B-vllm-c128-e2e-tpe-s11-r3-pruner-median-B-vllm-c128-measured-attempts/0ce11259c83c4de98ce96bd19cffce1d/execution.json`
- TRT: `trtllm-phase-b-pruning-gemma4-12b/B-trtllm-c128-e2e-tpe-s11-r2-pruner-median-B-trtllm-c128-measured-attempts/172f230e80454db486509353d62f0c05/execution.json`

| Component | Successful reference metadata | Remote W&B requirements.txt |
| --- | --- | --- |
| SGLang | 0.5.22 | 0.5.18 |
| Transformers with SGLang | 5.19.0 | 5.12.1 |
| vLLM | 0.32.0 | 0.30.0 |
| Transformers with vLLM | 5.19.0 | 5.17.0 |
| TensorRT-LLM | 1.3.0rc26 | 1.3.0rc26 |
| Transformers with TRT | 5.5.4 | 5.5.4 |

Remote requirements were retrieved read-only from these W&B runs:

- [SGLang](https://wandb.ai/maorblumberg-tel-aviv-university/nebius-optimization-jobs/runs/420f2fd314b1b349265afbbd)
- [vLLM](https://wandb.ai/maorblumberg-tel-aviv-university/nebius-optimization-jobs/runs/a9ee585aff688e956b14c59a)
- [TRT](https://wandb.ai/maorblumberg-tel-aviv-university/nebius-optimization-jobs/runs/f1c5d4b09907d37000a93a35)

W&B finished state alone does not prove any trial completed successfully. Earlier
remote SGLang run `e90aaf1b08ad24d86c63b7ee` records the same older versions.

## Optimizer source parity

The sibling checkout and shipped 915beb35 archive are byte-identical for:
`experiment/lab.py`, `experiment/hf_models.py`, `experiment/stability.py`,
`optimization.py`, `cli.py`, `server_utils.py`, `search_profiles.py`, and
`remote/worker.py`.

The notebook invokes `RepositoryBenchmark`, which ultimately calls shared
`run_single_configuration`; the remote flow invokes CLI Optuna, which also calls
that runner. Shared optimizer code does not replace installed `sglang`, `vllm`,
Transformers, CUDA kernels or psutil inside an existing GPU image. The source
comparison establishes local/vendor parity, not the exact source hash of every
installed file inside a remote image.

## Execution differences and implications

The notebook's `configure_hf_models` downloads the full pinned snapshot before
creating benchmark servers. Successful commands serve the local snapshot path.
Remote commands serve the repository ID with a pinned revision, and the logs
show approximately 120 seconds of weight download inside the existing readiness
window. The shared readiness implementation remains 60 attempts with five-second
poll sleeps. Sample successful reference readiness times, after snapshots were
available, were approximately 36 seconds (SGLang), 180 seconds (vLLM), and 80
seconds (TRT). Cold starts and different warmup choices are not equivalent.

The successful reference SGLang command used Triton, disabled CUDA graphs,
server capacity 24, memory fraction 0.75 and prefill chunk 1536. It served the
same Gemma unified architecture and model revision. The remote memory limits
allow allocation, but the supplied logs then crash in SGLang's own Gemma code on
missing `lm_head_is_tied`. The older SGLang/Transformers runtime is a confirmed
mismatch; the precise runtime fix must be checked against source rather than
assuming a new Optuna parameter repairs the missing attribute.

The remote vLLM profile admits 1024-token batch budgets, while its multimodal
initialization explicitly requires at least 2496 for one item in the supplied
logs. Its auto backend and auto KV dtype constraints do not eliminate this
independent constraint. An app-side domain restriction can reject these values,
but does not align vLLM/Transformers versions or remove cold startup time.

TRT's native config is materialized by the optimizer; FLASHINFER must be checked
in the effective YAML and runtime dispatch. The supplied rerun reaches warmup,
hits readiness expiry, and then fails cleanup in psutil's `pidfd_open`. Those
logs do not demonstrate a new head-size MMHA failure. A successful reference
native config used FLASHINFER and chunked prefill false, unlike the app's fixed
chunked-prefill true; the exact successful configuration must be compared rather
than treating current notebook cell source as a record of every prior execution.

## Next implementation boundary

Reproduce one recorded successful configuration using its pinned snapshot and
runtime environment before expanding Optuna. Preserve the web workload identity
and requested concurrency; the reference's 1000 requests/concurrency 128 are
comparison evidence, not permission to change the web run's 100 requests/33.
Align runtime packages through an explicitly reviewed runtime update or a
supported, versioned patch delivery path. No such delivery path has been verified
here; changing host optimizer files alone does not change a running image.
No image rebuild, deployment, dependency installation, or paid job was performed
for this comparison.
