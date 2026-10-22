# Infiro public content evidence

## Implemented capabilities described

- Model research and assessment: `recommender/core.py`, `recommender/discovery.py`, source-linked claim confidence, saved selected revisions.
- Remote validation: `orchestration/nebius_jobs.py`, `job_payloads.py`, `prefect_flow.py`; worker manifests and inference evaluation.
- Optimization preparation and analysis: `optimization/preparation.py`, `contracts.py`, `analysis.py`, `orchestration/optimization_adapter.py`.
- Actual optimization execution remains in the sibling LLM Optimizer package. No second optimizer is introduced in the frontend.
- Framework preparation profiles cover vLLM, SGLang, and TensorRT-LLM; exact model/image compatibility requires verification.
- Dynamo is included in the requested ecosystem bar. The technology page explicitly states that no Dynamo execution adapter is claimed.

## Performance publication audit

The optimizer experiment directory contained 1,830 JSONL trial records in the inspected snapshot. Every `model_revision` value was a placeholder: `REPLACE_WITH_PHASE_B_MODEL_COMMIT`, `REPLACE_WITH_PHASE_A_MODEL_COMMIT`, `the model’s Hugging Face commit`, or `PLACEHOLDER_MODEL_REVISION`. None met the immutable revision requirement.

Top-level optimizer result summaries and legacy validation reports were also inspected. They did not establish the complete immutable model/runtime/hardware/workload context required for these public comparisons. Quality, adaptation, and fine-tuning data were excluded by scope.

No public speedup, engine ranking, measured-study replay, or provider billing guarantee is asserted. Publication status is explicit rather than disguised with fabricated numbers.

## Simulations

The Python generator uses actual Optuna 4.9.0 samplers. Its traces are educational synthetic-objective experiments, not inference measurements. Public JSON preserves versions and checksums. Two generation passes in the same CPU environment matched exactly.

## Asset provenance

Official technology marks and the custom optimizer symbol are documented in `web/public/logos/SOURCES.md`. They identify technologies, not sponsorship. The original design prototype remains unchanged.


## Imported optimizer run snapshot

`web/public/data/publication.json` now contains an explicitly selected run-export
snapshot imported by `scripts/import_public_optimizer_runs.py`. This supersedes
earlier no-published-results statements for the public run archive. All 114 runs
and 1,948 trial records are retained, including 6 crashed runs. 1,524 completed
measured trial records have valid throughput and p95 E2E latency for plotting.
Missing metadata is disclosed rather than used to exclude a run. The 98
Qwen3.5-4B runs and 16 Qwen3-4B runs stay in separate model groups.

Model, recorded draft model, seed and dataset/workload are primary context. Known
hardware, revisions and request fingerprints further distinguish cohorts. No draft
model is reported in this snapshot; this does not confirm that none was used.
Latency and TTFT are p95, throughput is output tokens/second. Preserve the
`MEASURED_ALLOCATION_WALL_PROXY` source label. The data is recorded inference
evidence, not independently validated performance claims or a general engine ranking.
The strict JSONL exporter retains its separate contract; the run importer follows
the user-approved metadata retention policy.
