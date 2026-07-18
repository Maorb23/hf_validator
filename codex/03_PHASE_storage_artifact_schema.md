# Codex Phase 3 — SQLite storage, artifact schema, local run manifest

Implement only Phase 3.

Assume Phases 1–2 are complete and tests pass.

Goal: implement structured local result storage and artifact manifests. Do not implement real model loading or fine-tuning yet.

Create/modify:

```text
src/hf_model_watch/storage/schemas.py
src/hf_model_watch/storage/sqlite_store.py
src/hf_model_watch/storage/postgres_store.py
src/hf_model_watch/storage/object_store.py
src/hf_model_watch/reporting/leaderboard.py

tests/test_storage_schemas.py
tests/test_sqlite_store.py
```

Storage design:

Use SQLite as default.

Postgres should be optional through SQLAlchemy URL but does not need deep tests in this phase.

Implement tables:

```text
models
model_revisions
scan_results
candidate_scores
runs
metadata_results
runtime_results
latency_results
quality_results
context_drift_results
finetune_runs
adaptation_results
comparisons
failures
artifacts
```

Minimum fields:

* run_id
* model_id
* revision
* model_family
* size_band
* started_at
* completed_at
* status
* failure_category
* failure_message
* selected_backend
* selected_dtype
* selected_quantization
* base_score
* adapted_score
* adaptation_lift
* latency_summary
* context_drift_summary
* artifact_uri

Implement `sqlite_store.py`:

* initialize DB
* insert scan result
* insert candidate score
* start run
* complete run
* record failure
* record artifact
* fetch leaderboard rows
* check whether model revision was already tested

Implement local artifact layout:

```text
artifacts/<run_id>/
  run_manifest.json
  trace_events.jsonl
  metadata/
    claims_vs_observed.json
```

`run_manifest.json` should include:

```json
{
  "run_id": "...",
  "model_id": "...",
  "revision": "...",
  "started_at": "...",
  "completed_at": null,
  "status": "started",
  "artifacts": {},
  "config_snapshot": {},
  "failure": null
}
```

Implement `object_store.py`:

* local copy/write helper
* S3-compatible upload helper skeleton using boto3
* dry-run mode
* no real upload required in tests

Wire CLI:

```bash
hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config
hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
```

For Phase 3, `run-local` should create a DB run and manifest without loading the model.

Leaderboard should produce Markdown even if scores are empty.

Tests:

* DB initializes
* insert/read run
* record failure
* already-tested revision works
* manifest writes valid JSON
* leaderboard generates a Markdown file

Success criteria:

```bash
uv run ruff check .
uv run pytest
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config
uv run hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
```

Must create:

```text
results/hf_model_watch.sqlite
artifacts/<run_id>/run_manifest.json
reports/leaderboard.md
```

Stop after Phase 3. Report files created/modified, commands run, test results, and any TODOs.
