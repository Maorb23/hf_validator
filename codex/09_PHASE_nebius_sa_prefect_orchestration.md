# Codex Phase 9 — Prefect controller and Nebius service-account orchestration

Implement only Phase 9.

Assume Phases 1–8 are complete and tests pass.

Goal: implement Prefect scheduled flow and Nebius job orchestration using service-account auth patterns inspired by the reference orchestrator script.

Create/modify:

```text
src/hf_model_watch/orchestration/prefect_flow.py
src/hf_model_watch/orchestration/nebius_jobs.py
scripts/run_prefect_controller_job.sh

tests/test_prefect_flow.py
tests/test_nebius_jobs.py
```

Prefect flow:

The flow should:

1. Run scanner.
2. Score/filter candidates.
3. Select top N eligible candidates.
4. For each candidate, create a Nebius serverless GPU job.
5. Pass model ID, revision, config, and artifact prefix.
6. Poll job status.
7. Ingest result manifests from object storage.
8. Update SQLite/Postgres store.
9. Generate leaderboard and report.

Do not require always-on GPU.

Support schedule once or twice daily through Prefect.

Nebius job module:

Implement `nebius_jobs.py`.

It should:

* build worker job command
* pass env vars
* support dry-run mode
* submit job through Nebius CLI or SDK if available
* poll status
* return structured result
* categorize failures as `nebius_job_failed` or `controller_auth_failed`

Controller shell script:

Implement `scripts/run_prefect_controller_job.sh`.

It must support Nebius service account auth using:

```bash
NB_PROFILE_NAME
NB_PROJECT_ID
NEBIUS_CONFIG_YAML_B64
NEBIUS_SERVICE_ACCOUNT_KEY_B64
SA_ID
PUBLIC_KEY_ID
NEBIUS_PRIVATE_KEY_B64
NEBIUS_PRIVATE_KEY_PATH
```

It should:

* use `set -euo pipefail`
* emit `TRACE_EVENT`
* install dependencies
* install/check Nebius CLI
* configure profile from service-account credentials
* run `hf-watch prefect-flow --config-dir config`
* support dry-run mode without credentials

Use reference orchestrator script patterns, but do not copy project-specific autoresearch logic.

Tests:

* dry-run Prefect flow selects mock candidates
* Nebius job payload/env construction works
* missing service account vars fails with typed category in non-dry-run
* dry-run mode does not require credentials
* controller script has expected safety flags and trace events

Success criteria:

```bash
uv run ruff check .
uv run pytest
uv run hf-watch prefect-flow --config-dir config --dry-run
DRY_RUN=1 bash scripts/run_prefect_controller_job.sh
```

Stop after Phase 9. Report files created/modified, commands run, test results, and any TODOs.
