# Codex Phase 8 — Nebius worker job and S3-compatible artifact upload

Implement only Phase 8.

Assume Phases 1–7 are complete and tests pass.

Goal: implement the real worker shell script for serverless GPU execution and artifact upload. Use the reference scripts as architectural guides.

Create/modify:

```text
scripts/run_model_worker_job.sh
src/hf_model_watch/orchestration/job_payloads.py
src/hf_model_watch/storage/object_store.py

tests/test_job_payloads.py
tests/test_object_store.py
```

Worker script requirements:

Implement `scripts/run_model_worker_job.sh`.

It must:

* use `set -euo pipefail`
* emit `TRACE_EVENT` records
* install minimal system dependencies
* install `uv`
* install project dependencies
* configure cache dirs
* read env vars
* decode config/payload from env where provided
* run `hf-watch worker --payload-json ...`
* upload artifacts to object storage when configured
* exit nonzero only on infrastructure-level fatal errors
* model-level failures should be recorded as typed results and uploaded

Required env vars:

```bash
RUN_ID
MODEL_ID
MODEL_REVISION
HF_TOKEN
CONFIG_B64
PAYLOAD_JSON_B64
AWS_S3_BUCKET
AWS_S3_PREFIX
AWS_ENDPOINT_URL_S3
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
WANDB_API_KEY optional
```

Artifacts go to:

```text
s3://<bucket>/<prefix>/<run_id>/
```

Local artifacts go to:

```text
/workspace/output/<run_id>/
```

Implement object store upload:

* local dry-run mode
* S3-compatible endpoint support
* recursive artifact upload
* structured `TRACE_EVENT` for upload started/completed/failed
* record uploaded artifact URI in DB/manifest

Implement `job_payloads.py`:

Payload fields:

```json
{
  "run_id": "...",
  "model_id": "...",
  "revision": "...",
  "config": {},
  "artifact_prefix": "...",
  "skip_finetune": false,
  "finetune_smoke_only": false
}
```

Worker CLI:

```bash
hf-watch worker --payload-json payload.json
```

The worker should run the same pipeline as local run, but use `/workspace/output`.

Tests:

* payload validates
* missing required field fails cleanly
* object store dry-run works
* artifact URI generation works
* upload failure maps to `artifact_upload_failed`

Success criteria:

```bash
uv run ruff check .
uv run pytest
PAYLOAD_JSON_B64=<base64-test-payload> bash scripts/run_model_worker_job.sh
```

The shell command may run in dry-run mode locally. It must create local artifacts and not require real AWS credentials for dry-run.

Stop after Phase 8. Report files created/modified, commands run, test results, and any TODOs.
