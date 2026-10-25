# Preparation, trial outcomes, and W&B

Preparation uses one progress message and hides draft downloads. A ready message
appears before the approval form only after preparation and secret reference checks
finish. Approval remains explicit. Failed/cancelled preparation keeps recovery
controls and diagnostics. Saved configuration files remain available after readiness.

Study lifecycle state is separate from benchmark outcome. A finished controller can
have failed trials. The API exposes `outcome` as `partial_success` when valid
candidates coexist with failures, or `no_valid_results` when all recorded trials
failed. Studies and Overview use these labels without changing controller IDs,
approvals, provider jobs, or results. Recovered provider timeouts are history,
not current error banners. A successful poll clears an error only for the same
provider operation and remote job.

The study page lists successful, failed, and pruned trials per framework and loads
saved execution log tails on demand. `/app/api/jobs/<UUID>/diagnostics` checks
ownership before accessing storage, reads only the approved study's framework log
paths, returns the last 16 KiB per framework, and redacts known credentials,
bearer tokens, and URL queries. Logs render as plain text, and responses are private
and uncached. Reading logs never starts jobs or changes artifacts.

## vLLM investigation

The two Qwen3.5-4B runs reported 99 successful requests out of 100. The validity
check correctly rejected both trials. The `EngineDeadError` happened during
optimizer-requested teardown after the benchmark and is not evidence of the
missing request's cause. The uploaded summaries and metric rows do not retain that
request's exception. Current source considers streams without a first decoded token
unsuccessful, and also catches transport and response parsing errors. Which case
occurred in these historical runs remains unknown; do not infer it from HTTP 200.

The sibling optimizer now preserves up to 20 request errors (index, category,
2,000-character error), without prompts or generated text. Telemetry saves these
before objective validation can raise. Optuna summaries retain failed-trial reasons,
request accounting, and request errors. Infiro carries these as `failed_trials`
metadata and displays them independently of valid candidates. A CPU regression
reproduces 99/100 and verifies that it remains invalid and has no accepted objective.

These remote-worker source changes require rebuilding the affected worker images
and using new reviewed digests to collect new detailed request evidence. No image
was built or published here. Existing saved logs are available immediately. The
scoped optimizer changes are also recorded in
[optimizer-failure-diagnostics.patch](patches/optimizer-failure-diagnostics.patch).
Apply that patch only to another optimizer checkout that does not already contain
the changes.

## Trial counts

The inspected historical study stored `trials: 1` in Configure choices, approval,
and execution request. Its uploaded optimizer metadata confirms budget 1 and
`stop_reason: trial_budget`. There is no evidence that a two-trial request was
shortened by the optimizer. The exact historical UI interaction cannot be recovered.

Configure now controls the trial input, preserves it across polling, and remounts
when switching recommendations. Submission checks the acknowledged count. The
preparation screen shows the saved count, and approval continues to show it.
Preparation rejects any effective framework trial budget that differs from the
saved choices. API-to-worker-command and browser tests cover a count of two.
Existing approved requests are never rewritten.

## W&B operations

Use the same environment for Django and the platform worker:

```bash
export INFIRO_ENV_FILE=/absolute/path/to/existing/credentials.env
export INFIRO_WANDB='{"project":"YOUR_PROJECT","entity":"YOUR_ENTITY","secret_version_id":"YOUR_REFERENCE"}'
export PYTHONPATH="$PWD/src:$PWD/../bento_extend/llm-optimizer/src"
python web/manage.py wandb_check --prepared PREPARED_UUID
```

This read-only command resolves a parent secret or version to an immutable version,
checks project ownership and payload-read access, verifies `WANDB_API_KEY` is
present, and prints pinned configuration without credential values. It does not
create W&B runs or GPU jobs. Use its printed configuration in both processes.
W&B API authentication and remote network access are confirmed when a real run
connects, not by the SecretStash check alone.

Restart Django after rebuilding the frontend. Restart the platform worker with the
same exports. Worker startup prints the W&B project it loaded. New preparation pins
W&B references before hashing approval. If Django has tracking configured but a
preparation worker saved no tracking, approval displays a warning to restart that
worker and prepare again.

```bash
python web/manage.py platform_worker
```

For preparation without processing or recovering approved GPU studies:

```bash
python web/manage.py platform_worker --prepare-only
```

Both modes use the same single-worker lock; stop one before starting the other.
Changes require no database migration. Run `npm run build` in `web/` before
restarting Django; `npm run test:platform-state` exercises the UI with mocked APIs
and does not require a database or provider credentials.

## Files changed for this work

- `web/src/components/Platform.jsx`: preparation, trial count, outcome labels, logs, W&B warning.
- `web/infiro_platform/study_status.py`, `study_logs.py`: outcome metadata and saved log reads.
- `web/infiro_platform/views.py`, `urls.py`: owner-scoped diagnostic API and outcome serialization.
- `web/infiro_platform/services.py`, `nebius_provider.py`: trial-budget guard, error recovery, W&B pinning.
- `web/infiro_platform/queue.py`, `management/commands/platform_worker.py`: preparation-only mode and W&B startup status.
- `web/infiro_platform/management/commands/wandb_check.py`: read-only reference and payload verification.
- `web/infiro_platform/tests.py`, `test_nebius_provider.py`: regression checks.
- `web/tests/platform-state-browser.mjs`, `web/package.json`: isolated browser checks and command.
- `src/hf_model_watch/optimization/contracts.py`, `src/hf_model_watch/orchestration/optimization_adapter.py`, `tests/test_optimization.py`: failed-trial evidence separate from accepted measurements.
- `docs/platform-study-diagnostics.md`, `docs/patches/optimizer-failure-diagnostics.patch`: operations and runtime patch.
- Sibling `llm-optimizer`: `src/llm_optimizer/bench_client.py`, `cli.py`, `optimization.py`, `tests/test_failure_diagnostics.py`.

Local operation also saved only non-secret W&B configuration references to ignored
`web/private/worker-wandb.env`; existing credential files and values were not changed.
A fresh configuration was prepared with two trials per framework and W&B enabled.
It remains unapproved. No GPU job, image build/publish, or database reset was performed.
