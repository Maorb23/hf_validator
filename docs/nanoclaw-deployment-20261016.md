# NanoClaw deployment, 2026-10-16

Discovery now keeps the existing HF/evidence ranking within the requested size band rather
than sorting by tiny parameter-count differences. Equal-score candidates get source diversity,
and failed initial candidates can be replaced within the shared research budget. This reuses
the existing scorer and runtime checks; it does not hardcode preferred model IDs.

Nemotron discovery validation uses a strict JSON schema, treats the original task as data,
and requires exactly one assessment per candidate. Failed or incomplete validation remains
explicitly unvalidated; malformed output never becomes evidence. A live replay returned valid
assessments, while keeping unsupported task-fit claims unknown.

NanoClaw now receives compact status responses rather than raw evidence dumps. The previous
large responses were truncated by OpenCode; the agent then extracted selected fields with
shell commands and missed the selection question. The compact response puts that question
first and provides a separate evidence tool for detailed active or historical evidence.

The existing optimizer controller database is explicitly allowlisted and mounted at
`/workspace/extra/hf-model-watch/optimizer-controller`. No cloud credentials were mounted.
The runtime includes the lightweight optimizer controller code and read-only connection probe.
Execution defaults are dry-run, with a 60-minute job timeout and 16 trials. No reconciliation
loop was started. Explicit approval, exact hardware/price, duration and maximum study cost are
required at the NanoClaw submission boundary.

Validation included 107 focused Python tests, an isolated deployed-container smoke test,
an offline replay of the failing saved Hub scan for both 2 × 16GB and 2 × 32GB, and a live
Nemotron schema replay. The offline replay returned Qwen3-8B, Qwen3.5-9B and Qwen2.5-7B-Instruct
using the previously saved evidence. It does not establish current live Hub results or speed.

An actual NanoClaw conversation on isolated run `run_conversation_test_20261016T070029`
returned a concise shortlist and the selection question, accepted “number 1”, persisted
Qwen/Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`, read the saved workload,
GPU setup and budget, checked the dry-run controller connection and offered optimization.
The user's active recommendation was unchanged. No new controller studies or GPU jobs were
created. The test then explicitly declined optimization. Local transcripts were archived under the
backup directory’s `verification/` subdirectory.

Live GPU execution remains untested and unapproved. Exact benchmark token sizes, concurrency,
GPU platform/preset and actual price still need confirmation. Existing worker-image tags must
be checked for the current optimizer contract before live use. Job-timeout cost checks use the
user-supplied GPU price and exclude storage and other non-GPU charges. Dry-run connectivity
must not be described as readiness for a live benchmark.

Before deployment, the previous runtime, agent directory, template, effective container
configuration, NanoClaw database, optimizer database snapshot, mount allowlist and image tag
were preserved in:

`/home/maorb/hf-model-watch-nanoclaw/backups/before-discovery-conversation-fix-20261016T065433Z/`

The earlier selection-handoff backup remains at:

`/home/maorb/hf-model-watch-nanoclaw/backups/before-selection-handoff-20261016T032951Z/`

The deployment updates mounted Python modules and agent instructions; the existing container
image was retained and the agent container restarted. No image rebuild was needed.
