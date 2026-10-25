# Local platform verification

The private platform is implemented at `/app`, separate from the existing public
website. See [startup and operations](authenticated-platform.md) for installation,
exact server/worker commands, the offline demonstration and live requirements.

## Results

| Check | Result |
| --- | --- |
| Django platform and public-serving tests | **65 passed**, including private S3 payload transport, standalone worker bootstrap, integrity/size/ownership failures, upload-before-create ordering, duplicate protection, remote ownership checks, serialized metadata snapshots and narrowly redacted validation messages |
| Existing optimization preparation, dataset and optimization checks after transport change | **117 passed** |
| Existing recommender, discovery, preparation, dataset, optimization, conversation and NanoClaw focused tests | **197 passed** |
| Sibling optimizer HF workload, remote, spot and expanded-search tests | **60 passed, 1 pre-existing failure** |
| Existing frontend data checks | **Passed** |
| Public export tests | **3 passed** |
| Astro production build | **Passed, 13 pages** |
| Ruff on new Python code | **Passed** |
| Migration drift / whitespace checks | **No changes detected / passed** |
| Private Playwright browser checks | **17 desktop/mobile screens passed**, including service-account connection creation and deletion with a generated fixture key |
| Existing public Playwright checks | **11 pages × 2 viewports, interactions and downloads passed** |

The unchanged sibling failure is
`tests/test_remote.py::test_uncertain_cancellation_does_not_consume_concurrency_slot`.
It attempts to submit another study while an uncertain cancellation remains active;
the sibling controller's current global active-study guard rejects that submission.
No sibling files were modified. The web platform namespaces controller tables by
study within the single application database and tests its own cancellation and
recovery paths.

The focused legacy test command was:

```bash
PYTHONPATH=src:../bento_extend/llm-optimizer/src \
  /tmp/infiro-platform-env/bin/python -m pytest -q \
  tests/test_recommender.py tests/test_discovery.py \
  tests/test_optimization_preparation.py tests/test_optimization_datasets.py \
  tests/test_optimization.py tests/test_nanoclaw_mcp.py tests/test_conversation.py
```

## What was exercised

Django tests cover session authentication/registration and CSRF, encryption and
masked metadata, missing keys, connection replacement/deletion, ownership checks
for every resource, isolated child credentials/caches, stale approvals, duplicate
submissions, expired leases and bounded retries, time/disk limits, dataset access
failures/schema/size/row/token limits, reproducibility and integrity, immutable
preparation recovery, trial-count propagation, private downloads, and cancellation.

The real optimizer preparation/controller and result-analysis code are exercised
with mocked providers, including comparable measured-result ingestion, Pareto
selection, running-study cancellation, and restart/idempotency behavior. The real
Nebius SDK message classes are used to verify secret-reference job specs; their
provider I/O is mocked. Tests verify pinned images, secret project/value matching,
remote ownership, storage scope and duplicate reconciliation. IAM renewal tests
verify that only an existing study's transport credential can change while its
approved project/request remains immutable.

The browser script creates real Django accounts and sessions, saves/deletes a
masked local HF test credential, seeds the explicitly labeled offline fixture,
requires an unchecked-by-default approval checkbox, and waits for the actual
separate worker and controller dry run. It verifies duplicate approval,
cancellation, an authenticated results download, another user's access denial,
keyboard access, empty browser storage, absence of runtime/console errors, and
responsive layouts without horizontal overflow. Browser calls use only the local
backend; no inference, research or GPU provider call is made by the fixture.

Screenshots are in ignored `web/screenshots/`:

- `platform-overview-desktop.png`, `platform-overview-mobile.png`
- `platform-connections-desktop.png`, `platform-connections-mobile.png`
- `platform-recommendations-desktop.png`, `platform-recommendations-mobile.png`
- `platform-datasets-desktop.png`, `platform-datasets-mobile.png`
- `platform-configure-desktop.png`, `platform-configure-mobile.png`
- `platform-approval-desktop.png`, `platform-approval-mobile.png`
- `platform-study-desktop.png`, `platform-study-mobile.png`
- `platform-cancelled-study-desktop.png`

`web/screenshots/platform-browser-results.json` records the private browser checks.
The existing public browser report/screenshots are retained alongside them.

## Practical limits

No GPU jobs, image publications or external deployments were performed. There are
no claimed hardware measurements from these tests. Dry runs produce real controller
state and downloadable unmeasured results, with empty measurement charts.

Live operation still requires rebuilt immutable worker images, appropriate project
and registry permissions/quota, private bucket policies, pinned SecretStash versions
matching the user's HF/storage credentials, persistent external keys/data and
production process/network isolation. Live execution remains disabled locally.
The SDK integration is mocked-provider tested; a separately approved hardware
smoke test remains a deployment step.

Large/default ShareGPT repositories can exceed the web limits and are explicitly
rejected. Native random workloads and smaller compatible custom ShareGPT datasets
are supported. No upload, S3 dataset-import, column-mapping, fine-tuning, provider
spending-cap, organization-management or account-email workflow is implied.
