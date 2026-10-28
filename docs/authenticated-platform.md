# Private Infiro platform

The private product lives at `/app`. It shares Django and the Astro/React build
with the public website, but has its own app shell and routes. Existing public
pages, recorded demonstrations, CLI commands and NanoClaw integrations are preserved.

## Local startup

Run from the `hf_validator` repository root. Python 3.11+ and Node compatible with
the existing Astro lockfile are required. The sibling checkout must contain the
updated `llm_optimizer.hf_workload` and remote controller contracts.

```bash
python3 -m venv /tmp/infiro-platform-env
/tmp/infiro-platform-env/bin/pip install -r requirements_cpu.txt -r web/requirements-platform.txt
export PYTHONPATH="$PWD/src:$PWD/../bento_extend/llm-optimizer/src"

# Create once. Do not regenerate keys when restarting the server or worker.
/tmp/infiro-platform-env/bin/python web/scripts/platform_env.py \
  --output "$HOME/.config/infiro/local.env" \
  --data-dir "$HOME/.local/share/infiro"
export INFIRO_ENV_FILE="$HOME/.config/infiro/local.env"

npm --prefix web ci
npm --prefix web run build
/tmp/infiro-platform-env/bin/python web/manage.py migrate --noinput
/tmp/infiro-platform-env/bin/python web/manage.py runserver 127.0.0.1:8000 --noreload
```

In a second terminal, from the same repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD/../bento_extend/llm-optimizer/src"
export INFIRO_ENV_FILE="$HOME/.config/infiro/local.env"
/tmp/infiro-platform-env/bin/python web/manage.py platform_worker
```

Open `http://127.0.0.1:8000/app/register`. Registration creates a real Django user;
login establishes a database-backed session. Sign-out and every mutating API
require CSRF tokens. Use one origin consistently rather than alternating localhost
and 127.0.0.1. Restart Django after frontend builds so WhiteNoise indexes new assets.

There is one application database and one worker supervisor. The supervisor runs
one short-lived child operation at a time; no Redis, Celery, Node server or
NanoClaw process is needed. Multiple studies can be monitored by successive ticks.
`platform_worker --once` processes one available operation and exits.

For the already provisioned verification environment in this session, use:

```bash
export INFIRO_ENV_FILE=/tmp/infiro-platform-local.env
export PYTHONPATH="$PWD/src:$PWD/../bento_extend/llm-optimizer/src"
/tmp/infiro-platform-env/bin/python web/manage.py runserver 127.0.0.1:8000 --noreload
# In another terminal with the same two exports:
/tmp/infiro-platform-env/bin/python web/manage.py platform_worker
```

The verification database/artifacts are under `/tmp/infiro-platform-data`. These
are local test resources; use the persistent home-directory commands for your own
accounts. Never commit an environment file, database, private workload or screenshot
that contains real user data.

## Offline end-to-end demonstration

Register an account, then run the following with the same environment exports:

```bash
/tmp/infiro-platform-env/bin/python web/manage.py platform_demo --username YOUR_USERNAME
```

This command injects an explicitly labeled local model/dataset fixture, exercises
the shared CPU ShareGPT sampler and real optimizer preparation, and prints the
private review URL. It performs no provider calls and does **not** approve the
request. Open the URL, inspect the hash and allocation, tick the approval checkbox,
and press **Approve dry run**. The separate worker runs the real optimizer
controller in dry-run mode, saves an authenticated `results.json` download, and
reports that no GPU was allocated and no inference was measured. A dry run never
manufactures latency, throughput or Pareto points.

The fixture uses a placeholder model/commit and score zero, clearly marked as an
offline fixture throughout the interface. It is never a production recommender
fallback. Real recommendations require the user's inference connection and use
the existing discovery, scanner, evidence, deterministic scoring and Nemotron code.

## Connections and credentials

| Connection | Stored credential | Settings and validation |
| --- | --- | --- |
| Nebius inference | Token Factory inference API key | Public HTTPS inference base URL; read `/models` |
| Nebius jobs | Service-account RSA private key (PEM or base64 PEM), or manual Cloud IAM bearer token | Project ID; service-account mode also requires service account ID and public key ID; SDK lists one job to establish read access |
| Hugging Face | HF read token | Optional immutable SecretStash version for live workers; HF `whoami` |
| Object storage | Access key ID, secret access key, optional session token | Bucket, region, optional public HTTPS endpoint and live-worker SecretStash version; `HeadBucket` |
| Tavily | Research API key | Read usage endpoint |

Validation runs in a child process and returns a success/failure state.
Failure diagnostics expose exception types, basename/function/line locations, HTTP
status codes and structured provider error codes, never raw exception messages,
URLs, local variables or credentials. They appear in the private preparation screen
and the separate worker's terminal. Django's terminal continues to show HTTP logs.
Nebius token renewal errors are reported immediately instead of hidden behind SDK
background refresh retries. `JwtKeyNotExists` requires correcting the registered
public-key ID and matching private key in the Nebius jobs connection.
For new unlabelled submissions, safe provider diagnostics also retain the actual
serialized label count/keys/types, request and trace IDs, and narrowly redacted
`metadata.labels` field-validation messages. Other provider messages and request
bodies remain excluded. Older failure records cannot recover previously discarded
validation text; a new submission requires explicit approval as usual.

If an unlabelled request fails with `label value length (135) exceeds maximum
value (64)`, compare that length with `image_reference_length` in the safe
diagnostics. The configured vLLM digest reference is 135 characters. A matching
length suggests that Nebius Jobs is copying the image reference into an internal
Compute label; it does not establish which internal label failed. Ask Nebius
support to investigate using the request/trace IDs, timestamp, empty submitted
labels, and image reference length. A digest suffix alone (`@sha256:` plus 64 hex
characters) is 72 characters, so shortening the repository cannot meet a
64-character limit. Do not truncate digests or silently replace reviewed images
with unverified tags.

The adapter now works around this failure using short private registry aliases
from `config/platform-image-aliases.json` (override with `INFIRO_IMAGE_ALIASES`).
Reviewed payloads retain the full digest references from `platform-images.json`.
Immediately before creation, the adapter fetches the alias manifest using the
selected jobs connection's IAM credentials and hashes its bytes. Only a match
with the reviewed digest permits the short tag to be sent in `JobSpec.image`.
Missing aliases, denied registry access, and changed manifests stop submission
before payload upload or job creation. The jobs connection needs registry pull
access. Tags must be maintained by trusted operators: verification happens before
submission, while the provider pulls the tag later, leaving a tag-mutation race.
Restart the web server and worker after updating configuration. Prepare and review
a new study; previously approved studies are not automatically resubmitted.

The `workload1` images add the optimizer package with frozen `workload_manifest`
support under `/opt/infiro-worker`, placed first on `PYTHONPATH`, while retaining
the original serving-runtime layers. The previous `expanded-v1` workers reject
that request field before benchmarking; dropping it would lose the reviewed
workload and is not a valid compatibility fix. `docker/platform-worker.Dockerfile`
describes the package-only overlay for future builds (the build context must
contain the current optimizer's `llm_optimizer` package). Publish new tags and
digests rather than moving the old reviewed aliases. The bootstrap reports only
exception type/module and basename/function/line frames on failure, without raw
exception messages, credentials or prompt text. GPU execution must still be
validated through a newly reviewed study.

Shared W&B tracking reuses `llm_optimizer.wandb_tracking.WandbTracker`. Configure
`INFIRO_WANDB` as JSON containing `project`, optional `entity`, and an immutable
`secret_version_id`. That SecretStash version must contain `WANDB_API_KEY`; the
jobs connection needs payload-read access in the same project. No W&B API key is
stored in request JSON, approval data, or inline provider environment values.
The tracking configuration is frozen into each new prepared review. Worker
environment variables enable the existing tracker and attach study, user, and
framework tags. Use one private operator-managed W&B project initially; users
see their own study results through the app. Restart the server and worker after
changing settings, then prepare and approve a new study.

Configuration downloads (`request.json`, `intent.json`, `workload.json`) remain
available on Configure, Approval, and Study pages through the owner-scoped saved
preparation, even after its job leaves the recent-jobs list. Reopening Configure
uses the previous choices; preparing again creates a new review.

To print all frameworks' saved benchmark log tails from the latest launched web
study, run from the repository root:

```bash
INFIRO_ENV_FILE=/tmp/infiro-platform-local.env \
PYTHONPATH=src:../bento_extend/llm-optimizer/src \
/tmp/infiro-platform-env/bin/python web/manage.py study_logs --latest
```

Use `--study STUDY_UUID` to select a specific web study. This reuses the optimizer's
S3 log reader and the study's storage connection, printing the last 16 KiB per
framework with its actual Nebius job ID. Nebius container logs may contain only
startup messages because the optimizer stores subprocess output in `execution.log`.

Read validation does not prove object-write, job-create or GPU quota permissions. In
service-account mode the official SDK obtains and refreshes IAM tokens automatically,
using an in-memory private key; no CLI profile or server key-file path is read.
The form maps existing `NB_PROJECT_ID`, `SA_ID`, `PUBLIC_KEY_ID`, and
`NEBIUS_PRIVATE_KEY_B64` (or PEM contents) to encrypted per-user credentials.
Manual IAM tokens expire; plan their lifetime for the approved timeout. A job credential can be
renewed for an already allocated study if the project settings remain identical;
its transport credential reference is updated while the original approved request
is retained. All prior reviews are invalidated. Other connections in active use
are protected from replacement, and all active connections are protected from deletion.

Credential envelopes use the maintained `cryptography` library's authenticated
Fernet encryption, bound to user ID, connection UUID and provider kind. Configure
`INFIRO_CREDENTIAL_KEYS` externally. A comma-separated list supports MultiFernet
decryption during rotation; new writes use the first key. Retain old keys until
all affected credentials are replaced. Back up keys separately from database
backups. Missing keys reject credential writes. Only masked metadata is returned.
The local environment helper creates a persistent session key as well.

Queue entries contain connection IDs and versions, never credential values.
The supervisor constructs a fresh allowlisted child environment without changing
`os.environ`; each job has separate HOME, temporary directories, HF/dataset caches
and AWS config paths. Child processes receive neither the credential encryption
key nor Django's session signing key. Dotenv discovery, ambient AWS credentials,
CLI profiles, proxies and shared HF login caches are excluded. A Linux
parent-death signal kills an operation when its supervisor dies.

## Data, ownership and approval

All queries for connections, datasets, recommendations, reviews, jobs, results and
artifact downloads are scoped to the authenticated owner. Private responses are
`no-store`; no prompt or private artifact is added to Astro's public data files.
Artifact paths use database IDs, verify containment and SHA-256, and require the
owner's session. There is no arbitrary storage proxy or presigned URL endpoint.

Prepared payloads include a pinned model and tokenizer, frozen workload hash,
dataset revision or random-generator code hash, GPU allocation, image choices,
frameworks, timeout, storage prefix, connection names/versions, and estimated
limits. Explicit approval checks the full canonical digest again. Selection,
configuration or connection changes invalidate reviews. Duplicate approval
returns the existing study job; it does not create another allocation.

The existing llm-optimizer controller is reused through a small SQLite adapter
that namespaces its tables by study UUID in the same database. Its reconciliation,
submission-unknown handling, cancellation, manifests and results logic run within
that study's scope and credentials. No second durable controller database is
created. This adapter is covered by real controller tests and must be checked
against upstream schema changes when upgrading llm-optimizer.

Claims retain evidence URLs and documented/verified, inferred, unknown or
measured status. An external draft mention does not activate speculative execution;
preparation retains the existing exact revision/runtime/image evidence gate.
Results use `OptimizationAdapter.results` and the existing `analysis.analyze`
selection/Pareto implementation, including comparability and failed-request checks.

## Workloads and limits

Custom HF datasets accept repository ID/URL, revision, configuration, split,
prompt count, seed, token lengths and concurrency. Only ShareGPT conversations are
supported. The shared `llm_optimizer.hf_workload.prepare_hf_workload` validates
roles/nonempty text, samples without replacement and freezes one sample. The web
boundary preflights file sizes and bounds streaming iteration. Unknown download
sizes fail closed. Tokenizer loading disables remote code execution.

The first human prompt is tokenized, repeated or truncated to the requested input
length, with a fixed requested output length. This is a serving benchmark, not
conversation quality evaluation. Previews expose at most three samples of 400
characters, only to their owner. A custom workload never falls back silently.

Default random mode freezes the optimizer's existing native random-token sampler,
with seed, tokenizer commit and generator source hash. The default HF ShareGPT
source is also offered and goes through the same bounded validation; large
ShareGPT repositories are explicitly rejected rather than bypassing limits.

| Limit | Value |
| --- | --- |
| Selected ShareGPT file / tokenizer preflight | 1 GiB / 128 MiB; workload network reads bounded to 1 GiB |
| Scanned rows / source tokens | 10,000 / 2,000,000 |
| Row size / frozen artifact | 256 KiB / 16 MiB |
| Prompt count / requested tokens | 1,000 / 2,000,000 |
| Input / output / concurrency | 16,384 / 8,192 / 256 |
| ShareGPT child memory / disk / execution | 6 GiB / 1.25 GiB / 300 seconds |
| Study timeout / trials per framework | 60–240 minutes / 1–100 |
| Active operations per user | 10 |

Download/row/token failures are explicit. ShareGPT dataset and preparation jobs get a 1 GiB network-read budget and a 1.25 GiB disk budget. Other worker operations keep the 128 MiB network-read and 256 MiB disk budgets. Disk usage is checked by the supervisor;
Linux resource limits also bound file size, address space, CPU time and descriptors.
Public HTTPS endpoints must resolve entirely to global IPs; the child revalidates
and pins actual socket destinations, including redirects and DNS changes. Outbound
port 443 only. Nebius SDK endpoints are fixed by the official SDK, not user input.

Arbitrary column mapping, local uploads, S3 dataset imports and fine-tuning are
not implemented or advertised. Other production services such as account email
verification/reset, organization roles and billing are also outside this version.

## Worker recovery and cancellation

Claims use a database compare-and-set and an operation deadline. A worker
restart recovers expired leases. Failed operations retry at most three consecutive
attempts with backoff. Successful study polling resets that failure count.
Prepared requests and sampled datasets are reused on recovery rather than
regenerated. Controller submission uses the study's stable idempotency key and
reconciles uncertain provider acceptance by deterministic job name.

Cancellation is ownership-scoped and persisted. Work not submitted yet is cancelled
without allocating resources. Running studies use the existing controller/provider
cancellation and reconciliation. Poll failures for an allocated study become
`attention_required`; they never silently erase a potentially allocated resource.
Cancel can requeue reconciliation. Failed or cancelled non-study operations have
an explicit Retry action; a terminal study needs a new prepared request and approval.
Provider-confirmed cancellation may lag the
request. A new local review cannot alter an already running remote configuration.

## Live deployment and images

Live execution is disabled by default (`INFIRO_LIVE_ENABLED=0`) and was not used
during development. The included Python SDK adapter uses Nebius's official
[SecretStash environment-reference contract](https://github.com/nebius/api/blob/main/nebius/ai/v1/job.proto)
and [job service](https://github.com/nebius/api/blob/main/nebius/ai/v1/job_service.proto).
It does not call the Nebius CLI. The legacy command adapter is blocked in web jobs.
New SDK job requests omit custom labels, matching the existing CLI request shape.
Remote access checks the selected project and exact private study UUID/framework
name; retrieval also checks the requested provider job ID. If older ownership
labels are present, conflicting or incomplete labels are rejected as well.

Before enabling live mode:

Live worker payloads are uploaded as private, content-addressed JSON objects under
`<approved-prefix>/<run-id>/<framework>/launch-<sha256>.json`. The job environment
contains only a small S3 reference (bucket/key, size, hash and study identity), not
the prompt text. A trusted Python bootstrap is embedded in JobSpec arguments; it
downloads at most 16 MiB, verifies length/hash/identity, and calls the existing
`llm_optimizer.remote.worker.run_worker` entry point. Download peer/redirect
checks reuse the platform network policy. Credentials remain SecretStash refs;
there are no presigned URLs or credentials in the payload/reference/command.
Upload failures prevent job creation. Retries use the same content-addressed
object, and existing provider jobs are checked before another upload/create.

Preparation writes private draft `request.json`, `intent.json` and `workload.json`
artifacts before calling Nebius SecretStash. The saved draft has no approval digest.
If reference verification fails, the draft remains downloadable by its owner and
cannot be approved or submitted. Retrying reuses the saved request and workload.
The digest is issued only after references are pinned and verified.

The reference lookup uses the selected Nebius jobs connection, not the inference
API key or HF/storage credentials, to authenticate with IAM. It calls
`SecretVersionService.Get` and `SecretService.Get`; a secret ID needs an active
primary version. The caller needs permission for those metadata reads, and the
secret must belong to the selected project. Before submission the adapter also
calls `PayloadService.Get`, requiring payload-read access, and checks `HF_TOKEN`
and/or `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`
against the encrypted connection values. No secret-management write is performed.
The payload reader role is `mysterybox.payload-viewer`; the general `editor` role
does not grant payload access. Add the jobs service account to a group with that
role scoped to the relevant secrets/project. See the
[SecretStash payload-access documentation](https://docs.nebius.com/mysterybox/secrets/get).

The platform defaults are saved in `config/platform-images.json`. They pin the
existing CLI `expanded-v1` images to registry digests resolved on 2026-10-23.
`INFIRO_SERVING_IMAGES` overrides that file when set, using the same JSON mapping.
Restart Django and the worker after changing either setting, then prepare a new
configuration. Previously approved configurations retain their original images.
Registry digest resolution reads manifest metadata only; it does not download
image layers, run a container, or establish worker compatibility.

1. Each selected worker image needs Python, boto3, the matching llm-optimizer code,
   its `remote.worker.run_worker` entry point, `hf_workload.py`, and frozen manifest
   verification support. The storage bootstrap itself is supplied by the platform
   and does not require an image rebuild. Rebuild unsupported images with those dependencies.
   Publish images through the operator's separate release process and configure
   `INFIRO_SERVING_IMAGES` as a JSON mapping of frameworks to immutable
   `registry/image@sha256:DIGEST` references. Legacy mutable defaults are dry-run
   review values only; live preparation rejects them.
2. Provision private bucket policies scoped to the displayed per-user/per-review
   storage prefix. The SDK storage factory additionally checks the approved
   bucket, prefix, run, framework and artifact names.
3. In each user's Nebius project, provision immutable SecretStash secret versions
   for worker credentials. The storage version must contain named entries
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN`.
   An HF version must contain `HF_TOKEN` if a token is connected. Add secret or version IDs
   to their respective Connections entries alongside the encrypted credentials.
   Live preparation resolves secret IDs to their primary version, verifies project
   ownership, and binds the immutable version ID into the approval summary. Remote
   jobs use that pinned version even if the secret's primary version later changes.
   The adapter reads the pinned versions, verifies their project and compares
   their values to the reviewed user connections before submission. Job specs
   contain **only version references**, never inline secret values. The platform
   does not create or modify provider secrets.
4. Grant the user's IAM principal the necessary project job and secret-read/use
   permissions. Configure registry access and network/region/quota settings as
   required by the chosen images and project. Default configured GPUs/presets
   are the existing repository catalog, not a guarantee of available capacity.
5. Use persistent external keys/database/private disk, HTTPS, the existing
   production Django host/proxy settings and a supervised worker. Run the backend
   and worker under dedicated OS accounts with deployment-level outbound network
   rules and filesystem/container isolation. Python socket guards protect the
   supported SDK paths; they are not a sandbox for arbitrary native code.
6. Only after operator verification set `INFIRO_LIVE_ENABLED=1`, prepare a fresh
   review, and obtain explicit approval in the application. No live hardware
   behavior or provider permissions have been verified in this task.

`INFIRO_PROVIDER_FACTORY` defaults to
`infiro_platform.nebius_provider:factory`; an operator may supply another trusted
Python factory implementing the same controller interface. It is never accepted
from an HTTP request.

Hourly and total limits validate the **estimate** before approval. Prices are user
inputs; storage, CPU, network and provider price changes are excluded. The timeout
is sent to Nebius, but there is no enforced provider-dollar spending cap.

## Verification commands

```bash
export PYTHONPATH="$PWD/src:$PWD/../bento_extend/llm-optimizer/src"
/tmp/infiro-platform-env/bin/python web/manage.py test infiro_platform infiro_site --noinput
/tmp/infiro-platform-env/bin/python -m pytest -q \
  tests/test_recommender.py tests/test_discovery.py \
  tests/test_optimization_preparation.py tests/test_optimization_datasets.py \
  tests/test_optimization.py tests/test_nanoclaw_mcp.py tests/test_conversation.py
npm --prefix web test
python3 web/tests/test_export.py
/tmp/infiro-platform-env/bin/ruff check --no-cache web/infiro_platform web/scripts/platform_env.py

# Server and worker must both be running. Installs Chromium locally if needed.
cd web
PLAYWRIGHT_BROWSERS_PATH=/tmp/infiro-playwright npx playwright install chromium
cd ..
INFIRO_URL=http://127.0.0.1:8000 \
  INFIRO_PYTHON=/tmp/infiro-platform-env/bin/python \
  PLAYWRIGHT_BROWSERS_PATH=/tmp/infiro-playwright \
  node web/tests/platform-browser.mjs

# Existing public website browser regression suite:
cd web
INFIRO_URL=http://127.0.0.1:8000 \
  PLAYWRIGHT_BROWSERS_PATH=/tmp/infiro-playwright npm run check:browser
```

The private browser script registers local test users, saves/deletes a deliberately
invalid masked HF test credential without validation, seeds the offline fixture,
requires explicit UI approval, waits for the actual worker/controller dry run,
downloads its real unmeasured results, and checks a second user's access denial.
It checks desktop 1440×1000 and mobile 390×844 layouts, runtime/console errors,
horizontal overflow, keyboard access and empty browser storage. It verifies
duplicate approval and cancellation too. It writes 15 screenshots and a
verification JSON under ignored `web/screenshots/platform-*`.

## Recommendation display names and ShareGPT configuration

Recommendation names are optional, trimmed display metadata (160 characters;
control characters are rejected). Blank and legacy names resolve without writes
to a stable request excerpt, creation time in UTC and eight-character ID. Owners
can rename through `PATCH /app/api/recommendations/<id>` with only `{"name": ...}`.
Names are returned separately from selected model IDs and study names. Renaming
never changes selection versions, prepared requests, approval digests, study names
or execution IDs. Apply migration `0003_recommendation_name` when deploying this
web change; no execution artifacts need rewriting.

Configure offers Random, default HF ShareGPT, and previously frozen custom
ShareGPT workloads. Its inline dataset editor reuses the Datasets API and isolated
validation worker. Enter a repository, revision, optional configuration and split;
the tokenizer defaults to the selected pinned model. Validate and freeze, then
select the ready workload on Configure. Only ready datasets owned by the user can
be prepared. A frozen sample must match the selected tokenizer and requested
length/count/seed/concurrency settings; validate a new sample to change them.

The default is [`anon8231489123/ShareGPT_Vicuna_unfiltered`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered), revision `main`, split
`train`, using the pinned `ShareGPT_V3_unfiltered_cleaned_split.json` file. The
repository has multiple unrelated files and is about 4.35 GB total; this selected
conversation file is about 673 MB at the current main revision. The Datasets form
can select one relative JSON file from any repository. A selected file is pinned
by resolved commit and included in the frozen workload identity. A selected JSON
file may be up to 1 GiB; the tokenizer remains capped at 128 MiB. Without a file
selection, every non-document repository file is counted toward the 1 GiB cap.
There is no silent fallback to random or another source. Errors include advice on
schema, access, revision/split, missing file and size limits. Deterministic validation rejections stop immediately rather than entering the transient worker retry loop; an oversized preparation links back to Configure with guidance to choose Random or a smaller source.
Validation reports access/size preflight, tokenizer loading and conversation
sampling stages. Private previews contain at most three prompts of 400 characters.
Dataset commit, configuration, split, tokenizer revision and workload SHA-256
remain visible in Approval and study details.

The shared sampler accepts `conversations` arrays of `from`/`value` objects with
`human`, `gpt` or `system` roles, nonempty text and at least one human turn. It
samples only the first human turn, repeating/truncating its token IDs to the input
length. Output length is a fixed serving-workload target, not an answer-quality
score. Preparation stores one manifest for every selected framework and trial;
workers must consume that manifest rather than resample the source.

Configure downloads are explicitly **Previous configuration files**. They belong
to the earlier preparation and do not represent unsaved form edits. The new
Approval page exposes its own files after preparation finishes.

Connection cards now contain expandable setup instructions and static copyable
command templates. They explain the form fields separately from process
environment settings. Validation scopes follow the implementation: inference
`/models`, HF `whoami`, S3 `HeadBucket`, Tavily `/usage`, and Nebius project job
listing. These establish the corresponding authentication/read connectivity,
not successful model generation, object writes or GPU-job creation. W&B's help
covers project/entity, immutable SecretStash versions, payload-read permission,
`INFIRO_WANDB`, restarting both Django and the preparation worker, and the read-only
`wandb_check --prepared` command. Secret access checks do not confirm W&B run
connectivity. Help and clipboard templates never interpolate connection values.

### Image compatibility for this change

The checked-in `config/platform-images.json` digests map to `workload1` aliases
for vLLM, SGLang and TensorRT-LLM in `config/platform-image-aliases.json`. Those
images are documented above as including `/opt/infiro-worker` and the optimizer's
frozen `workload_manifest` path. These UI, metadata and validation-progress changes
require **no additional image rebuild for that configured set**. An installation
still using `expanded-v1` needs the existing package overlay described by
`docker/platform-worker.Dockerfile`; retain the reviewed manifest contract.
This assessment uses repository configuration and the existing deployment record;
running remote image contents were not probed. No images were built or published,
and no GPU studies were submitted to verify runtime behavior.

### Incremental ShareGPT JSON preparation

Selected JSON data files are read at the pinned dataset commit in 64 KiB chunks.
The web worker frames and decodes one record at a time, supporting top-level
JSON arrays (including pretty-printed arrays) and JSONL. It does not pass arrays
to Hugging Face's whole-document JSON loader. The 6 GiB memory ceiling and all
existing row, token, schema and frozen-workload limits still apply. Sampling
and workload identity continue to use the shared first-human-turn sampler.

Select an explicit JSON data file. For wrapped objects, compressed files or other
formats, convert offline to an array of ShareGPT records or one record per line.
Shard into files with at most 10,000 rows and rows below 256 KiB, upload them to
Hugging Face, then select one shard and validate again. A large file can pass the
1 GiB preflight and still be rejected for row or source-token limits. Memory
exhaustion is terminal and requires a smaller source or workload before retrying.

The built-in unfiltered ShareGPT source uses a deterministic prefix of at most
10,000 source rows. Every conversation must still pass the shared schema checks;
invalid conversations (including empty later turns or missing human turns) are
excluded as whole records. Invalid rows count toward the scan cap. The frozen
manifest records the prefix policy, scan count, exclusion count, and original
source row indices, all covered by its hash. This policy applies only to the
built-in repository, selected file, default configuration and train split.
Custom sources retain strict schema rejection. Existing frozen workloads are
reused unchanged; prepare a new configuration to use the corrected default sample.

### Persistent default ShareGPT source cache

The worker retains the first 10,000 source records for the public default
ShareGPT file under `INFIRO_PRIVATE_ROOT/users/<owner-id>/sharegpt-cache`.
Cache keys include the repository, resolved commit, selected file, row cap,
row-byte cap and cache format version. Changing tokenizer, seed or request
lengths reuses the source records while generating a new frozen workload.
The cache stores invalid records too, preserving original row indices and
schema exclusion counts. Private and custom sources keep job-local loading.
No tokenizer or credential cache is shared across jobs.

On first use, the source is read with bounded 1 MiB network prefetch and parsed
incrementally into a bounded JSONL prefix. Later preparations verify its checksum
and read it locally. A partial download never becomes a cache hit; a lock and
atomic completion marker protect publication. Progress distinguishes initial
source caching, local reuse and tokenization, with counts every 1,000 rows.

The cache is limited to 1 GiB, and its bytes are included in the existing
1.25 GiB combined source-cache/job disk budget. Memory, network, row and token
limits remain unchanged. No automatic deletion occurs: if old revisions fill
the cache or an integrity check fails, preparation stops with guidance for the
operator to remove unused revisions (or the account's cache) before retrying.
Restart the preparation worker to enable caching for newly queued preparations;
existing frozen artifacts remain unchanged. The first preparation still pays
the download cost; tokenizer loading, tokenization and runtime research still
run as required.

The default ShareGPT prefix policy is now `default_sharegpt_valid_prefix_v2`.
Records exceeding 256 KiB are excluded as whole conversations, without raising
the limit. The incremental parser discards buffered bytes once the cap is reached
and drains that record using bounded nesting/string state. Unsafe or incomplete
framing still fails. Records whose escaped JSON representation exceeds the same
cap are also excluded. Custom datasets retain strict row-size rejection.

Oversized records count toward the 10,000 source-row cap and remain represented
by exclusion placeholders in source-cache format version 2. The cache stores
bounded envelopes around ordinary rows or placeholders; it never stores oversized
record contents. Frozen manifests include `oversized_rows_skipped`, original
source indices and the updated policy in their hash. Old cache entries and frozen
workloads remain unchanged; new preparations use the new cache key and policy.

### Selected-file preparation contracts and validation failures

The optimizer's `HFDataset` semantic contract accepts an optional relative JSON
`data_file`, preserving the pinned selected file through input normalization,
intent generation and `request.json`. Preparation compares the complete dataset
identity against the frozen manifest, including this field. A selected file
requires a frozen manifest at the optimizer boundary; it cannot silently fall
back to repository-level resampling. Older manifests without a selected file
retain their existing identity comparison.

Pydantic configuration validation failures are terminal on the first attempt,
including failures recognized from older saved diagnostics. Provider/transient
errors retain their retry policy. Diagnostics expose only allowlisted field
paths and validation codes, omitting input values, arbitrary field names,
validator messages and contexts. An already submitted study retains its
controller reference and becomes `attention_required` on validation failure.
Restart the worker to use these changes. Failed preparations require a new
preparation from Review configuration; existing frozen requests remain unchanged.

## Delete saved history

Recommendations now have **Delete recommendation**; completed, failed, cancelled and
past dry-run studies have **Delete study**. A trial inspector in a finished study
also has **Delete trial**. Each action first offers **Confirm deletion** and
**Keep history**. Deleting a recommendation removes its dependent study history.
Active or worker-leased operations must finish or be cancelled before removal.

Deletion is persisted in SQLite, scoped to the authenticated owner, and protected
by CSRF. Deleted resources disappear from bootstrap and cannot be opened or retried
through their old API URLs. Original execution artifacts and provider logs remain
on disk; this is workspace-history deletion, not a storage purge. Individual trial
removal leaves original results intact, filters the visible measurements and
reuses the optimizer analysis to recompute selections and Pareto fronts. Incomplete
legacy context leaves a selection unavailable rather than retaining a deleted one.
Original resource usage totals remain unchanged.

Background operations display only active work. Finished, failed and cancelled
research operations no longer produce stale Retry panels.

Migration `0004_job_deleted_at_job_deleted_trials_and_more` adds the persistent
history fields. After updating another installation, run `web/manage.py migrate`,
rebuild `web/` and restart Django and the worker. No GPU jobs are needed to apply
or test this feature.
