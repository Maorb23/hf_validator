# Railway website and authenticated app

Railway runs one CPU service: Gunicorn serves the Astro artifact and Django `/app/`
APIs, and `platform_worker` handles the durable queue and preparation. Nebius runs
GPU workloads. The root `Dockerfile` remains the separate model-worker image.
The existing Railway PostgreSQL service stores accounts, encrypted connections,
the durable queue, and optimizer controller tables. One application service still
uses `/data` for private artifacts, frozen workloads, caches and worker locks. Do
not add app replicas, regions, or a separate worker service with this file layout.
Local development without `DATABASE_URL` continues to use SQLite.

## Files and service configuration

- `docker/web.Dockerfile`: Node builds `web/dist`, then a Python CPU image installs
  `web/requirements-railway.txt`. It never installs the root GPU project.
- `vendor/llm-optimizer/`: licensed source snapshot pinned to full commit
  `915beb35e4b2b398f1009c9e22d0fd3da3243d29`, checked by SHA-256 before pip
  installation. No sibling checkout or private Git token is required.
- `deploy/web_start.py`: requires the mounted `/data` volume, takes an exclusive
  service lease, validates encryption configuration, runs migrations, then starts
  Gunicorn and the worker. A failed migration starts neither service. Either
  service exiting stops its peer and exits nonzero so Railway restarts the unit.
  SIGTERM/SIGINT stop both, with a bounded 20-second shutdown and forced cleanup.
  The worker also kills/reaps isolated preparation children; interrupted jobs
  recover from durable leases on the next start.
- `.railway/railway.ts`: the `web` partial manages one application service and its
  artifact volume, preserving your existing PostgreSQL service. One replica in
  `europe-west4-drams3a`, one 10 GiB
  volume mounted at `/data`, `/healthz`, no overlap, 30-second drain and failure
  restarts. Choose region and initial size before provisioning. Artifact/cache
  growth requires monitoring volume space; monitor PostgreSQL separately. Disable
  Railway Serverless sleeping.

Use Railway's current [Infrastructure as Code workflow](https://docs.railway.com/infrastructure-as-code)
with CLI 5.42.1 or newer and Node 24 or newer. Install the pinned authoring SDK with
`npm ci --prefix .railway`; run `npm run check --prefix .railway` locally.
Create/link the intended Railway project and environment, select this repository
and its reviewed branch as the `infiro-web` service source, and keep the build root
at `/`. The config deliberately does not guess a repository URL, account, or domain.
Review `railway config plan` before an operator applies the configuration with
`railway config apply`. These commands can provision resources; they are not part
of local validation. Preserve existing resources when importing an existing service
with `railway config pull`; do not create a replacement volume over existing data.

If configuring through the dashboard instead, use exactly these settings:
Dockerfile `docker/web.Dockerfile`, start `python deploy/web_start.py`, one replica
in one region, persistent volume `/data`, health path `/healthz`, health timeout
300 seconds, restart on failure (10 retries), overlap 0, drain 30 seconds. Leave
pre-deploy commands empty. Volumes are mounted at runtime, so migrations belong
in startup, never in an image build or Railway pre-deploy command.

Generate a Railway HTTPS domain or attach the approved custom domain to this
service. Railway supplies `PORT`; Gunicorn listens on `0.0.0.0:$PORT`. Both `/` and
`/app/` use that same HTTPS origin. No CORS or second frontend service is needed.
Optional public build argument `SITE_URL=https://<deployment hostname>` sets
canonical URLs and sitemap. Do not pass any secret as a build argument.

Railway's [health checks](https://docs.railway.com/deployments/healthchecks) use host
`healthcheck.railway.app` and require a 2xx response. Include it in allowed hosts.
Only `/healthz` is exempt from HTTPS redirection for the internal probe; user
routes redirect to HTTPS and use secure session/CSRF cookies. Trust the forwarded
SSL header only behind Railway's trusted edge. `/healthz` reports frontend presence;
startup migration gating and process supervision cover database startup/worker
failure. Railway's probe is deployment-time only, not continuous monitoring.
Volume-backed redeploys have downtime; plan a maintenance window.

## Railway Variables and secrets

Set these on the runtime service in Railway Variables, not in a committed file:

```text
INFIRO_ENV=production
DJANGO_SECRET_KEY=<persistent secret of at least 50 characters>
DJANGO_ALLOWED_HOSTS=<deployment hostname>,healthcheck.railway.app
DJANGO_TRUST_PROXY_SSL_HEADER=1
INFIRO_CREDENTIAL_KEYS=<persistent Fernet encryption key>
INFIRO_PRIVATE_ROOT=/data/private
DATABASE_URL=${{Postgres.DATABASE_URL}}
INFIRO_LIVE_ENABLED=0
```

In the **web service → Variables**, add a reference variable `DATABASE_URL` pointing
to your existing PostgreSQL service's `DATABASE_URL`. `${{Postgres.DATABASE_URL}}`
is the Railway reference syntax when the service is named `Postgres`; use its
actual name or Railway's reference picker. Use the private connection URL, not
`DATABASE_PUBLIC_URL`. Keep it as a runtime secret. Remove the old
`INFIRO_DATABASE` variable from the Railway web service; it is only a local SQLite
fallback and is ignored when `DATABASE_URL` is present. Production container
startup requires a PostgreSQL URL and refuses to silently create SQLite instead.
The IaC partial preserves the existing reference and does not provision or replace
Postgres. See [Railway PostgreSQL connections](https://docs.railway.com/databases/postgresql).

Hostnames have no scheme, port, or path; add any custom hostname explicitly.
Use a password manager to generate/store the Django secret (for example,
`secrets.token_urlsafe(64)`) and a Fernet key (`Fernet.generate_key()`). Generate
keys privately, outside shared logs and shell history. Keep both stable across
restarts. For a migration, use the **original** encryption key(s), never generate
replacement keys for existing ciphertext. Escrow keys separately from backups.

Enter provider secrets only through the authenticated Connections UI; they are
encrypted in PostgreSQL and passed to isolated child environments only as needed.
Do not set them as Astro variables, `PUBLIC_*`, Docker ARG/ENV, source files,
committed dotenv files, command-line arguments, or public JSON. The dedicated
Docker ignore file allows only build inputs and excludes private directories,
SQLite, dotenv and key files. Docker COPY statements name source directories
explicitly. Never place credentials under `web/src`, `web/public`, or `config`.
Gunicorn access logging is off; do not enable request/body or environment dumps.
Keep existing redacted diagnostics; never print decrypted connection values.

## Accounts and provider connections

Open `https://<hostname>/app/register` to create a username/password account
(minimum 12 characters), then log in at `/app/login`. Registration is currently
public, with no email invitation/verification; each user owns their connections,
datasets and studies. Provider accounts are separate from the application account.
In `/app/connections`, save and validate each needed connection:

| Connection | Account and values | Verification |
| --- | --- | --- |
| Nebius inference | Create a Nebius AI Studio account/API key; enter API key and its HTTPS OpenAI-compatible endpoint | Inference validation/model access; independent of Jobs IAM |
| Nebius jobs | Create a Nebius project and service account; enter project ID, service account ID, public key ID and private key; alternatively a short-lived IAM token | Project/job permissions; token expiry requires renewal |
| Hugging Face | Create an HF account, accept required gated model/dataset terms, create a read token | HF identity and access to pinned model/dataset revisions |
| Object storage | Provision a private S3-compatible bucket; enter permanent access key ID, secret key, bucket, region and public HTTPS endpoint if needed | HeadBucket plus intended prefix read/write permissions |
| Tavily | Create a Tavily account/API key | Search validation and sufficient quota |

Provider validation makes outbound requests and may consume provider quota, but
is not permission to launch GPU jobs. Missing, unverified or denied connections
must remain visible as errors. HF access is not evidence of runtime support.
For live jobs, HF/storage credentials additionally need immutable Nebius
SecretStash version references; UI credentials alone do not prove worker access.
See [provider requirements and live prerequisites](../authenticated-platform.md).

## Move an existing SQLite installation to PostgreSQL

For a new installation with no accounts/data to retain, set `DATABASE_URL` and
start the service; migrations create the PostgreSQL schema automatically. For
existing data, use a maintenance window and a fresh, dedicated target database:

1. Block new requests and stop **all** web writers and local platform/controller
   workers. Keep `INFIRO_LIVE_ENABLED=0` on the target. Reconcile any running
   Nebius workloads first: stopping an app does not cancel remote allocations.
2. Save the exact source/image/optimizer revisions and original Django secret and
   `INFIRO_CREDENTIAL_KEYS` in secure escrow. Do not generate new encryption keys
   for existing ciphertext. Find the actual source SQLite path and private root
   (defaults: `web/private/platform.sqlite3` and `web/private`).
3. Back up the stopped SQLite database with its backup API, check integrity, and
   copy the complete private root, including user/job directories, frozen workloads,
   artifacts and caches. Keep an encrypted, checksummed original backup:

   ```python
   import sqlite3
   with sqlite3.connect('/source/private/platform.sqlite3') as source:
       with sqlite3.connect('/secure-backup/platform.sqlite3') as target:
           source.backup(target)
           assert target.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
   ```

4. Export Django records from that backup using the updated CPU environment:

   ```bash
   env -u DATABASE_URL INFIRO_DATABASE=/secure-backup/platform.sqlite3 \
     python web/manage.py dumpdata --natural-foreign \
     --exclude contenttypes --exclude auth.permission \
     --output /secure-backup/platform.json
   ```

   Preserve primary keys: **do not use `--natural-primary`**. Credential envelopes
   bind user and connection IDs, and artifact paths bind user/job IDs. Natural
   foreign keys resolve permissions against target migrations. The fixture and
   backup remain private; never put either in Git or an image. `dumpdata` excludes
   the optimizer's unmanaged controller tables; import those in step 6.
5. Provision `/data`, and with the destination app/worker stopped, securely transfer
   the complete private tree to `/data/private`. Preserve names/paths, verify
   checksums and set private permissions (`0700` directories, `0600` files). Keep
   the SQLite backup in a secure maintenance location for the controller import;
   it is no longer the running application DB. Do not mount private files under
   the public Astro `web/dist/data` directory. `railway run` executes locally and
   does not mount the remote volume; use an authenticated maintenance session or
   container attached to the target volume. Never expose a public upload route.
6. In the maintenance environment, configure `DATABASE_URL` for the dedicated
   target PostgreSQL database, the original credential keys, and private root.
   Keep all app/worker processes stopped while running:

   ```bash
   python web/manage.py migrate --noinput
   python web/manage.py loaddata /secure-backup/platform.json
   python web/manage.py import_sqlite_controllers --source /secure-backup/platform.sqlite3
   ```

   `loaddata` preserves IDs and resets managed sequences. The controller importer
   copies only UUID-scoped optimizer tables for existing study jobs, preserves run
   IDs/requests and refuses to overwrite nonempty target controllers. Each study
   import is transactional; if an import fails after earlier studies completed,
   restore the target's pre-import backup or use a fresh empty target before
   retrying the whole procedure. Never restart the importer against partially
   populated controllers or an unrelated application's database.
7. Verify all saved credentials decrypt without printing them, model counts and
   identities match, artifact SHA-256 checks pass, and existing dry-run controller
   status/run IDs remain intact. Review environment-specific paths and remote
   references in saved payloads. Re-prepare unapproved drafts if needed; never
   silently regenerate an approved request. Start one app service and perform
   the restart/redeploy acceptance checks below before switching traffic.
8. Keep the original installation stopped and the matched SQLite/private backup
   for rollback. Do not allow both installations to write. New writes in PostgreSQL
   are not mirrored back to SQLite.

For an existing PostgreSQL installation, restore a coordinated `pg_dump` and private
artifact backup instead of using the SQLite export/import procedure.

## Local checks and acceptance

From repository root, build and test without credentials or remote jobs:

```bash
python -m venv /tmp/infiro-railway-env
/tmp/infiro-railway-env/bin/pip install -r web/requirements-railway.txt
npm ci --prefix web
npm run build --prefix web
npm test --prefix web
npm ci --prefix .railway
npm run check --prefix .railway
# Optional: explicitly select a disposable LOCAL PostgreSQL test server (needs CREATE DATABASE).
# Never point this at your production Railway database.
export INFIRO_TEST_DATABASE_URL=postgresql://test_user@127.0.0.1:5432/test_admin
PYTHONPATH=src:web /tmp/infiro-railway-env/bin/python -m pytest -q tests/test_web_deployment.py
PYTHONPATH=src:web /tmp/infiro-railway-env/bin/python web/manage.py test infiro_site infiro_platform --noinput
# Run the same Django checks against a disposable local PostgreSQL database too.
DATABASE_URL="$INFIRO_TEST_DATABASE_URL" PYTHONPATH=src:web \
  /tmp/infiro-railway-env/bin/python web/manage.py test infiro_site infiro_platform --noinput
# Where a Docker daemon is available:
docker build -f docker/web.Dockerfile -t infiro-web:review .
```

For a local container check, supply a reachable local PostgreSQL `DATABASE_URL`.
Use the container with a named volume mounted at `/data`, runtime Variables from
a private, ignored env file, and `-p 127.0.0.1:8000:8000 -e PORT=8000`. For a local
HTTP probe supply `Host: <allowed hostname>` and `X-Forwarded-Proto: https` to
simulate the edge. Browser authentication needs actual local TLS, or explicitly
local settings in an isolated test volume. Never disable production cookie/HTTPS
settings on the real service. Restart the container, then recreate it from the
new image using the same volume and keys. Do not delete the volume.

The automated deployment tests cover migration ordering/failure, peer failure,
SIGTERM, fresh-process persistence, encrypted credentials, private artifacts,
HTTPS redirect/health, authentication, frozen datasets and real offline controller
dry runs. Fixtures never claim live provider verification or GPU measurements.
The Django tests cover dataset formats/safety, preparation, approval, isolation,
redaction and worker recovery with mocked provider traffic.

After an operator deploys, record these checks on the actual HTTPS origin:

- Public `/`, `/technology`, `/benchmarks`, `/evaluation-guide`, demos, JS/CSS and
  `/healthz` work; unknown pages return 404 and HTTP redirects to HTTPS.
- Register/login/logout work, session and CSRF cookies are secure, mutations
  require CSRF, and anonymous `/app/` redirects to login. Another account cannot
  read connections, frozen workloads or artifacts.
- Validate saved connections and confirm credential decryption without displaying
  plaintext. Missing original keys are a restore failure, not a reason to reset.
- Freeze a small permitted dataset at pinned dataset/model revisions, inspect its
  hash/preview, prepare with **dry run** selected, review and approve that exact
  dry run. Completion must report unmeasured dry run, with no GPU allocation.
  Alternatively `python web/manage.py platform_demo --username <test-user>` seeds
  an explicit offline fixture, with no external provider verification.
- Restart, then redeploy the same reviewed release on the same volume. Verify
  account login, connection decryption, frozen identity, prepared request digest,
  job/controller identity, and artifact checksums/downloads remain unchanged.
  Do not approve a new study just to test persistence.

## Enable live GPU execution later

Leave `INFIRO_LIVE_ENABLED=0` until an operator verifies immutable serving image
**digests** and their framework/GPU compatibility, the matching optimizer worker
contract, Nebius project/job quota and IAM create/get/list/cancel permissions,
registry access, private bucket/prefix read-write access, and immutable SecretStash
versions containing the required HF/storage secrets (and W&B if configured).
Check payload-read permissions and the execution service account, not just secret
metadata visibility. See the detailed [live checklist](../authenticated-platform.md#live-deployment-and-images).
Set `INFIRO_SERVING_IMAGES`/`INFIRO_IMAGE_ALIASES` to reviewed mappings if the
checked-in configuration is not appropriate; never trust a mutable tag as proof.

Only then set `INFIRO_LIVE_ENABLED=1` (and update the reviewed IaC value so a
later apply does not reset it), restart the one service and prepare a fresh
request for explicit user approval. Railway remains the website/CPU preparation
and controller host; Nebius runs the GPU serving/benchmark jobs. No deployment
check in this repository enables live mode or submits those jobs.

## Backups, restore and rollback

Stop writers and the worker, then take a PostgreSQL custom-format dump with
`pg_dump --format=custom --file=/secure-backup/platform.dump` using private `PG*`
connection variables, and back up the complete private artifact tree. Do not put
passwords or credential-bearing URLs in command arguments/logs. Enable backups
on the PostgreSQL service as well as the application's artifact volume and retain
encrypted off-site copies. Neither backup alone preserves the complete app.
Record image/source revision, optimizer archive checksum, DB migration level, key versions and archive checksum with each backup. Include
remote object storage retention/versioning and controller resource IDs for live
installations; the Railway volume does not back up Nebius/S3. Test restores on an
isolated volume with live mode disabled and original keys before relying on them.

For restore, stop the app/worker, preserve the current tree as an incident backup,
restore the PostgreSQL dump into an empty dedicated target database using
`pg_restore --exit-on-error --single-transaction`, restore the matched private tree
to the artifact volume, restore original secrets, run database/checksum checks,
then start the matching release and validate before reopening traffic. Never restore an old controller database while remote jobs are
still active without reconciling resource IDs and remote state.

For rollback, first stop writes and the worker. A Railway image rollback does not
roll back PostgreSQL migrations or artifacts. If schemas are backward compatible,
restart the previous reviewed image with the current database and volume after a
backup.
Otherwise restore the pre-deploy matched backup and its release/keys, accepting
and documenting loss of changes since that backup. Do not blindly reverse Django
migrations or attach an empty volume. Volume-backed rollbacks have downtime.

## Key rotation and worker recovery

For Fernet rotation, stop writers/worker and take a matched backup. Set
`INFIRO_CREDENTIAL_KEYS=new_key,old_key` (all necessary older keys last), restart
only for controlled maintenance, and re-encrypt stored ciphertext transactionally
using the configured MultiFernet. In a private `manage.py shell` on the mounted
service, without printing any key or decrypted value:

```python
from django.db import transaction
from infiro_platform.models import Connection
from infiro_platform.security import cipher, unseal
with transaction.atomic():
    for connection in Connection.objects.all():
        unseal(connection)  # verifies identity and decryption
        connection.ciphertext = cipher().rotate(connection.ciphertext.encode()).decode()
        connection.save(update_fields=['ciphertext'])
        unseal(connection)
```

Verify all accounts' connections before removing old keys, take a new backup,
restart and verify again with only the new key. Keep old keys in escrow for old
backups. Rotation preserves connection IDs/versions and job references; do not
log plaintext or replace records. Changing `DJANGO_SECRET_KEY` separately logs
users out; coordinate that as its own maintenance action.

On worker failure the supervisor stops Gunicorn, exits nonzero and Railway
restarts the unit. Check redacted logs, volume space, keys, database integrity,
and whether another worker holds the lease. Never delete lease files to bypass
an active process. OS locks release when the process exits. Expired running
operations recover after their lease (normally 330 seconds), with bounded retries
and backoff. Failed validation may be terminal; correct inputs and use the UI
retry/reprepare flow. Live controller errors can become `attention_required`:
inspect existing provider/controller IDs and remote allocations, renew credentials
when required, and reconcile before retrying. Do not clone studies, force states
in SQL, or launch another worker. Restarting Railway does not terminate Nebius
GPU jobs; cancel/reconcile them through the existing controller/provider flow.

## Local verification record

PostgreSQL support is exercised with a disposable local PostgreSQL 16 server,
including migrations, credential decryption, controller dry runs, a real isolated
worker child, SQLite controller import, private artifact checksums, key rotation,
and Gunicorn/worker restarts. The full Django suite passed against PostgreSQL
and local SQLite, along with ten deployment checks and three PostgreSQL-specific
checks (including concurrent idempotent submission and safe controller import).
No connection to the production Railway database was used.

Docker Desktop's daemon is unavailable in this WSL environment. The full Docker
image build, mounted-container/redeploy test, real Railway HTTPS routing and live
provider validation remain operator acceptance steps. No external deployment or
GPU job was launched. Production `check --deploy` retains the optional HSTS
subdomain/preload advisories until the domain owner chooses that policy.

## Keep existing accounts during deployments

Production Django settings now refuse to start without a PostgreSQL `DATABASE_URL`,
including when started without the Railway supervisor. Keep the web service reference
pointed at the same PostgreSQL service and database in the same Railway environment.
A redeploy runs migrations; it does not recreate users. Never run database flush or
replace the database as a deployment step. Keep `DJANGO_SECRET_KEY` and
`INFIRO_CREDENTIAL_KEYS` stable, and retain the `/data` private-artifact volume.

Before switching an existing SQLite installation, back up its database and private
artifacts and follow the SQLite migration procedure above, including users and
controller data. Do not register replacement accounts before restoring existing data.
If accounts disappeared after an earlier deployment, compare the old/new database
references and inspect the old volume or backups. Recovery requires that original
data; code changes cannot restore an erased container filesystem.

After deployment, verify an existing account can log in and see its saved connections
and runs, then repeat after a restart and redeploy. Take a PostgreSQL backup before
migration or changing database references.

Random token workloads are no longer offered for new preparations or approvals.
Use the default bounded HF dataset or select a validated frozen dataset. Historical
random runs remain readable. New storage connections use permanent access keys;
existing encrypted temporary credentials remain readable by execution workers.
Failed/cancelled runs and completed runs with failed trials expose a rerun action.
It creates a new review of the original request, linked to the original run, with
fresh provider checks and explicit approval. The review supports validated
`request.json` editing/upload; this reruns the configured search, not individual
Optuna trial records.
