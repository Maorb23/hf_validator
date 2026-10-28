# Railway website and authenticated app

Railway runs one CPU service: Gunicorn serves the Astro artifact and Django `/app/`
APIs, and `platform_worker` handles the durable queue and preparation. Nebius runs
GPU workloads. The root `Dockerfile` remains the separate model-worker image.
This setup intentionally keeps SQLite and local artifacts; do not add replicas,
regions, a separate worker service, or ephemeral storage for private state.

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
- `.railway/railway.ts`: one service, one replica in `europe-west4-drams3a`, one 10 GiB
  volume mounted at `/data`, `/healthz`, no overlap, 30-second drain and failure
  restarts. Choose region and initial size before provisioning. SQLite and cache
  growth require monitoring free space. Disable Railway Serverless sleeping.

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
INFIRO_DATABASE=/data/private/platform.sqlite3
INFIRO_LIVE_ENABLED=0
```

Hostnames have no scheme, port, or path; add any custom hostname explicitly.
Use a password manager to generate/store the Django secret (for example,
`secrets.token_urlsafe(64)`) and a Fernet key (`Fernet.generate_key()`). Generate
keys privately, outside shared logs and shell history. Keep both stable across
restarts. For a migration, use the **original** encryption key(s), never generate
replacement keys for existing ciphertext. Escrow keys separately from backups.

Enter provider secrets only through the authenticated Connections UI; they are
encrypted in SQLite and passed to isolated child environments only as needed.
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
| Object storage | Provision a private S3-compatible bucket; enter access key ID, secret key, optional session token, bucket, region and public HTTPS endpoint if needed | HeadBucket plus intended prefix read/write permissions |
| Tavily | Create a Tavily account/API key | Search validation and sufficient quota |

Provider validation makes outbound requests and may consume provider quota, but
is not permission to launch GPU jobs. Missing, unverified or denied connections
must remain visible as errors. HF access is not evidence of runtime support.
For live jobs, HF/storage credentials additionally need immutable Nebius
SecretStash version references; UI credentials alone do not prove worker access.
See [provider requirements and live prerequisites](../authenticated-platform.md).

## Move an existing installation

1. Keep `INFIRO_LIVE_ENABLED=0` on the target. Schedule maintenance, block new local
   requests, stop local web writes and **all** local platform/controller workers.
   Reconcile any running Nebius workloads first: stopping Railway/local processes
   does not cancel remote GPU allocations. Do not start a second controller.
2. Record the exact application/optimizer revisions and save the original Django
   secret and `INFIRO_CREDENTIAL_KEYS` in secure escrow. Find the actual source DB
   and private root; defaults are `web/private/platform.sqlite3` and `web/private`.
3. With writers stopped, use SQLite's backup API for a consistent DB copy:

   ```python
   import sqlite3
   with sqlite3.connect('/source/private/platform.sqlite3') as source:
       with sqlite3.connect('/secure-backup/platform.sqlite3') as target:
           source.backup(target)
           assert target.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
   ```

   Copy the **entire** private root to the backup directory as well, preserving
   user/job directory names, frozen workloads, artifacts, and workload caches.
   Replace the copied database with the backup API copy; omit stale SQLite journal,
   WAL/SHM and lease files only after clean shutdown. Never copy only the main DB
   while writers are active. Make an encrypted archive and checksum it; restrict
   permissions and retain an untouched source backup.
4. Provision `/data` before application startup. With the destination app stopped,
   use an authenticated administrative transfer/maintenance container attached to
   that volume to extract the encrypted backup into `/data/private`, then verify
   checksums, SQLite integrity and permissions (`0700` directories, `0600` files).
   Railway SSH operates on the mounted remote container; `railway run` executes
   locally and does **not** mount the volume. Use an encrypted private storage
   transfer if direct copying is unavailable; never expose a public upload route,
   public bucket, or bake a backup into an image. Stop the maintenance container
   before starting the app. Do not run app migrations during the transfer.
5. Set the target Variables including the **original** credential keys. Start one
   service. Startup applies migrations after mount. Check every migrated
   connection decrypts, sample artifacts match stored hashes, and frozen dataset
   identities survive. Review stored payloads/controller state for old absolute
   paths or environment-specific references. Re-prepare unapproved drafts if
   needed; never mutate an approved request or regenerate its approval silently.
6. Run the acceptance checklist below, including a restart and redeploy. Switch
   DNS/traffic only after verification. Keep the old installation stopped and the
   original backup for rollback; never allow both installations to write.

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
PYTHONPATH=src:web /tmp/infiro-railway-env/bin/python -m pytest -q tests/test_web_deployment.py
PYTHONPATH=src:web /tmp/infiro-railway-env/bin/python web/manage.py test infiro_site infiro_platform --noinput
# Where a Docker daemon is available:
docker build -f docker/web.Dockerfile -t infiro-web:review .
```

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

Enable Railway volume backup schedules and separately retain encrypted off-site
backups of a quiesced private tree plus a SQLite backup API copy as above. A volume
snapshot without coordinated writers may need SQLite recovery and does not prove
DB/artifact consistency. Record image/source revision, optimizer archive checksum,
DB migration level, key versions and archive checksum with each backup. Include
remote object storage retention/versioning and controller resource IDs for live
installations; the Railway volume does not back up Nebius/S3. Test restores on an
isolated volume with live mode disabled and original keys before relying on them.

For restore, stop the app/worker, preserve the current tree as an incident backup,
restore a matched DB/private-tree backup to the volume, restore original secrets,
run integrity/checksum checks, then start the matching release and validate before
reopening traffic. Never restore an old controller database while remote jobs are
still active without reconciling resource IDs and remote state.

For rollback, first stop writes and the worker. A Railway image rollback does not
roll back SQLite migrations or artifacts. If schemas are backward compatible,
restart the previous reviewed image with the current volume after a backup.
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

On 2026-10-28, the dedicated requirements installed into an isolated Python 3.12
CPU environment, using only the vendored optimizer snapshot and this checkout.
Dependency checks passed without torch or serving runtimes. Astro production build
and 13 frontend tests passed; the Railway SDK graph check passed; 132 Django tests
and 53 recommender/discovery tests passed. Seven focused deployment checks exercise real
Gunicorn/worker restarts, shutdown and failure handling, encrypted-state/artifact
persistence and the documented Fernet rotation. Production `check --deploy` reports
the existing optional HSTS subdomain/preload advisories: those are not enabled
without a domain owner deciding their policy.

Docker Desktop's daemon is unavailable in this WSL environment. The full Docker
image build, mounted-container/redeploy test, real Railway HTTPS routing and live
provider validation remain operator acceptance steps; local process tests do not
claim those were performed. No external deployment or GPU job was launched.
