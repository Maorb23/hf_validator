# Infiro website and private platform

The authenticated product is a separate `/app` area with its own React app shell,
session authentication, encrypted connections, private datasets, recommendations,
configuration review and explicit approval, plus a durable worker/controller
integration. Public pages and bundled demonstrations remain intact.

For the private product, use **[local startup, dry-run demonstration, provider
requirements and tests](../docs/authenticated-platform.md)**. Its database,
migrations, CPU dependencies and separate worker are required for `/app`; the
public-only serving instructions below remain available for public previews.
See the [verification report](../docs/platform-verification.md) for tested behavior,
screenshots and the unchanged sibling optimizer test failure.

Astro pages and React visualization islands built from the approved dark design, served by a lightweight Django layer. The built HTML and frontend URLs are preserved. The existing Python inference workflows and the `design-prototypes/` comparison remain separate.

## Run the website with Django

From the repository root, enter **`web/`** first:

```bash
cd web
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-django.txt
npm ci
npm run build
python manage.py runserver 127.0.0.1:8000
```

Open **http://localhost:8000**. Routes include `/technology`, `/architecture`, `/benchmarks`, `/methodology`, `/evaluation-guide`, `/demo`, `/demo/search`, `/demo/pareto`, `/demo/journey`, and `/demo/studies`.

No database, migrations, `collectstatic`, API keys, or GPU dependencies are needed. Django serves `dist/`; it does not run Astro or proxy to Node. Rebuild after frontend edits, then restart Django to refresh its static asset index. `manage.py check` detects missing or incomplete builds.

For subsequent starts, use `source .venv/bin/activate` and `python manage.py runserver 127.0.0.1:8000` from `web/`. From the repository root you can use `web/.venv/bin/python web/manage.py runserver 127.0.0.1:8000`.

## Frontend development and Astro fallback

The original workflow remains available:

```bash
cd web
npm run dev
```

This starts Astro at http://localhost:4321. `npm run preview` serves the built site on that port. Both remain independent of Django.

The server binds loopback. File polling accommodates a Windows-mounted WSL checkout. Astro telemetry is disabled. The foreground development server uses `--ignore-lock` to avoid agent-specific background startup.

## Build and check

```bash
npm run build
npm test
python tests/test_export.py
python manage.py check
python manage.py test infiro_site
INFIRO_URL=http://127.0.0.1:8000 PLAYWRIGHT_BROWSERS_PATH=/tmp/infiro-playwright npm run check:browser
```

For the Python check above, run from repository root as `python web/tests/test_export.py`, or from `web/` as `python tests/test_export.py`. It uses only the standard library.

Install Chromium with `PLAYWRIGHT_BROWSERS_PATH=/tmp/infiro-playwright npx playwright install chromium` if necessary. Browser checks use the running server; `INFIRO_URL` selects a different local origin. Screenshots and verification results are written to ignored `screenshots/`.

Stop the Astro dev server before using `npm run preview` on port 4321. The frontend build requires neither Python nor the sibling optimizer checkout. Set `INFIRO_URL` as above to verify Django; browser tests otherwise default to port 4321. No provider credentials belong in this project.

## Data and simulations

- `public/data/simulations.json`: actual Optuna TPE/GP/random traces on two deterministic synthetic objectives and two seeds; 24 trials per strategy. Includes dependency versions and canonical-data checksum.
- `public/data/inference-examples.json`: synthetic serving examples, identical formulas across engine labels. Not inference measurements.
- `public/data/publication.json`: no published measured studies. Audited optimizer files contain placeholder model revisions; other inspected records do not establish complete publication context.

The seed and objective control the recorded experiment. Playback controls reveal stored observations; changing a latency limit filters examples locally. Neither feature launches optimization or inference.

To regenerate offline datasets in an isolated CPU environment:

```bash
python -m venv /tmp/infiro-sim-env
/tmp/infiro-sim-env/bin/pip install -r scripts/requirements-simulations.txt
/tmp/infiro-sim-env/bin/python scripts/generate_simulations.py
```

Generation was run twice with the same environment and produced the same checksum. No timestamp enters the canonical payload. Exact cross-platform floating-point reproducibility is not promised; retain the bundled artifact for identical browser playback.

## Curating measured data

The repository-level `scripts/export_public_inference_results.py` accepts explicit optimizer JSONL paths and rejects placeholders, incomplete workloads, failed benchmarks, nonfinite metrics, and missing runtime identity. It exports an allowlist of inference metrics/settings and reuses `optimization.analysis.pareto_front`.

Run from repository root:

```bash
PYTHONPATH=src python scripts/export_public_inference_results.py --input /path/to/reviewed-trials.jsonl --output /tmp/reviewed-public-snapshot.json
```

Review the snapshot before publication. The current public site deliberately has no measured-study replay because no audited data qualifies. Do not silently replace `publication.json` with unreviewed exports: measured cohort selection and replay presentation require a reviewed publication change.

## Future publishing

No public deployment is configured or authorized. `dist/` is a static artifact. Set `SITE_URL` to the approved canonical origin at build time to populate canonical links and the sitemap. Without a domain the sitemap is valid but empty; no guessed production hostname is embedded.

For deployment, use Gunicorn with `infiro_site.wsgi:application`, not Django's dev server. See [Django deployment and rollback](../docs/web/deployment.md). `/healthz` reports whether the frontend artifact is present.

The name Infiro is provisional. Contact details have not been supplied, so the real conversion is an ungated evaluation-guide download; no fake contact form or booking action is shown.

## Public/private boundary

Public pages and bundled demonstrations remain independently buildable. Private
data is never exported to public JSON or static builds. Live submission is disabled
by default and requires verified immutable worker images and matching Nebius
SecretStash credential references; local testing launches no GPU jobs. See the
[platform guide](../docs/authenticated-platform.md) for the precise behavior and
remaining deployment requirements. Billing is not implemented.

### Import recorded optimizer runs

From the repository root, explicitly select the run-export JSON:

```bash
python scripts/import_public_optimizer_runs.py \
  --input ../bento_extend/llm-optimizer/results/sorted_results/q35_4_runs_c128.json \
  --output web/public/data/publication.json
cd web
npm test
npm run build
```

Restart the Django process after rebuilding: WhiteNoise indexes built assets at startup.
The imported snapshot is served at `/benchmarks` (Published measurements) and
`/demo/studies` (run replay). No database, upload endpoint, or GPU execution is involved.

All selected runs and trial states are retained, including crashed runs and incomplete
metadata. Placeholder values become explicit unknowns. Model, recorded draft model,
seed, dataset/workload, request fingerprints, and known hardware/revision define
comparison cohorts. Unreported draft models are not assumed absent. Only completed
measured trials with positive finite throughput and p95 E2E latency, and no reported
failed requests, enter the chart and Pareto frontier. Other trials remain inspectable.
Imported latency and TTFT are p95, while synthetic examples use mean latency.
The source `MEASURED_ALLOCATION_WALL_PROXY` is preserved rather than reinterpreted.
Snapshots contain a SHA-256 of the input; internal paths and raw payloads are omitted.

The earlier `scripts/export_public_inference_results.py` remains available for its
separate strict JSONL export workflow; it is not used to reject these run exports.

Benchmark filters are staged: select model, engine, search strategy, data group or
run, then press **Apply filters**. The default includes all matching data groups.
Changing model, engine or strategy clears a specific group restriction; select a
group explicitly to narrow results. The highest observed throughput is surfaced
and the trial table sorts throughput descending. Mixed-group browsing does not
merge Pareto frontiers or permit cross-group configuration comparisons.


The seven-model import and metadata resolutions are documented in
[benchmark import verification](../docs/benchmark-import.md). The importer accepts
repeated `--input` arguments and `--merge web/public/data/publication.json` to preserve
existing runs. Backups are rejected. Baseline/DSpark variants have an explicit filter;
study and trial inspectors expose sanitized execution evidence and metadata conflicts.
No minimum-duration threshold applies.
