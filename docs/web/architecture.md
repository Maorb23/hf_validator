# Public website architecture

The approved dark prototype is the visual foundation. The website lives in `web/`; the two design explorations remain in `design-prototypes/`.

## Boundaries

Astro renders public pages to static HTML. React islands provide architecture selection, search-trace replay, Pareto visualization, filters, comparisons, and the model journey. Styles and SVG charts use the existing dark design tokens; custom SVG keeps the chart payload small and allows native keyboard-selectable points and table alternatives. No chart library or client-side optimization engine is necessary.

Shared layout and components are independent of providers. No service credentials, live database, controller process, or application API is required. No authentication, `/app`, GPU actions, or public deployment was implemented.

Django now serves the unchanged Astro artifact. `web/infiro_site/` routes pretty page paths to their built HTML and WhiteNoise serves the original asset/data URLs. There is no Django template rewrite or Node proxy. The serving layer has no database configuration or inference imports. A build-reference check catches incomplete artifacts, and `/healthz` provides frontend-only readiness. See `deployment.md` for local setup and Gunicorn configuration.

## Data

`useDataset` loads versioned local JSON and supports recoverable errors. The search playground replays real Optuna-generated traces. The inference explorer computes illustrative dominance and constraints locally; actual measured-data export reuses the existing Python Pareto analysis. Neither mechanism schedules trials.

Public measured-study replay is intentionally unavailable: audited artifacts do not meet the publication contract. The studies page describes the missing evidence. It does not substitute a simulated trial for a measured study.

The repository-level exporter accepts explicit source files, rejects incomplete metadata, retains source fingerprints, and strips settings outside a small public allowlist. Publication remains an explicit reviewed content change.

## Reproducibility

Two generation runs produced the same SHA-256 payload: `b3adb1d24a402d6f702b786f912e384baa3cab99ce9d11d7f4cf5501e044bc02`. This applies to the search simulation dataset in the recorded CPU environment. It is not a cross-platform numerical guarantee.

## Development integration

Development and production use explicit NODE_ENV settings and distinct Vite dependency caches. This prevents a production prebundle from overwriting React's development JSX runtime while the preview is running. File polling supports the Windows-mounted checkout.

## Future work

An authenticated operational application can live under `/app` with a same-origin Python API. It must reuse the existing recommendation operations, exact prepared-request approval, validation workers, and optimizer controller. It must not make public builds depend on those services.
