# Verification

The static production artifact was built and served locally with Astro preview. No public deployment or GPU operation was performed.

- All 11 public content routes checked at 1440×1000 and 390×844.
- Screenshots captured for every route and viewport in ignored `web/screenshots/`.
- No browser page errors or horizontal overflow in the final run.
- Search playback, stepping, reset, objective/seed changes, and strategy selection verified.
- Benchmark filters, latency infeasibility, Pareto-only view, and configuration comparisons verified.
- Model journey progression, restart, architecture node selection, mobile navigation, and Escape dismissal verified.
- All 15 discovered local link/download destinations returned successful responses.
- Unknown route returned HTTP 404.
- Simulated dataset failure produced a recoverable error; retry loaded the dataset successfully.
- Four data/analysis checks passed, plus three Python publication-boundary tests.
- Offline sampler generation was repeated and produced an identical dataset checksum.
- `git diff --check` passed.

The development checks uncovered and fixed a React JSX-runtime dependency-cache collision between build and dev processes, explicit select labeling, and a decorative hero-glow overflow. Development and production caches are now separate. Production checks were repeated after the final changes.

No verified measured-study replay is asserted: source candidates did not satisfy the publication requirements. Generated search traces and serving examples are consistently identified as simulations.

## Django serving verification

The existing frontend was rebuilt without source changes and served locally by Django 5.2 / WhiteNoise through Gunicorn on port 8000. The same 11-page desktop/mobile browser suite passed against Django: all interactive demos, local links/downloads, 404 behavior, and dataset recovery remained functional, with no page errors or horizontal overflow.

Serving-layer tests additionally cover byte-for-byte parity with every built page; both slash forms; JavaScript, CSS, JSON, font and download MIME types; immutable asset caching; ETags; HEAD and range requests; host validation; traversal and symlink rejection; missing-build readiness; production configuration defaults; and read-only behavior. The Django test runner skips database setup. Astro build/preview remains available for rollback. No public deployment was performed.
