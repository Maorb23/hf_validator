# Benchmark import verification

The Benchmarks explorer and study replay now include all seven selected exports,
with no study-duration cutoff. The source exports were read without modification;
backup files were not selected. See [machine-readable audit](benchmark-import-audit.json)
for counts and state breakdowns.

| Export | Studies | Trials | Valid measurements | Diagnostics |
| --- | ---: | ---: | ---: | ---: |
| MiniCPM5-2B | 8 | 118 | 84 | 34 |
| Gemma 4 12B | 16 | 228 | 171 | 57 |
| DSpark Gemma 4 12B | 5 | 98 | 62 | 36 |
| Qwen3-4B | 24 | 474 | 395 | 79 |
| DSpark Qwen3-4B | 24 | 445 | 218 | 227 |
| Qwen3-8B | 6 | 120 | 112 | 8 |
| DSpark Qwen3-8B | 6 | 120 | 79 | 41 |
| Selected exports | 89 | 1,603 | 1,121 | 482 |

Eighteen studies and 354 trial numbers overlap the existing publication. The final
publication contains **185 studies and 3,197 trials: 2,394 valid measurements and
803 diagnostics**. All 114 prior studies and 1,948 prior trial records remain;
original trial IDs, states, measurement values and eligibility are preserved.
Overlapping studies receive richer provenance and corrected identities. Deduplication
uses `(run ID, recorded trial number)` because the earlier importer used row positions
in display IDs. Contradictory measurements on the same trial raise an error.

Every selected study and trial records concurrency **128**. Config, workload,
environment, summary (where present), and per-trial execution concurrency were checked.
There are no concurrency conflicts in these inputs. Server batch/sequence limits are
preserved as serving parameters and do not override client concurrency.

Thirty-two studies have stale model labels. Snapshot repository and immutable revision,
`hf_target_model_id`, and tokenizer identity establish the resolved model; copied
`Qwen/Qwen3.5-4B` fields remain visible in the conflict evidence. Sixteen studies have
DFLASH descriptive metadata alongside native `dspark` execution settings. Native vLLM,
SGLang and TensorRT-LLM speculative settings establish the DSpark variant. Draft model
identities and revisions remain attached. Baseline is an inference from the absence of
recorded draft/speculative settings, disclosed in `variantBasis`; legacy runs lacking
this evidence show an unreported variant.

All four studies shorter than five minutes remain: `7hksyocg` (recorded 0 seconds,
no trials), `0bxo3s5e` (255.475 seconds), `x911x1uq` (150.238 seconds), and `by7lkkyu`
(181.022 seconds). Each of the latter three contributes two valid measured trials.
Study final status, duration and interruption do not disqualify complete individual
measurements. Twenty valid measurements from four crashed studies are retained. Failed, pruned and incomplete trials stay inspectable as diagnostics;
they cannot enter comparisons, rankings or Pareto fronts.

Imported chart metrics are p95 E2E latency, output throughput and p95 TTFT. Mean
measurements, if present, have separate fields. Values were checked directly against
each source trial, without recomputation from timing proxies. Framework/runtime,
model/draft revisions, workload, request fingerprints, hardware and measurement source
separate comparison cohorts. Unknown essential provenance isolates studies rather than
asserting compatibility. Mixed-group browsing displays observed values; it does not
establish normalized rankings across workloads or engines.

The inspector preserves sampler settings (including multivariate/group TPE), pruner,
objectives, constraints, budgets, effective settings, sampled parameters, available
execution timings, request accounting and sanitized log evidence. Original W&B run links
and source checksums are exposed. Local paths, commands, endpoints and credentials are
removed or redacted; no local log files are read or published.

Limitations: none of the 89 selected studies provides an explicit termination reason,
so all remain unavailable. Final state and recorded interruption are shown separately.
Some exports lack execution-phase details or contain only local log-file pointers;
those cannot supply public log contents. Measurements are imported evidence, including
`MEASURED_ALLOCATION_WALL_PROXY`, and are not independent runtime validation. No GPU
jobs were launched. This change builds the local site; public deployment is not configured.

## Reproduce

For an independent source audit, save the prior publication before importing and run
`scripts/audit_public_optimizer_import.py --publication web/public/data/publication.json
--before /path/to/prior-publication.json --source-dir /path/to/sorted_results
--report /tmp/benchmark-audit.json`.

From the repository root, use the existing importer with repeated explicit `--input`
arguments and `--merge web/public/data/publication.json`. The merger reads the existing
publication before writing, so input and output may be the same path. Select only the
seven filenames in `web/public/data/publication.json`'s `sources` manifest; `.bak` inputs
are rejected. The previous publication checksum is retained alongside the seven source
checksums, and overlapping studies retain both provenance hashes.

```bash
python -m pytest -q tests/test_public_optimizer_import.py web/tests/test_export.py
cd web
npm test
npm run build
npm run preview
# In another terminal:
npm run check:browser
node tests/benchmark-import-browser.mjs
```

The focused browser test checks all seven model/variant combinations at 1440px and
390px, keyboard point selection, source links, unavailable termination reasons,
inspector metadata, short/empty studies and horizontal overflow. The existing browser
suite covers all public routes and regression interactions.

Completed validation: 14 Python importer/publication-boundary tests, eight frontend
data tests, the production build (13 pages), the independent source audit, and a
full idempotent re-import. The existing browser suite passed for 11 public routes
at desktop/mobile sizes, including dataset recovery and regression interactions.
The focused import suite passed all 14 model/variant/viewport combinations, plus
all four short/empty studies, with keyboard selection, source links, expanded
metadata and no page overflow or browser errors. Browser evidence is saved under
ignored `web/screenshots/`. Long conflict strings wrap within the mobile inspector.
