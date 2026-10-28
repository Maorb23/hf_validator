# Web optimization controls and execution evidence

Studies → Choose a model directly accepts a Hugging Face repository and branch,
tag or commit. A credential-isolated `model` worker operation verifies access and
pins the commit. It stores selection provenance without generating recommender
scores or discovery results. Configure then reuses normal workload preparation
and runtime research. Public-model pinning needs no inference account; preparation
still needs the configured inference connection. Private or gated repositories
use the selected Hugging Face connection.

Before approval, Request JSON → Edit request.json loads the full request on
demand. Users can edit it or upload JSON, then validate and save. Supported edits
include serving argument domains, native sampler/pruner settings, objectives,
trial counts, GPU type/count, matching preset and timeout. Model identity,
immutable workload, images, config files, storage and provider policies remain
protected; use a new configuration to change those. Arbitrary component imports
and unverified speculative overrides are rejected. The editor uses the native
optimizer request and serving-argument parsers, plus the web budget bounds.

Saving creates a new prepared review, regenerates intent and artifacts, refreshes
allocation estimates, and queues existing provider-reference validation. It
invalidates the source review without overwriting its artifacts. No study starts
until the revised request has its own digest and explicit approval. Stale,
already-approved and other users' requests cannot be edited. Open editors disable
approval, and navigation to a revised request clears previous approval consent.

Configure saves the selected objectives, sampler and pruner in the private
prepared configuration. Preparation uses the existing semantic `StudyContext`
and optimizer request contracts. Approval binds the settings into the digest;
execution submits that exact request. The results artifact includes
`optimization_settings`, trial records and framework timing totals.

New web runs use output throughput (maximize), 95th-percentile end-to-end latency
(minimize), and 95th-percentile time to first token (minimize), matching the
experiment and v2 notebook's latency semantics. Every pair and the three-objective
combination are available at every concurrency. Workspace loading serves display
metadata without importing the optimizer package. Native optimizer validation
checks actual study requests, and its direction helper drives result selection;
tests check the display metadata against those helpers. Existing saved mean-based
requests and results retain their original metrics. CLI defaults remain unchanged.
Three-objective plots offer a choice of two display axes; Pareto membership and
selection still use all three objectives. The optional naming field is Run name.

Workspace updates bypass browser caches, cancel outdated reads before actions,
and reject stale responses. Polling pauses during an action and resumes afterward;
returning to the tab also requests current data. Update failures and timeouts
show a retry control while automatic polling continues. Successful actions are
not reported as failed solely because the subsequent workspace read failed.

Samplers are TPE, multivariate TPE, random, grid, GP and NSGA-II. Multivariate
TPE uses the native `TPESampler` with `sampler_kwargs: {"multivariate": true}`;
the setting is preserved in approval and saved results. It can use the same
supported scalar objectives and Pareto combinations as standard TPE. Grid uses the optimizer's
generated search space; GP requires its optional dependencies on the worker.
Pruners are none, median, successive halving and Hyperband, using optimizer
defaults and the fixed workload seed. Pruning requires an unconstrained scalar
objective. Configure disables it for pairs, and backend validation rejects
unsupported combinations before approval. Arbitrary component imports, custom
kwargs and uncalibrated threshold pruning are not exposed by the web API.

Worker `metadata.execution.timings_s` provides `launch_to_ready` and `benchmark`
seconds. Partial phases, negative/nonfinite values and absent evidence are
unavailable. Zero is retained when explicitly measured. Framework startup and
benchmark totals require evidence for every recorded trial; GPU allocation
seconds and request latency metrics are never substituted for these durations.

Framework queue wait is controller acceptance to the first observed running
state. It includes provider waiting and polling delay, and is an observation,
not a precise provider scheduling measurement. The first observation persists
across retries and terminal reconciliation. Runs completed between polls, dry
runs and old runs with no running observation have unavailable queue timing.
Queue wait is a framework allocation measurement, so it is not copied to each
trial. A trial queue duration is shown only if the worker reports one.

Results retain failed, pruned and invalid trials for inspection while excluding
them from the comparison plot and selection policy. The plot uses the selected
objectives; scalar studies plot recorded trial order on the horizontal axis.
Colors identify frameworks, white rings identify Pareto points, and focus or
hover exposes trial identity and all three metrics. Points support Enter/Space;
table rows support pointer selection and native inspection buttons. The inline
inspector receives focus and Escape/Close returns focus to its trigger. Tables
and the mobile plot can scroll locally by keyboard or touch.

Older workers may not export failed/pruned trial parameters or timings. These
remain unavailable. Logs use the existing owner-scoped, scrubbed diagnostics
endpoint and show a bounded framework log tail explicitly shared across trials;
they do not claim to be isolated trial logs. No historical timings are backfilled
and no live GPU allocation is needed for the regression suite.
