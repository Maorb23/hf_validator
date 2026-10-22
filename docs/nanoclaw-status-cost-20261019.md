# NanoClaw status and chat-cost update

The status chat previously used the optimizer ID in a recommendation-only field, then
retried after an MCP process exit and delegated troubleshooting to another agent. A status
request also fetched S3 artifacts. The persistent provider conversation had grown past 217K
tokens, multiplying the cost of each extra model call.

Changes:

- `openclaw_adapter.py`: validate either recommendation or study IDs; resolve study IDs
  within the current group. Unknown or ambiguous references return an error.
- `optimization/workflow.py`: lightweight status reads skip S3 and result analysis.
  Full result analysis remains available via `include_results=true`.
- `nanoclaw_mcp.py`: unexpected exceptions return sanitized tool errors instead of
  terminating stdio. Tool descriptions explain both IDs and local status behavior.
- NanoClaw `container/agent-runner/src/providers/opencode-config.ts`: HF Model Watch only
  gets context/output ceilings of 32768/4096 tokens, automatic compaction and pruning,
  four agent steps, and disabled bash/task recovery and general/explore subagents.
- Template and deployed group instructions require concise status replies and at most
  one reconnect attempt; no shell investigations or delegation for routine status errors.

Backups of installed modules, provider code/tests, group instructions and the provider
conversation database are in
`/home/maorb/hf-model-watch-nanoclaw/backups/before-status-cost-fix-20261019T101112Z/`.
Run history and saved workload preferences were preserved. No optimizer submission,
controller tick or GPU launch was performed.

Validation: 95 focused Python tests, 32 offline provider configuration tests, and the
NanoClaw agent-runner TypeScript check passed. An isolated test against the installed Python
runtime and snapshots of real state returned the same study via either identifier in 24.3 ms
and 2.6 ms, with 424-byte responses. A failed lookup left stdio usable. Network access was
turned off during this test.

These timings measure the tool, not end-to-end model/chat latency or billed cost.
An existing oversized provider conversation may need a one-time compaction on its next turn.
The provider patch is saved in `patches/nanoclaw-hf-chat-cost.patch`.

## Resumed model correction

The first deployment capped only the configured default model. The existing native session
still selected its older saved model, so the cap did not trigger compaction for that model.
The runner now reads recent local session metadata before a resumed HF Model Watch prompt,
registers the saved model with the same context/output ceilings, and rebuilds the native
server configuration before sending a prompt if required. It prefers the saved user-model
selection over a compaction assistant model and never switches the user's chosen model.
Unknown metadata or an unconfigured provider fails before sending an uncapped prompt.

Backup of the prior implementation and provider history:
`/home/maorb/hf-model-watch-nanoclaw/backups/before-resumed-context-20261019T110119Z/`.

Validation: 78 provider tests passed, including a real OpenCode native test with a loopback
fake model API and external networking disabled. That test seeded a legacy-model session
with 220K reported input tokens, changed the configured default, then verified that the saved
model remained selected and automatic compaction removed the original history before the
next normal model request. No paid model APIs were used. TypeScript and Prettier checks passed.
The next real conversation turn may still make one larger summarization call; subsequent
normal calls should use compacted history. Live billed token usage has not been measured.

## Inherited input-limit correction

A live `/provider` inspection then showed both Luna entries had `context=32768` but still
inherited `input=922000` from the provider catalog. That separate input limit prevented
compaction. The HF policy now explicitly caps `limit.input` to `context-output` (28672
with the default 32768/4096 policy), preserving any smaller configured input limit.
OpenCode's [configuration schema](https://opencode.ai/config.json) exposes these as separate limits.

Backup: `/home/maorb/hf-model-watch-nanoclaw/backups/before-input-limit-20261019T112920Z/`.
The native regression now seeds the inherited 922K input limit and cached input tokens,
verifies the native resolved model catalog reports 28672 input tokens, and verifies
compaction precedes normal resumed requests. 79 tests and TypeScript/Prettier checks passed.
No paid model API was called during validation. Existing oversized history may still incur
one larger compaction request on the next user turn; normal subsequent requests should
use compacted history. Live billed usage after this correction has not been measured.

## Live verification after the input-limit correction

Read-only inspection of the existing OpenCode session database on October 19 confirmed
the two terminal health checks used compacted history. No additional chat was run for
this inspection.

- At 11:36:34 UTC, the first check triggered automatic compaction. Its summarization
  call used 87,933 input tokens, counting uncached input plus cache writes and reads.
  The following normal calls used 18,743 and 19,055 input tokens.
- At 11:45:45 UTC, the second, brief health check used 19,105 and 19,259 input tokens
  across its two model calls. These counts include cached input; the raw `tokens.input`
  field alone was only 3 for each call and would understate context size.
- The earlier 11:17 check used 140,688 and 140,835 input tokens per call. The second
  check therefore reduced input context by approximately 86% per call.
- OpenCode recorded costs of $0.00047483 and $0.00045474 for the second check,
  totaling $0.00092957. These are local recorded estimates, not independently verified
  provider billing. The one-time summarization call recorded $0.0232035.

The second check's saved response reports the controller healthy and connected, with
no jobs submitted. The first terminal response reached the agent step limit despite
completing its health check; the second returned the requested brief answer. This
verifies live context reduction for these turns, without establishing future usage
or invoice totals.

## Study lifecycle and context isolation correction

The later approved start returned the previously linked study rather than submitting a
new request: that study was already cancelled at 10:01 UTC. The workflow previously
returned any existing reference before considering its terminal state, and all
preparations shared the same idempotency key.

Each preparation now receives a fresh key. Repeating submission of that preparation
remains idempotent. A fresh approved preparation can replace a terminal reference;
an active or uncertain study blocks a different request. The controller also checks
for nonterminal jobs inside its submission transaction, preventing concurrent direct
submissions from creating two active studies. The workflow serializes submissions
across recommendation runs using a separate shared lock. Responses explicitly mark
whether a new submission was created. The prepared request and intent equality checks
remain mandatory.

Conversation context now omits saved study intents, old results and recommendation
ancestry, and checks the current controller status. Terminal studies appear only as
`not_active`. Audit files remain on disk. HF Model Watch starts a fresh native OpenCode
session for each incoming turn, including queued follow-ups, and explicitly preserves
the saved model choice. Previous messages and compaction summaries are excluded from
new model requests. Current workload, selection and preparation are recovered through
MCP state rather than replayed chat history. Other groups retain their existing behavior.

Installed into the host runtime and NanoClaw source; the HF agent was reloaded without
a chat and the controller restarted. Initial backups:
`/home/maorb/hf-model-watch-nanoclaw/backups/before-study-lifecycle-20261019T122956Z/`;
original regression tests are backed up under `before-study-lifecycle-20261019T123523Z/`.
Cross-repository patches: `patches/nanoclaw-study-lifecycle.patch` and
`patches/optimizer-study-lifecycle.patch`.

Validation: 158 focused HF tests, 7 controller service tests, 64 provider tests,
TypeScript and formatting checks passed. The native regression used the real OpenCode
binary with a loopback fake model endpoint, verifying a different session ID, preserved
legacy model selection, input cap of 28672, and exclusion of both raw history and old
compaction text. No paid model requests or new studies were launched. The final local
queue read found zero nonterminal jobs and the same newest study as before deployment.
Live end-to-end chat behavior after this deployment remains unverified; the next attempt
must prepare a fresh request and obtain approval before submitting it.

## Follow-up state recovery correction

The first live fresh-session preparation exposed another dependency on chat history:
selection and preparation had not updated `conversation.json`'s active run. That pointer
still referenced an older incomplete recommendation. The agent then sent optimization
budget answers to the recommendation clarification tool, which rejected the completed
recommendation. The previous lifecycle deployment did not catch this end-to-end gap.

Selection and preparation now publish the active run, remove completed job pointers,
and persist effective workload choices. Context lookup prioritizes that pointer over
historical job output and returns a compact current selection, optimization controls,
choices, input stage and next tool instead of the full recommendation research.
Preparation preserves explicit dry-run/trial choices even before model selection.
`provide_active_input` routes a selected model's optimization answers to preparation,
without reopening discovery or holding nested conversation locks. Group instructions
also require stage-specific routing and preserving explicit pending controls.

Validation: 162 focused Python tests passed, including fresh adapter instances for
pending controls, selection, budget clarification and final preparation. A CLI input
test was isolated from remembered user state by giving it a temporary state root.
Installed runtime and group instructions were backed up under
`/home/maorb/hf-model-watch-nanoclaw/backups/before-followup-state-20261019T130142Z/`.
The active run's explicit dry-run/two-trial controls were restored; its budget and GPU
price were left unchanged. No model chat or optimization submission was launched.

## Discovery before deployment intake

Fresh-state discovery now accepts model requirements without deployment inputs. A separate
`DiscoveryRequestSpec` permits unknown GPU memory and budget while explicit-model
`RequestSpec` retains its required fields. Discovery still uses the shared Hub scanner
and ranker, defers only unavailable hardware/pricing checks, and reports unknown deployment
fit. Known memory and budget violations continue to exclude candidates. Discovery does
not invent a GPU price or claim a supplied workload was benchmarked.

The CLI no longer prompts for optimization inputs before recommendation research. Selection
still triggers preparation and missing deployment inputs are collected there. Context hides
optimization execution defaults before selection and retains shortlist reasons/limitations.
Repeating a legacy request that stopped on deployment-only questions can start discovery
again. Standing instructions were updated in `instructions.prepend.md`, which survives
container restarts, rather than only the generated `CLAUDE.md`.

Validation: 166 focused Python tests passed, including size-only CLI discovery, unknown-fit
candidate retention, known memory exclusion, unchanged explicit-model requirements, and
retry of old deployment questions. Installed files match tested source and the group was
reloaded. No live model chats or optimizer submissions were used for this validation.
Initial backup: `before-optional-discovery-20261019T132040Z`; final deployment backup:
`before-optional-discovery-20261019T132143Z`, under the existing backups directory.

## Active discovery status precedence

A running recommendation can coexist with an old `active_run_id`. Context lookup previously
reported the running job and rendered the old run's `needs_input` state together. It now
suppresses the old run while a new recommendation job is starting/running and prefers that
job's resulting run after completion. Starting a new discovery also clears the active run
pointer; audit files remain intact. Persistent instructions tell the agent to report only
scan progress while the job runs.

The user-provided trace demonstrates discovery completed and returned a shortlist with
speculative-decoding pairings. The incorrect part was the earlier status message, which
mixed scan progress with stale missing-field questions. Installed the status correction and
reloaded the HF agent. No new chat or optimizer job was started. Backup:
`/home/maorb/hf-model-watch-nanoclaw/backups/before-active-discovery-status-20261019T133232Z/`.
