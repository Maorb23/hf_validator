# Speculative decoding in the web app

Configure exposes one selector: Off uses ordinary decoding; Auto preserves the
existing research and exact revision/image evidence checks; Use Speculative
decoding head requires DSpark and the following fixed mapping, without research,
substitution or fallback:

| Target | Head |
| --- | --- |
| Qwen/Qwen3-4B | deepseek-ai/dspark_qwen3_4b_block7 |
| Qwen/Qwen3-8B | deepseek-ai/dspark_qwen3_8b_block7 |
| Qwen/Qwen3-14B | deepseek-ai/dspark_qwen3_14b_block7 |

Unmapped models retain Off and Auto. Explicit head preparation checks the pinned
target, immutable image, selected runtime, and mapped head config block_size=7.
It resolves the head commit from Hugging Face and saves it in the request.
No compatibility registry is required or read. Remove the obsolete
`INFIRO_DSPARK_COMPATIBILITY` environment variable from web and worker services.
Configuration checks do not establish GPU execution support; head status is
unknown until tested on the actual runtime. Preparation failures stop the run.

The local optimizer checkout at `bento_extend/llm-optimizer` and the vendored
915beb35 archive have identical `experiment/hf_models.py` and `remote/worker.py`
files. The integration uses their existing server argument rendering, remote
optimizer command, expanded profiles and Optuna execution. vLLM gets
`speculative_config` with method `dspark`, mapped repository, pinned revision and
seven speculative tokens. SGLang gets algorithm `DSPARK`, the mapped repository
and draft revision, eight draft tokens, one step and top-k one. Runtime loaders
fetch the pinned snapshots rather than requiring pre-baked draft paths.

Worker images must include DSpark architecture support for these Qwen heads,
not merely a parser accepting DSPARK. vLLM must support the speculative config
fields above; SGLang must support DSPARK and
`--speculative-draft-model-revision`. Keep the existing frozen-workload and
credential-safe worker overlay and the vendored optimizer version. Rebuild and
pin any image lacking these capabilities; changing the web configuration alone
does not add runtime support. No new image has been built or GPU-verified here.
A runtime rejection must fail the study, never retry with ordinary decoding.

Approval shows the resolved method and head, with the revision in configuration
details. Frozen workloads, Optuna, collection, request editing protections,
cost review and explicit launch approval remain in place. No GPU allocation
occurs during preparation.

CPU validation: `tests/test_dspark.py`, `tests/test_optimization_preparation.py`,
Django `infiro_platform.test_dspark` and `infiro_platform.test_study_inputs`, and
`web/tests/platform-state-browser.mjs`. These verify configuration and command
construction, not GPU support or speedup.
