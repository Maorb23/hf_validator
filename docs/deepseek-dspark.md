# DeepSeek DSpark runs

Configure now separates the speculative method from the evidence policy. Auto
retains the existing research-driven behavior, including Off, Prefer and Require.
Selecting DeepSeek DSpark requires that method for every selected framework; it
never falls back to target-only decoding. Nemotron research is unchanged and
cannot substitute a third-party drafter for the selected DeepSeek checkpoint.

The integration adapts llm-optimizer's `experiment/hf_models.py` settings to its
remote request contract. The notebook helper itself requires CUDA and downloads
full snapshots, so it cannot run inside web preparation. Optuna, argument
rendering and expanded search profiles still run in llm-optimizer.

Official sources:

- https://github.com/deepseek-ai/DeepSpec
- https://huggingface.co/collections/deepseek-ai/deepspec
- https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py

## Operator setup

No shipped worker image is newly declared compatible by this change. Before
enabling a tuple, load its exact target and draft revisions in the immutable
worker image, run a real benchmark using the intended expanded search profile,
and retain its results and logs. Check that the intended draft architecture is
loaded and that completed requests are valid. CPU command construction tests or
an algorithm name in `--help` do not establish GPU execution support.

Configure `INFIRO_DSPARK_COMPATIBILITY` as a JSON array in both web and worker
services. The worker passes it to isolated preparation children. Each record is:

```json
{
  "target_model": "Qwen/Qwen3-4B",
  "target_revision": "<40-character target commit>",
  "draft_model": "deepseek-ai/dspark_qwen3_4b_block7",
  "draft_revision": "<40-character draft commit>",
  "runtime": "vllm",
  "serving_image": "<registry/image>@sha256:<64-character digest>",
  "verification_artifact": "s3://<bucket>/<GPU-verification-results>",
  "source_urls": ["https://github.com/deepseek-ai/DeepSpec"]
}
```

These are placeholders, not a verified compatibility record. The array defaults
to empty. The UI disables frameworks without an exact target revision/image
match, and the backend repeats the checks independently of the browser. Changing
an image or target commit requires new verification. Duplicate matching records
are rejected. No credentials belong in these records.

vLLM receives `method=dspark`, the DeepSeek draft repository and pinned revision,
and seven speculative tokens. Its installed version must support these fields.
SGLang receives `DSPARK`, an eight-token verification window, one step and top-k
one, matching the optimizer helper. SGLang records must additionally contain
`draft_path`, an absolute snapshot directory baked into that exact image with
the draft commit as a path component (for example
`/opt/drafts/snapshots/<40-character draft commit>`). This avoids downloading a
mutable draft revision at execution time. Verify this directory's provenance
when building the image.

Qwen3 4B/8B/14B and Gemma 4 12B IT have explicit DeepSeek draft mappings. A mapping
does not authorize execution. In particular, the sibling optimizer's Gemma
investigation documents that stock SGLang 0.5.22 lacks `Gemma4DSparkModel` even
after its parser repair. Do not register that combination without a working
architecture implementation and successful GPU validation. TensorRT-LLM is
disabled in this app's explicit DSpark path pending a verified remote adapter.

Approval displays the method, drafter revision, image, and verification artifact.
The request editor cannot remove or replace required speculative arguments.
No GPU job is launched during preparation.

## Local validation

Use the CPU environment and install the sibling llm-optimizer checkout without
its GPU dependencies, then run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_dspark.py tests/test_optimization_preparation.py tests/test_recommender.py tests/test_discovery.py
python web/manage.py test infiro_platform.test_dspark infiro_platform.test_study_inputs
cd web
npm run build
```

These checks cover preparation, remote argument serialization, rejection of
mismatched evidence, and existing Auto research behavior. They do not substitute
for the GPU verification required above. Use the same compatibility records,
dependency versions and image digests locally and on Railway; then verify preparation
on Railway before approving execution.
