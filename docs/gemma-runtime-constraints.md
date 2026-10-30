# Gemma 4 12B remote optimization

The exact target `google/gemma-4-12B-it` receives model-scoped constraints after
llm-optimizer generates its expanded profile. `merge_server_args` replaces
matching domains; shared `SPACES` and `EXPANDED_CHOICES` are never mutated.
Other model IDs retain their profiles. Nemotron research still runs in ordinary
preparation. Workload identity, token counts, seed and client concurrency do not
change. Existing prepared requests must be prepared again to receive these limits.

The sibling optimizer's lab notebook and `search_profiles.py` were inspected.
The checked-out notebook defines generic expanded domains; the Gemma-specific
values supplied with this task are applied here as remote overrides:

| Runtime | Constraints |
| --- | --- |
| vLLM | attention_backend=auto, kv_cache_dtype=auto |
| SGLang | attention_backend=triton, both CUDA graph backends disabled |
| TRT-LLM | native YAML attn_backend=FLASHINFER, enable_chunked_prefill=true |

SGLang additionally uses max_running_requests=[8,16] (1 for client concurrency 1),
mem_fraction_static=0.8, chunked_prefill_size=[1024,2048,4096], and context_length
=max(1024, input_tokens+output_tokens). The observed 96-slot SWA pool requested
53.47 GiB against 31.55 GiB available. Sixteen slots substantially reduce that
reservation; the workload-sized context bounds per-request cache growth and
smaller prefill chunks limit transient allocations. This is a conservative
starting domain, not a measured memory-fit guarantee. Client concurrency remains
unchanged: excess requests queue at the server. Large workloads or smaller GPUs
can still exhaust memory and need further tuning.

TRT configs retain the optimizer's coupled batch sizes, memory fractions,
scheduler and graph settings. Constraining content-addressed YAML avoids a flat
flag being ignored or leaving TRTLLM in generated files. Tests materialize those
files through `remote.worker.optimizer_command`, parse native arguments, and
read the resulting config. Upstream 1.3.0rc26 exposes `attn_backend` as an LLM API
field, merges serve YAML into LLM arguments, and dispatches FLASHINFER to
FlashInferAttention **only if FlashInfer is available**:

- [Serve configuration merge](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc26/tensorrt_llm/commands/serve.py)
- [LLM arguments](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc26/tensorrt_llm/llmapi/llm_args.py)
- [Attention dispatch and fallback](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc26/tensorrt_llm/_torch/attention_backend/utils.py)

The current worker images do not include a dispatch guard. Therefore the native
config selects FLASHINFER, but successful dispatch must be checked in runtime
logs: upstream can fall back to TRTLLM attention when FlashInfer is unavailable.
This change does not establish that FlashInfer supports all Gemma layers or
fixes the head-size-512 MMHA failure. Validate actual backend logs, weight
loading, warmup and a completed workload on the exact runtime image before
calling GPU support verified.

## Existing-image compatibility

The shipped optimizer archive and checksum are unchanged from the pre-Gemma
915beb35 snapshot used by the existing images. The active change is limited to
app-side profile constraints and request validation, using existing server args
and native config files. It requires no GPU worker image rebuild, new image
digest or runtime package injection. The sibling checkout is untouched.

The proposed readiness/cache/backend-guard changes are retained only as the
inactive [optimizer-readiness-assets.patch](patches/optimizer-readiness-assets.patch)
for future review. That patch is **not applied** to the shipped optimizer.
Generated requests omit its new fields even after contract serialization.
Execution configurations that supply `server_ready_timeout_seconds` or
`cache_pinned_model_assets` are rejected with an explicit explanation, and the
request editor cannot add them. Reprepare any draft produced by the earlier
worker-extension implementation; do not approve it for an existing image.

Readiness keeps the existing approximately 300-second polling behavior, separate
from framework job `timeout_minutes`. The vLLM API processor timeout observed
previously can still recur; increasing job timeout does not extend readiness.
No new asset prefetch step is installed. Existing HF tokenizer/processor caches
may help if already populated, but no persistent cache is provisioned or assumed.
Enabling a longer readiness window, proactive pinned-asset caching or the TRT
fallback guard later requires an explicit worker update and a new review.

Request editing validates the final Gemma domains, rejects missing constraints,
conflicting Optuna overrides and alternative config paths, and preserves native
TRT files. The three opt-in Qwen DSpark mappings remain unchanged. Gemma has no
mapped DSpark head and explicit selection remains rejected; Auto retains its
existing exact revision/image research checks. These constraints do not certify
any research-selected speculative combination.

Remaining GPU validation: vLLM processor readiness under the existing deadline,
SGLang peak KV/prefill memory at intended workload/concurrency, TRT FlashInfer
selection and head-size-512 warmup, and completed comparable results using the
same frozen workload. No paid GPU jobs were launched for this change.
