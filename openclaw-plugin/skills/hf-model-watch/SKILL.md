---
name: hf-model-watch
description: Recommend Hugging Face models through the restricted hf-model-watch tools.
---

# Hugging Face model recommendations

Use `recommend_models` when a user describes an inference workload. The returned run is the
canonical record. If its status is `needs_input`, ask the returned questions and send only those
answers to `provide_recommendation_input` for the same run.

Present at most the returned shortlist, in its existing order. Preserve evidence classifications,
unknowns, limitations, and blockers exactly. `insufficient_evidence` is a valid result: explain it
and offer to revise constraints. Never invent capabilities, candidates, measurements, eligibility,
or stronger evidence.

Call `select_model` only after the user explicitly names or confirms a listed candidate; pass
`confirmed: true`. Never interpret a preference, remembered detail, or earlier tentative statement
as selection authorization.

Use `record_model_feedback` for feedback about the selected model. Describe it as user-reported,
not measured or verified. For quality or capability feedback, ask whether the user wants a revised
search before calling `revise_recommendation`. For latency, throughput, or out-of-memory feedback,
record available workload and hardware context, then explain that optimization is not connected.
Do not automatically revise every complaint.

Use `get_run_status` after reconnecting or when the current run is uncertain. Runs and feedback are
canonical; conversation memory is only a convenience. Confirm recalled hardware, license, or budget
before using it because those fields affect cost or eligibility. Do not store secrets or unrelated
prompts as preferences.

There is no optimization or benchmark tool in Phase 2. The returned optimization action is a
non-executable placeholder.
