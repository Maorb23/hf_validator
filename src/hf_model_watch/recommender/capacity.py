"""Transparent planning estimates, not measured peak GPU memory."""
def estimate_memory(weights_gb, use_draft=False, draft_weights_gb=None):
    if weights_gb is None:
        return None, {"basis": "unknown target weights"}
    weights = float(weights_gb)
    draft = (float(draft_weights_gb) if draft_weights_gb is not None else weights * .25) if use_draft else 0.
    overhead = max(1., weights * .10)
    return weights + draft + overhead, {
        "target_weights_gb": weights, "draft_weights_gb": draft,
        "draft_basis": "documented weights" if use_draft and draft_weights_gb is not None else "25% target fallback" if use_draft else "no draft",
        "runtime_kv_reserve_gb": overhead,
        "basis": "target + draft + max(1 GB, 10% target) runtime/KV reserve; context and concurrency unmeasured",
    }
