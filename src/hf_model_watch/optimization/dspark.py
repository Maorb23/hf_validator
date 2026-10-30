"""Deterministic explicit DSpark settings; configuration is not GPU verification."""
import json
import re

DRAFTS = {
    "Qwen/Qwen3-4B": "deepseek-ai/dspark_qwen3_4b_block7",
    "Qwen/Qwen3-8B": "deepseek-ai/dspark_qwen3_8b_block7",
    "Qwen/Qwen3-14B": "deepseek-ai/dspark_qwen3_14b_block7",
}

class DSparkConfigurationError(ValueError):
    """Static configuration errors."""

def capabilities(images):
    return {"drafts": DRAFTS, "runtimes": [f for f in ("vllm", "sglang") if images.get(f)]}

def resolve(model, revision, frameworks, images, draft_revision=None):
    if model not in DRAFTS:
        raise DSparkConfigurationError("No mapped speculative decoding head for this target model.")
    if not frameworks or any(f not in {"vllm", "sglang"} for f in frameworks):
        raise DSparkConfigurationError("DSpark requires vLLM or SGLang; deselect TensorRT-LLM.")
    if not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        raise DSparkConfigurationError("DSpark requires a pinned target revision.")
    arguments, heads = {}, []
    for runtime in frameworks:
        image = images.get(runtime, "")
        if not re.fullmatch(r"\S+@sha256:[a-f0-9]{64}", image):
            raise DSparkConfigurationError("DSpark requires an immutable worker image.")
        if runtime == "vllm":
            config = {"method": "dspark", "model": DRAFTS[model], "num_speculative_tokens": 7}
            if draft_revision:
                config["revision"] = draft_revision
            arguments[runtime] = ["speculative_config=" + repr([json.dumps(config)])]
        else:
            arguments[runtime] = [
                "speculative_algorithm=['DSPARK']",
                "speculative_draft_model_path=" + repr([DRAFTS[model]]),
                "speculative_num_draft_tokens=[8]", "speculative_num_steps=[1]",
                "speculative_eagle_topk=[1]",
            ]
            if draft_revision:
                arguments[runtime].append("speculative_draft_model_revision=" + repr([draft_revision]))
        heads.append(dict(target_model=model, target_revision=revision, draft_model=DRAFTS[model],
            draft_revision=draft_revision, runtime=runtime, serving_image=image, method="dspark",
            status="unknown", block_size=7, execution_status="configured; GPU support untested"))
    return arguments, heads

def pinned_draft(model):
    """Resolve the fixed repository and check its block size without downloading weights."""
    from huggingface_hub import HfApi, hf_hub_download
    draft = DRAFTS[model]
    revision = HfApi().model_info(draft).sha
    if not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        raise DSparkConfigurationError("Could not pin the mapped head revision.")
    from pathlib import Path
    config = json.loads(Path(hf_hub_download(draft, 'config.json', revision=revision)).read_text())
    if config.get('block_size') != 7:
        raise DSparkConfigurationError("Mapped DSpark head requires block_size 7.")
    return revision
