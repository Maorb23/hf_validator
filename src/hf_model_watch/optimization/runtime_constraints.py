"""Model-scoped runtime constraints, not claims of GPU-verified support."""

import ast
import hashlib
import json

GEMMA = "google/gemma-4-12B-it"


def overrides(framework, context):
    if context.model_id != GEMMA:
        return []
    domains = {
        "vllm": {"attention_backend": ["auto"], "kv_cache_dtype": ["auto"]},
        "sglang": {
            "attention_backend": ["triton"],
            "cuda_graph_backend_decode": ["disabled"],
            "cuda_graph_backend_prefill": ["disabled"],
            # 96 slots required 53.47 GiB of SWA KV against 31.55 GiB available.
            # Cap at 16 (about one sixth of that slot allocation), leaving room
            # for prefill and full-attention KV. This is not a fit guarantee.
            "max_running_requests": [1] if context.concurrency == 1 else [8, 16],
            "mem_fraction_static": [0.8],
            "chunked_prefill_size": [1024, 2048, 4096],
            "context_length": [
                max(1024, context.workload.input_tokens + context.workload.output_tokens)
            ],
        },
    }
    return [key + "=" + repr(values) for key, values in domains.get(framework, {}).items()]


def constrain_trt_files(args, files):
    """Keep optimizer-generated coupled capacities/configs and constrain YAML."""
    from llm_optimizer.search_profiles import CONFIG_PREFIX

    replacements, constrained = {}, {}
    for name, text in files.items():
        config = json.loads(text)
        config.update(attn_backend="FLASHINFER", enable_chunked_prefill=True)
        text = json.dumps(config, sort_keys=True)
        new_name = hashlib.sha256(text.encode()).hexdigest()[:16] + ".yaml"
        replacements[CONFIG_PREFIX + name] = CONFIG_PREFIX + new_name
        constrained[new_name] = text

    def replace(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, (tuple, list)):
            return [replace(v) for v in value]
        return value

    result = []
    for arg in args:
        if CONFIG_PREFIX in arg:
            key, raw = arg.split("=", 1)
            # Deduplicate configs made identical by the mandatory restrictions.
            values = replace(ast.literal_eval(raw))
            values = list({repr(v): v for v in values}.values())
            arg = key + "=" + repr(values)
        result.append(arg)
    return result, constrained


def validate_request(payload):
    """Reject removal, bypasses and conflicting tuned overrides for Gemma."""
    if payload["model"] != GEMMA:
        return
    from llm_optimizer.args import ArgScope, parse_args_str
    from llm_optimizer.predefined import SERVER_CONFIGS

    from hf_model_watch.optimization.contracts import StudyContext

    context = StudyContext.model_validate(payload["semantic_context"])
    for framework in payload["frameworks"]:
        base = payload["server_args"].get(framework, [])
        tuned = {
            **payload.get("optuna", {}),
            **payload.get("optuna_by_framework", {}).get(framework, {}),
        }.get("server_args", [])
        args = [*base, *tuned]
        required = dict(a.split("=", 1) for a in overrides(framework, context))
        found = {}
        for arg in args:
            for item in parse_args_str(arg, ArgScope.SERVER, SERVER_CONFIGS[framework], False):
                names = item.name if isinstance(item.name, tuple) else (item.name,)
                if set(names) & set(required):
                    if len(names) != 1 or names[0] in found:
                        raise ValueError(
                            "Gemma runtime constraints cannot be overridden or coupled."
                        )
                    found[names[0]] = item.values
        for name, values in required.items():
            allowed = ast.literal_eval(values)
            actual = found.get(name, [])
            if not actual or any(
                v not in allowed or type(v) is not type(allowed[0]) for v in actual
            ):
                raise ValueError("Gemma runtime constraint requires " + name + "=" + values)
        # Reject config-file escape hatches in frameworks constrained by flat flags.
        if framework in {"vllm", "sglang"}:
            from llm_optimizer.search_profiles import argument_names

            if any(
                argument_names(a)
                & {
                    "config",
                    "extra_llm_api_options",
                    "model_impl",
                    "hf_overrides",
                    "json_model_override_args",
                }
                for a in args
            ):
                raise ValueError(
                    "Gemma runtime constraints cannot be bypassed with configuration overrides."
                )
        if framework == "trtllm":
            for arg in args:
                for item in parse_args_str(arg, ArgScope.SERVER, SERVER_CONFIGS[framework], False):
                    if item.name == "backend" and item.values != ["pytorch"]:
                        raise ValueError("Gemma TRT requires the PyTorch backend.")
            from llm_optimizer.search_profiles import CONFIG_PREFIX, argument_names

            if any(
                argument_names(a)
                & {"attn_backend", "enable_chunked_prefill", "extra_llm_api_options"}
                for a in args
            ):
                raise ValueError("Gemma TRT settings must use the constrained native config.")
            configs = []
            for arg in args:
                if "config" not in argument_names(arg):
                    continue
                key, raw = arg.split("=", 1)
                names = key.split(":")[0].split("*")
                values = ast.literal_eval(raw)
                for value in values:
                    path = value[names.index("config")] if len(names) > 1 else value
                    if not isinstance(path, str) or not path.startswith(CONFIG_PREFIX):
                        raise ValueError("Gemma TRT requires a pinned native config.")
                    text = (
                        payload.get("server_config_files", {})
                        .get(framework, {})
                        .get(path[len(CONFIG_PREFIX) :])
                    )
                    config = json.loads(text or "{}")
                    if (
                        config.get("attn_backend") != "FLASHINFER"
                        or config.get("enable_chunked_prefill") is not True
                    ):
                        raise ValueError("Gemma TRT requires FLASHINFER and chunked prefill.")
                    configs.append(config)
            if not configs:
                raise ValueError("Gemma TRT native config is required.")
