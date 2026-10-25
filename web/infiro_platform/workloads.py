"""Bounded loading around the optimizer's shared ShareGPT sampler."""

import json
import os

from llm_optimizer.hf_workload import prepare_hf_workload, validate_manifest

MAX_DOWNLOAD = 128 * 1024 * 1024
MAX_ROWS = 10000
MAX_SOURCE_TOKENS = 2_000_000
MAX_ROW_BYTES = 256 * 1024
DEFAULT_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"


class WorkloadLimitError(ValueError):
    """Static, safe-to-display limit rejection from the web boundary."""


def materialize(config, *, api=None, loader=None, tokenizer=None):
    from huggingface_hub import HfApi

    api = api or HfApi(token=os.getenv("HF_TOKEN") or False)
    # Preflight before any dataset or tokenizer file download. Unknown sizes fail
    # closed; large multi-file repositories are deliberately unsupported in v1.
    try:
        info = api.dataset_info(
            config["repo_id"], revision=config.get("revision"), files_metadata=True
        )
        model = api.model_info(
            config["model"], revision=config.get("model_revision"), files_metadata=True
        )
    except Exception:
        raise ValueError(
            "Custom HF dataset/tokenizer loading or access failed; check access and dependencies"
        ) from None
    if not info.sha or not model.sha:
        raise ValueError("Dataset and tokenizer commits must resolve before loading.")
    sizes = [f.size for f in info.siblings if not f.rfilename.endswith((".md", ".gitattributes"))]
    if not sizes or any(s is None for s in sizes) or sum(sizes) > MAX_DOWNLOAD:
        raise ValueError("Dataset exceeds the 128 MiB download limit, or file sizes are unknown.")
    tokenizer_files = [
        f
        for f in model.siblings
        if "token" in f.rfilename
        or f.rfilename.endswith((".model", "vocab.json", "merges.txt", "config.json"))
    ]
    if (
        any(f.size is None for f in tokenizer_files)
        or sum(f.size for f in tokenizer_files) > MAX_DOWNLOAD
    ):
        raise ValueError("Tokenizer exceeds the 128 MiB limit, or file sizes are unknown.")
    if loader is None:
        from datasets import DownloadConfig, load_dataset

        def loader(repo, **kwargs):
            return load_dataset(
                repo,
                **kwargs,
                streaming=True,
                download_config=DownloadConfig(max_retries=0),
                token=os.getenv("HF_TOKEN") or False,
            )

    original_loader = loader
    rejected = []

    def reject(message):
        rejected.append(message)
        raise WorkloadLimitError(message)

    def bounded_loader(*args, **kwargs):
        for index, row in enumerate(original_loader(*args, **kwargs)):
            if index >= MAX_ROWS:
                reject("Dataset exceeds the 10,000 row scan limit; choose a smaller split.")
            if len(json.dumps(row).encode()) > MAX_ROW_BYTES:
                reject("Dataset row exceeds the 256 KiB limit.")
            yield row

    if tokenizer is None:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config["model"],
                revision=model.sha,
                trust_remote_code=False,
                token=os.getenv("HF_TOKEN") or False,
            )
        except Exception:
            raise ValueError(
                "Custom HF dataset/tokenizer loading or access failed; "
                "check access and dependencies"
            ) from None
    original_tokenizer = tokenizer

    class BoundedTokenizer:
        total = 0

        def encode(self, text):
            tokens = original_tokenizer.encode(text)
            self.total += len(tokens)
            if self.total > MAX_SOURCE_TOKENS:
                reject("Dataset exceeds the 2,000,000 source token scan limit.")
            return tokens

        def decode(self, tokens):
            return original_tokenizer.decode(tokens)

    # Pinned revisions prevent a branch moving between preflight and sampling.
    def prepare():
        return prepare_hf_workload(
            {
                k: v
                for k, v in {
                    "repo_id": config["repo_id"],
                    "revision": info.sha,
                    "config": config.get("config"),
                    "split": config.get("split", "train"),
                }.items()
            },
            config["model"],
            model.sha,
            config["input_tokens"],
            config["output_tokens"],
            config["num_prompts"],
            config["seed"],
            api=api,
            loader=bounded_loader,
            tokenizer=BoundedTokenizer(),
        )

    try:
        manifest = prepare()
    except ValueError:
        if rejected:
            raise WorkloadLimitError(rejected[0]) from None
        raise
    if len(json.dumps(manifest).encode()) > 16 * 1024 * 1024:
        raise ValueError("Frozen workload exceeds the 16 MiB limit.")
    return validate_manifest(manifest)


def identity(manifest):
    return {k: v for k, v in manifest.items() if k != "requests"} | {
        "sample_count": len(manifest["requests"])
    }


def random_workload(config, *, tokenizer=None):
    """Freeze the optimizer's native random token sampler once on the CPU."""
    import hashlib
    import inspect
    from dataclasses import asdict

    import numpy as np
    from llm_optimizer.bench_client import sample_random_requests
    from llm_optimizer.hf_workload import fingerprint

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model"],
            revision=config["model_revision"],
            trust_remote_code=False,
            token=os.getenv("HF_TOKEN") or False,
        )
    # Child process only: no shared generator or server environment is changed.
    np.random.seed(config["seed"])
    rows = sample_random_requests(
        config["input_tokens"],
        config["output_tokens"],
        config["num_prompts"],
        1.0,
        tokenizer,
        "",
        random_sample=False,
    )
    generator_revision = hashlib.sha256(
        inspect.getsource(sample_random_requests).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset": {
            "repo_id": "infiro/random",
            "revision": generator_revision,
            "config": None,
            "split": "generated",
        },
        "tokenizer": config["model"],
        "tokenizer_revision": config["model_revision"],
        "seed": config["seed"],
        "row_indices": list(range(len(rows))),
        "length_policy": "optimizer_native_random_fixed_length",
        "requests": [asdict(row) for row in rows],
    }
    manifest["sha256"] = fingerprint(manifest)
    return validate_manifest(manifest)
