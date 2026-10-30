"""Bounded loading around the optimizer's shared ShareGPT sampler."""

import json
import os
import re
from itertools import islice

from llm_optimizer.hf_workload import (
    ShareGPTSchemaError,
    fingerprint,
    first_human,
    prepare_hf_workload,
    validate_manifest,
)

from .limits import SHAREGPT_DOWNLOAD_BYTES, TOKENIZER_DOWNLOAD_BYTES
from .workload_cache import CACHE_FULL, CACHE_INVALID

MAX_DOWNLOAD = SHAREGPT_DOWNLOAD_BYTES
MAX_ROWS = 10000
MAX_SOURCE_TOKENS = 2_000_000
MAX_ROW_BYTES = 256 * 1024
DEFAULT_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DEFAULT_DATA_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


class WorkloadLimitError(ValueError):
    """Static, safe-to-display limit rejection from the web boundary."""


HF_LOADING_ADVICE = {
    401: "Authentication failed; check the HF read-token connection.",
    403: "Access denied; check token permissions and gated repository terms.",
    404: "Repository or revision not found, or not accessible with this token.",
    429: "Hugging Face rate limit reached; retry later with an HF read-token connection.",
    None: "Check worker network access, repository access and installed dependencies.",
}
HF_LOADING_ERRORS = frozenset(
    f"Custom HF {stage} loading or access failed. {advice}"
    for stage in ("dataset metadata", "tokenizer metadata", "tokenizer files")
    for advice in HF_LOADING_ADVICE.values()
)


def loading_error(stage, exc):
    """Describe the failed stage without exposing provider text or credentials."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    advice = HF_LOADING_ADVICE.get(status, HF_LOADING_ADVICE[None])
    return ValueError(f"Custom HF {stage} loading or access failed. {advice}")


FORMAT_ERROR = (
    "Unsupported or malformed dataset JSON. Convert to a top-level JSON array of "
    "objects or JSONL (one record per line), then shard into files with "
    "at most 10,000 rows and 256 KiB per row; select one JSON file and validate again."
)
MEMORY_ERROR = (
    "Preparation exceeded the worker memory limit. Shard ShareGPT into smaller JSON "
    "arrays or JSONL files, select one file, reduce prompt/token counts and validate again."
)


def iter_json_records(stream, *, skip_oversized=False):
    """Frame one JSON object at a time; never decode the enclosing document.

    Retained input is one 64 KiB chunk and at most one bounded source row.
    Both pretty-printed arrays and newline-delimited objects are supported.
    """
    def characters():
        while chunk := stream.read(64 * 1024):
            yield from chunk

    chars = iter(characters())

    def nonspace():
        return next((c for c in chars if c not in b" \r\n\t"), None)

    first = nonspace()
    array = first == ord("[")
    start = nonspace() if array else first
    if array and start == ord("]"):
        if nonspace() is not None:
            raise WorkloadLimitError(FORMAT_ERROR)
        return
    while start is not None:
        if start != ord("{"):
            raise WorkloadLimitError(FORMAT_ERROR)
        row = bytearray(b"{")
        stack, quoted, escaped, oversized = [123], False, False, False
        for c in chars:
            if not oversized:
                row.append(c)
                if len(row) > MAX_ROW_BYTES:
                    if not skip_oversized:
                        raise WorkloadLimitError("Dataset row exceeds the 256 KiB limit.")
                    oversized = True
                    row.clear()  # Drain the remainder without retaining its bytes.
            if quoted:
                if escaped:
                    escaped = False
                elif c == 92:
                    escaped = True
                elif c == 34:
                    quoted = False
            elif c == 34:
                quoted = True
            elif c in (123, 91):
                stack.append(c)
                if len(stack) > 256:
                    raise WorkloadLimitError(FORMAT_ERROR)
            elif c in (125, 93):
                expected = 123 if c == 125 else 91
                if not stack or stack.pop() != expected:
                    raise WorkloadLimitError(FORMAT_ERROR)
                if not stack:
                    break
        if stack:
            raise WorkloadLimitError(FORMAT_ERROR)
        if oversized:
            yield None  # Internal exclusion placeholder preserves source indices.
        else:
            try:
                record = json.loads(row)
            except (ValueError, UnicodeError, RecursionError):
                raise WorkloadLimitError(FORMAT_ERROR) from None
            yield record
        if array:
            separator = nonspace()
            if separator == ord("]"):
                if nonspace() is not None:
                    raise WorkloadLimitError(FORMAT_ERROR)
                return
            if separator != ord(","):
                raise WorkloadLimitError(FORMAT_ERROR)
            start = nonspace()
            if start is None:
                raise WorkloadLimitError(FORMAT_ERROR)
        else:
            # JSONL requires a newline between records, not concatenated JSON.
            newline = False
            for c in chars:
                if c not in b" \r\n\t":
                    break
                newline |= c == 10
            else:
                return
            if not newline:
                raise WorkloadLimitError(FORMAT_ERROR)
            start = c
    raise WorkloadLimitError(FORMAT_ERROR)


ADAPTER_VERSION = 1


def adapt_record(row, source_format, index):
    """Normalize supported text-only schemas before the shared sampler runs."""
    def invalid(detail):
        raise WorkloadLimitError(
            f"Malformed {source_format} record at row {index + 1}: {detail}"
        )

    if source_format == "sharegpt":
        return row
    if not isinstance(row, dict):
        invalid("expected an object.")
    if source_format == "alpaca":
        instruction = row.get("instruction")
        extra = row.get("input", "")
        if not isinstance(instruction, str) or not instruction.strip():
            invalid("instruction must be nonempty text.")
        if not isinstance(extra, str):
            invalid("input must be text when present.")
        prompt = instruction + ("\n\n" + extra if extra.strip() else "")
    elif source_format == "messages":
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            invalid("messages must be a nonempty list.")
        prompt = None
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get("role") not in ("system", "user", "assistant")
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()):
                invalid("each message needs a system/user/assistant role and nonempty text content.")
            if prompt is None and message["role"] == "user":
                prompt = message["content"]
        if prompt is None:
            invalid("messages must include a user message.")
    else:
        raise WorkloadLimitError("Unsupported dataset format; choose sharegpt, alpaca or messages.")
    return {"conversations": [{"from": "human", "value": prompt}]}


def materialize(config, *, api=None, loader=None, tokenizer=None, progress=None):
    source_format = config.get("format", "sharegpt")
    if source_format not in ("sharegpt", "alpaca", "messages"):
        raise WorkloadLimitError("Unsupported dataset format; choose sharegpt, alpaca or messages.")
    progress = progress or (lambda message: None)
    progress("Checking dataset access, pinned revisions and download sizes")
    from huggingface_hub import HfApi

    api = api or HfApi(token=os.getenv("HF_TOKEN") or False)
    # Preflight before any dataset or tokenizer file download. Unknown sizes fail
    # closed; large multi-file repositories are deliberately unsupported in v1.
    try:
        info = api.dataset_info(
            config["repo_id"], revision=config.get("revision"), files_metadata=True
        )
    except MemoryError:
        raise WorkloadLimitError(MEMORY_ERROR) from None
    except Exception as exc:
        raise loading_error("dataset metadata", exc) from None
    try:
        model = api.model_info(
            config["model"], revision=config.get("model_revision"), files_metadata=True
        )
    except MemoryError:
        raise WorkloadLimitError(MEMORY_ERROR) from None
    except Exception as exc:
        raise loading_error("tokenizer metadata", exc) from None
    if not info.sha or not model.sha:
        raise ValueError("Dataset and tokenizer commits must resolve before loading.")
    if config.get("data_file"):
        selected = [f for f in info.siblings if f.rfilename == config["data_file"]]
        if len(selected) != 1:
            raise ValueError("Selected JSON data file was not found at this dataset revision.")
        sizes = [selected[0].size]
    else:
        sizes = [
            f.size for f in info.siblings if not f.rfilename.endswith((".md", ".gitattributes"))
        ]
    if not sizes or any(size is None for size in sizes) or sum(sizes) > MAX_DOWNLOAD:
        raise ValueError(
            "Selected dataset files exceed the 1 GiB download limit, or sizes are unknown."
        )
    tokenizer_files = [
        f
        for f in model.siblings
        if "token" in f.rfilename
        or f.rfilename.endswith((".model", "vocab.json", "merges.txt", "config.json"))
    ]
    if (
        any(f.size is None for f in tokenizer_files)
        or sum(f.size for f in tokenizer_files) > TOKENIZER_DOWNLOAD_BYTES
    ):
        raise ValueError("Tokenizer exceeds the 128 MiB limit, or file sizes are unknown.")
    default_source = (
        source_format == "sharegpt"
        and config["repo_id"] == DEFAULT_DATASET
        and config.get("data_file") == DEFAULT_DATA_FILE
        and not config.get("config")
        and config.get("split", "train") == "train"
    )
    if loader is None:
        if (not config.get("data_file") or config.get("config")
                or config.get("split", "train") != "train"):
            raise WorkloadLimitError(
                "Select one JSON file; split/configuration filtering is unavailable."
            )
        from huggingface_hub import HfFileSystem

        def loader(repo, **kwargs):
            if not config.get("data_file"):
                raise WorkloadLimitError(FORMAT_ERROR)
            def remote_rows():
                # Prefetch a bounded MiB instead of one HTTP range per parser chunk.
                fs = HfFileSystem(token=False if default_source else os.getenv("HF_TOKEN") or False)
                path = f"datasets/{repo}@{info.sha}/{config['data_file']}"
                with fs.open(path, "rb", block_size=1024 * 1024,
                             cache_type="readahead") as stream:
                    for row in iter_json_records(stream, skip_oversized=default_source):
                        # Escaping Unicode/control characters can also expand a
                        # bounded raw row beyond the existing canonical-size cap.
                        if default_source and len(json.dumps(row).encode()) > MAX_ROW_BYTES:
                            row = None
                        yield row

            cache_root = os.getenv("INFIRO_SHAREGPT_CACHE")
            if default_source and cache_root:
                from .workload_cache import cached_prefix, read_cached_prefix

                path = cached_prefix(
                    cache_root,
                    {"repo_id": repo, "revision": info.sha,
                     "data_file": config["data_file"], "max_rows": MAX_ROWS,
                     "max_row_bytes": MAX_ROW_BYTES, "format_version": 2},
                    lambda: islice(remote_rows(), MAX_ROWS),
                    progress,
                )
                yield from read_cached_prefix(path, MAX_ROW_BYTES)
            else:
                yield from remote_rows()

    original_loader = loader
    rejected = []
    # The bundled unfiltered source contains malformed conversations. Keep the
    # same schema predicate, but make eligibility filtering explicit and hashed.
    source_indices = []
    sampling = {
        "policy": "default_sharegpt_valid_prefix_v2",
        "max_source_rows": MAX_ROWS,
        "source_rows_scanned": 0,
        "schema_rows_skipped": 0,
        "oversized_rows_skipped": 0,
    }

    def reject(message):
        rejected.append(message)
        raise WorkloadLimitError(message)

    def bounded_loader(*args, **kwargs):
        try:
            rows = original_loader(*args, **kwargs)
            if default_source:
                rows = islice(rows, MAX_ROWS)
            for index, row in enumerate(rows):
                if index >= MAX_ROWS:
                    reject("Dataset exceeds the 10,000 row scan limit; choose a smaller split.")
                oversized = (default_source and row is None) or (
                    len(json.dumps(row).encode()) > MAX_ROW_BYTES
                )
                if oversized and not default_source:
                    reject("Dataset row exceeds the 256 KiB limit.")
                if (index + 1) % 1000 == 0:
                    progress(f"Validating and tokenizing dataset source rows: {index + 1}")
                if default_source:
                    sampling["source_rows_scanned"] += 1
                    if oversized:
                        sampling["oversized_rows_skipped"] += 1
                        continue
                    try:
                        first_human(row)
                    except ShareGPTSchemaError:
                        sampling["schema_rows_skipped"] += 1
                        continue
                    source_indices.append(index)
                yield adapt_record(row, source_format, index)
        except WorkloadLimitError as exc:
            reject(str(exc))
        except MemoryError:
            reject(MEMORY_ERROR)
        except ValueError as exc:
            if str(exc) in {CACHE_FULL, CACHE_INVALID}:
                reject(str(exc))
            raise

    progress("Loading the pinned tokenizer")
    if tokenizer is None:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config["model"],
                revision=model.sha,
                trust_remote_code=False,
                token=os.getenv("HF_TOKEN") or False,
            )
        except MemoryError:
            raise WorkloadLimitError(MEMORY_ERROR) from None
        except Exception as exc:
            raise loading_error("tokenizer files", exc) from None
    original_tokenizer = tokenizer

    class BoundedTokenizer:
        total = 0

        def encode(self, text):
            try:
                tokens = original_tokenizer.encode(text)
            except MemoryError:
                reject(MEMORY_ERROR)
            self.total += len(tokens)
            if self.total > MAX_SOURCE_TOKENS:
                reject("Dataset exceeds the 2,000,000 source token scan limit.")
            return tokens

        def decode(self, tokens):
            try:
                return original_tokenizer.decode(tokens)
            except MemoryError:
                reject(MEMORY_ERROR)

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

    progress(f"Validating {source_format} records and freezing prompts")
    try:
        manifest = prepare()
        if default_source:
            manifest["row_indices"] = [source_indices[i] for i in manifest["row_indices"]]
            manifest["sampling"] = sampling
            progress(
                f"Validated {sampling['source_rows_scanned']} source rows; "
                f"excluded {sampling['schema_rows_skipped']} invalid and "
                f"{sampling['oversized_rows_skipped']} oversized conversations"
            )
        if source_format != "sharegpt":
            manifest["dataset"]["source_format"] = source_format
            manifest["dataset"]["adapter_version"] = ADAPTER_VERSION
            manifest["sha256"] = fingerprint(manifest)
        if config.get("data_file"):
            manifest["dataset"]["data_file"] = config["data_file"]
            manifest["sha256"] = fingerprint(manifest)
    except ValueError as exc:
        if rejected:
            raise WorkloadLimitError(rejected[0]) from None
        if re.fullmatch(r"Insufficient valid dataset rows: need \d+, found \d+", str(exc)):
            raise ValueError(
                "Dataset has too few valid prompts; reduce prompt count or choose another split."
            ) from None
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
