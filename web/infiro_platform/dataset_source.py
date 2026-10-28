"""Validate the public dataset boundary without importing worker dependencies."""

import re
from urllib.parse import urlsplit


def dataset_id(value):
    if value.startswith("https://"):
        url = urlsplit(value)
        if url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ValueError("Use a credential-free Hugging Face dataset repository URL.")
        parts = url.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "datasets":
            raise ValueError("Use the dataset repository URL and select its JSON file separately.")
        value = "/".join(parts[1:])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value) or ".." in value:
        raise ValueError("Use a Hugging Face dataset repository ID or repository URL.")
    return value


def require_optimizer():
    # Called before queueing, not on workspace bootstrap or schema imports.
    from llm_optimizer.hf_workload import validate_manifest

    return validate_manifest
