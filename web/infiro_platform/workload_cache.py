"""Persistent, owner-scoped cache of the public default source's bounded prefix."""

import fcntl
import hashlib
import json
import os
from pathlib import Path

from .limits import SHAREGPT_DOWNLOAD_BYTES

CACHE_FULL = (
    "ShareGPT source cache exceeds its 1 GiB limit. Ask the operator to remove unused "
    "ShareGPT cache revisions for this account, then prepare again."
)
CACHE_INVALID = (
    "ShareGPT source cache failed its integrity check. Ask the operator to remove "
    "the ShareGPT cache for this account, then prepare again."
)


def directory_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            # A cache entry may have just been atomically published.
            continue
    return total


def cached_prefix(root, identity, source, progress):
    """Publish only complete prefixes; never treat an interrupted download as ready.

    Source JSONL includes invalid conversations so original indices, exclusions
    and source-row limits remain the same on cache hits. Tokenization is not cached.
    """
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    data = root / f"{key}.jsonl"
    meta = root / f"{key}.json"
    partial = root / f"{key}.partial"
    with (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if meta.exists():
            try:
                if meta.stat().st_size > 4096 or data.stat().st_size > SHAREGPT_DOWNLOAD_BYTES:
                    raise ValueError(CACHE_INVALID)
                saved = json.loads(meta.read_text())
                checksum = hashlib.sha256()
                with data.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        checksum.update(chunk)
                if saved["identity"] != identity or saved["sha256"] != checksum.hexdigest():
                    raise ValueError(CACHE_INVALID)
            except (OSError, ValueError, KeyError):
                raise ValueError(CACHE_INVALID) from None
            progress("Reusing locally cached ShareGPT source records")
            return data
        progress("Downloading and saving the bounded ShareGPT source prefix (first use)")
        checksum = hashlib.sha256()
        # Reuse this revision's partial filename after interruption. Only a
        # completed prefix with matching metadata is ever served to a sampler.
        with partial.open("wb") as stream:
            existing = directory_bytes(root)
            written = 0
            for index, row in enumerate(source()):
                line = json.dumps({"record": row}).encode() + b"\n"
                written += len(line)
                if existing + written + 4096 > SHAREGPT_DOWNLOAD_BYTES:
                    raise ValueError(CACHE_FULL)
                stream.write(line)
                checksum.update(line)
                if (index + 1) % 1000 == 0:
                    progress(f"Saving ShareGPT source records: {index + 1}")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, data)
        # Metadata is the completion marker and must be published last.
        marker = root / f"{key}.metadata-partial"
        with marker.open("w") as stream:
            json.dump({"identity": identity, "sha256": checksum.hexdigest()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(marker, meta)
        return data


def read_cached_prefix(path, max_row_bytes):
    """Read bounded cache envelopes; None denotes a drained oversized source row."""
    with path.open("rb") as stream:
        while line := stream.readline(max_row_bytes + 1024):
            if not line.endswith(b"\n"):
                raise ValueError(CACHE_INVALID)
            try:
                envelope = json.loads(line)
                if not isinstance(envelope, dict) or set(envelope) != {"record"}:
                    raise ValueError(CACHE_INVALID)
                yield envelope["record"]
            except (ValueError, UnicodeError, RecursionError):
                raise ValueError(CACHE_INVALID) from None
