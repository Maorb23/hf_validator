"""Failure metadata without exception messages, locals, URLs, or credentials."""

import re
from pathlib import Path


def failure_details(error):
    chain = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen and len(chain) < 5:
        seen.add(id(current))
        frames = []
        tb = current.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            frames.append(
                {
                    "file": Path(frame.f_code.co_filename).name,
                    "function": frame.f_code.co_name,
                    "line": tb.tb_lineno,
                }
            )
            tb = tb.tb_next
        item = {
            "type": type(current).__name__,
            "module": type(current).__module__,
            "frames": frames[-8:],
        }
        code = getattr(getattr(getattr(current, "status", None), "code", None), "name", None)
        if isinstance(code, str) and re.fullmatch(r"[A-Z_]{1,40}", code):
            item["provider_status"] = code
        for key in ("request_id", "trace_id"):
            identifier = getattr(getattr(current, "status", None), key, None)
            if isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
                item[key] = identifier
        errors = []
        for error in getattr(getattr(current, "status", None), "service_errors", [])[:8]:
            entry = {}
            for key in ("code", "service"):
                value = getattr(error, key, "")
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value):
                    entry[key] = value
            details = getattr(error, "details", None)
            if getattr(details, "field", None) == "bad_request":
                entry["fields"] = [
                    v.field
                    for v in details.value.violations
                    if re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,100}", v.field)
                ][:8]
            errors.append(entry)
        if errors:
            item["provider_errors"] = errors
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            item["http_status"] = status
        chain.append(item)
        current = current.__cause__ or current.__context__
    return {"exceptions": chain}


def submission_validation_details(error, metadata):
    """Expose only redacted label validation when no user labels were sent."""
    from .services import redact

    result = {"submitted_metadata": dict(metadata)}
    if metadata.get("label_count") != 0:
        return result
    messages = []
    seen = set()
    while error is not None and id(error) not in seen and len(seen) < 5:
        seen.add(id(error))
        for item in getattr(getattr(error, "status", None), "service_errors", [])[:8]:
            details = getattr(item, "details", None)
            if getattr(details, "field", None) != "bad_request":
                continue
            for violation in details.value.violations[:8]:
                if violation.field != "metadata.labels":
                    continue
                # Do not expose arbitrary provider response bodies/messages.
                # This field contains no user data in an unlabelled request.
                message = redact(violation.message)
                message = re.sub(r"https?://[^\s]+", "[URL omitted]", message)
                message = re.sub(r"(?i)Bearer\s+[A-Za-z0-9_.=/-]+", "Bearer [redacted]", message)
                message = re.sub(
                    r"(?:eyJ|v1\.|t1\.)[A-Za-z0-9_.=-]{20,}", "[token redacted]", message
                )
                message = re.sub(
                    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
                    "[private key redacted]",
                    message,
                    flags=re.DOTALL,
                )
                message = message[:1000]
                if message not in messages:
                    messages.append(message)
        error = error.__cause__ or error.__context__
    if messages:
        result["label_validation_messages"] = messages
    return result


def failure_summary(details):
    names = [item["type"] for item in details["exceptions"]]
    if any(
        "metadata.labels" in error.get("fields", [])
        for item in details["exceptions"]
        for error in item.get("provider_errors", [])
    ):
        metadata = details.get("submitted_metadata", {})
        image_length = metadata.get("image_reference_length")
        if metadata.get("label_count") == 0 and isinstance(image_length, int) and any(
            re.search(
                rf"label value length \({image_length}\) exceeds maximum value \(64\)",
                message,
                re.IGNORECASE,
            )
            for message in details.get("label_validation_messages", [])
        ):
            return (
                "Nebius rejected an internal label whose length matches the container image "
                "reference; no custom labels were submitted. This suggests a Nebius Jobs-to-Compute "
                "validation issue. Contact Nebius support with the request/trace IDs and image "
                "reference length in Safe provider diagnostics. Keep the reviewed image digest pinned."
            )
        return "Nebius reported metadata.labels validation failure. Inspect the submitted metadata and redacted validation message in Safe provider diagnostics."
    if any(
        item["type"] == "AuthenticationError" and item.get("module", "").startswith("openai")
        for item in details["exceptions"]
    ):
        return (
            "Nebius inference authentication failed. Check the selected inference connection's "
            "API key and endpoint. This is separate from Nebius jobs and SecretStash authentication."
        )
    codes = {
        error.get("code")
        for item in details["exceptions"]
        for error in item.get("provider_errors", [])
    }
    if any(
        item.get("provider_status") == "PERMISSION_DENIED"
        and any(
            error.get("service") == "mysterybox-dpl" for error in item.get("provider_errors", [])
        )
        for item in details["exceptions"]
    ):
        return (
            "SecretStash denied access to a secret payload. Grant payload-read permission "
            "(mysterybox.payload-viewer) "
            "to the service account selected in the Nebius jobs connection."
        )
    if "JwtKeyNotExists" in codes:
        return (
            "Nebius cannot find the configured service-account public key (JwtKeyNotExists). "
            "Replace the Nebius jobs connection with the registered public-key ID and its matching private key."
        )
    if any(name in names for name in ("TimeoutError", "ConnectTimeout", "ReadTimeout")):
        return "Provider connection or authentication timed out. Check provider access and network connectivity."
    if "MemoryError" in names or "ArrowMemoryError" in names:
        return (
            "Preparation exceeded the worker memory limit. Review the tokenizer and workload size."
        )
    if any(name in names for name in ("ImportError", "ModuleNotFoundError")):
        return "A required worker dependency is missing. See the safe technical diagnostics."
    if any(name in names for name in ("GatedRepoError", "RepositoryNotFoundError")):
        return "Hugging Face access failed. Check the selected token and model/dataset permissions."
    name = names[0] if names else "Error"
    name = name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", name) else "Error"
    return f"Operation failed ({name}). See the safe technical diagnostics for the failing step."
