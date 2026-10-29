"""Private content-addressed payload transport for existing optimizer images."""

import hashlib
import inspect
import json
import shlex

from .security import install_network_guard
from .workload_runtime import verify_frozen_workload

MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
REFERENCE_ENV = "INFIRO_WORKER_PAYLOAD_REF"


def payload_blob(payload, prepared, job):
    from llm_optimizer.remote.contracts import OptimizationRequest

    from .services import redact

    approved = OptimizationRequest.parse(prepared.payload).to_dict()
    if OptimizationRequest.parse(payload["request"]).to_dict() != approved:
        raise ValueError("Worker payload differs from the approved configuration.")
    if (
        payload["run_id"] != job.controller_run
        or payload["framework"] not in prepared.choices["frameworks"]
    ):
        raise ValueError("Worker payload is outside this run's authorized scope.")
    request = {
        k: v
        for k, v in approved.items()
        if k not in {"preemptible", "spot_pricing_policy_id", "follows_spot_price"}
    }
    worker_payload = {
        "run_id": payload["run_id"],
        "framework": payload["framework"],
        "request": request,
    }
    if redact(worker_payload) != worker_payload:
        raise ValueError("Worker payload contains a configured credential.")
    raw = json.dumps(
        worker_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise ValueError("Worker payload exceeds the 16 MiB transport limit.")
    sha = hashlib.sha256(raw).hexdigest()
    prefix = approved["prefix"].strip("/")
    if prefix != f"infiro/users/{job.owner_id}/prepared/{prepared.id}":
        raise ValueError("Worker payload storage is outside the user's authorized prefix.")
    name = f"launch-{sha}.json"
    reference = {
        "schema_version": 1,
        "bucket": approved["bucket"],
        "prefix": prefix,
        "key": f"{prefix}/{payload['run_id']}/{payload['framework']}/{name}",
        "sha256": sha,
        "size_bytes": len(raw),
        "run_id": payload["run_id"],
        "framework": payload["framework"],
    }
    return raw, reference


def load_and_run():
    """Self-contained trusted bootstrap embedded in JobSpec; never source user code."""
    import hashlib
    import json
    import os
    import re
    import socket
    import ssl

    import boto3
    from botocore.config import Config

    encoded_reference = os.environ["INFIRO_WORKER_PAYLOAD_REF"]
    if len(encoded_reference) > 4096:
        raise ValueError("Worker payload reference exceeds 4096 characters.")
    reference = json.loads(encoded_reference)
    if not isinstance(reference, dict):
        raise ValueError("Invalid worker payload reference.")
    limit = 16 * 1024 * 1024
    sha = reference.get("sha256", "")
    prefix = reference.get("prefix", "")
    run_id = reference.get("run_id", "")
    framework = reference.get("framework", "")
    size = reference.get("size_bytes")
    if (
        reference.get("schema_version") != 1
        or not re.fullmatch(r"[a-f0-9]{64}", sha)
        or not re.fullmatch(r"infiro/users/[1-9][0-9]*/prepared/[a-zA-Z0-9-]+", prefix)
        or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run_id)
        or framework not in {"vllm", "sglang", "trtllm"}
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 0 < size <= limit
        or reference.get("key") != f"{prefix}/{run_id}/{framework}/launch-{sha}.json"
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", reference.get("bucket", ""))
    ):
        raise ValueError("Invalid worker payload reference.")
    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.environ["AWS_DEFAULT_REGION"],
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
        config=Config(
            connect_timeout=10,
            read_timeout=20,
            retries={"max_attempts": 1},
            s3={"addressing_style": "path"},
        ),
    )
    # Apply the server's shared peer/redirect policy during payload retrieval.
    # Restore it before the normal worker downloads its (potentially large) model.
    originals = [
        (socket.socket, name, getattr(socket.socket, name))
        for name in ("connect", "connect_ex", "recv", "recv_into")
    ]
    originals.append((ssl.SSLSocket, "read", ssl.SSLSocket.read))
    install_network_guard(max_bytes=20 * 1024 * 1024)
    try:
        response = client.get_object(Bucket=reference["bucket"], Key=reference["key"])
        body = response["Body"]
        try:
            if response.get("ContentLength") != size:
                raise ValueError("Worker payload size differs from its approved reference.")
            raw = body.read(size + 1)
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Worker payload failed its integrity check.")
        finally:
            body.close()
    finally:
        try:
            client.close()
        finally:
            for cls, name, method in originals:
                setattr(cls, name, method)
    payload = json.loads(raw)
    request = payload.get("request", {})
    if (
        payload.get("run_id") != run_id
        or payload.get("framework") != framework
        or request.get("bucket") != reference["bucket"]
        or request.get("prefix", "").strip("/") != prefix
    ):
        raise ValueError("Downloaded payload belongs to a different run.")
    verification = verify_frozen_workload(payload)
    if verification is not None:
        print(json.dumps({"frozen_workload_verification": verification}), flush=True)
    from llm_optimizer.remote.worker import run_worker

    result = run_worker(payload)
    return 0 if result["status"] == "completed" else 1


def bootstrap_source():
    from .security import install_network_guard, public_ip

    return (
        "import socket, ipaddress\n"
        + inspect.getsource(public_ip)
        + "\n"
        + inspect.getsource(install_network_guard)
        + "\n"
        + inspect.getsource(verify_frozen_workload)
        + "\n"
        + inspect.getsource(load_and_run)
        + '\nif __name__ == "__main__":\n'
        + "    import sys\n    try:\n        code = load_and_run()\n"
        + '    except Exception as error:\n'
        + '        import json, traceback\n'
        + '        frames = traceback.extract_tb(error.__traceback__)\n'
        + '        details = {"type": type(error).__name__, "module": type(error).__module__,\n'
        + '                   "frames": [{"file": f.filename.rsplit("/", 1)[-1],\n'
        + '                               "function": f.name, "line": f.lineno}\n'
        + '                              for f in frames[-8:]]}\n'
        + '        print("Worker payload bootstrap failed; no benchmark started.",\n'
        + '              file=sys.stderr)\n'
        + '        if isinstance(error, ValueError) and str(error) == (\n'
        + '            "Worker image cannot consume the approved frozen workload. "\n'
        + '            "Rebuild the image with the configured optimizer before retrying."\n'
        + '        ):\n'
        + '            print(str(error), file=sys.stderr)\n'
        + '        print(json.dumps({"bootstrap_diagnostics": details}), file=sys.stderr)\n'
        + '        code = 2\n'
        + "    sys.exit(code)\n"
    )


def bootstrap_args():
    # Only trusted source is placed in command arguments. All run-specific data
    # lives in the small reference environment variable, with no shell involved.
    return "-c " + shlex.quote(bootstrap_source())
