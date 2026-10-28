"""CPU-only verification executed inside the actual serving worker image."""


def verify_frozen_workload(payload):
    import json
    import shlex
    from dataclasses import asdict
    from pathlib import Path
    from tempfile import TemporaryDirectory

    manifest = payload.get("request", {}).get("workload_manifest")
    if manifest is None:
        return None
    try:
        from llm_optimizer.bench_client import get_dataset, prepare_workload_manifest
        from llm_optimizer.hf_workload import validate_manifest
        from llm_optimizer.predefined.client import CLIENT_CONFIGS
        from llm_optimizer.remote.worker import optimizer_command

        expected = validate_manifest(manifest)
        if "workload_path" not in CLIENT_CONFIGS:
            raise ValueError("Missing frozen workload argument")
        with TemporaryDirectory(prefix="frozen-workload-check-") as directory:
            output = Path(directory) / "results.json"
            path = output.parent / "custom-workload.json"
            path.write_text(json.dumps(expected), encoding="utf-8")
            command = optimizer_command(payload, output)
            arguments = [command[i + 1] for i, v in enumerate(command[:-1]) if v == "--client-args"]
            paths = [
                shlex.split(v.partition("=")[2])
                for v in arguments
                if v.startswith("workload_path=")
            ]
            if paths != [[str(path.resolve())]]:
                raise ValueError("Worker command does not use frozen workload")
            args = {"model": expected["tokenizer"], "workload_path": str(path)}
            actual, metadata = prepare_workload_manifest(args)
            if actual != expected or metadata.get("workload_sha256") != expected["sha256"]:
                raise ValueError("Client changed frozen workload identity")
            # Exercise the request sampler too: a loader-only probe would miss
            # an image that accepts the option but still samples random data.
            requests = [asdict(row) for row in get_dataset(args, None)]
            if requests != expected["requests"]:
                raise ValueError("Client changed frozen requests")
        return {
            "status": "verified",
            "sha256": expected["sha256"],
            "request_count": len(expected["requests"]),
            "framework": payload["framework"],
        }
    except Exception:
        raise ValueError(
            "Worker image cannot consume the approved frozen workload. "
            "Rebuild the image with the configured optimizer before retrying."
        ) from None
