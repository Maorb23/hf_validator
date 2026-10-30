"""Internal child entry point. Only the worker supplies a database job UUID."""

import os
import resource
import sys
from datetime import timedelta

import django


def main():
    import ctypes
    import signal

    # Stop an in-flight child if its worker dies; a restarted worker can then
    # safely recover its lease without a second process racing the old one.
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise SystemExit("Parent-death isolation is unavailable.")
    if os.getppid() != int(os.environ["INFIRO_WORKER_PID"]):
        raise SystemExit("Parent worker already stopped.")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (280, 290))
    from .limits import DEFAULT_NETWORK_BYTES, SHAREGPT_DOWNLOAD_BYTES

    large_workload = os.environ.get("INFIRO_LARGE_WORKLOAD") == "1"
    file_limit = SHAREGPT_DOWNLOAD_BYTES if large_workload else DEFAULT_NETWORK_BYTES
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    django.setup()
    from django.utils import timezone

    from .models import Job
    from .security import install_network_guard
    from .services import perform

    job = Job.objects.get(pk=sys.argv[1], state="running")
    job.progress.pop("diagnostics", None)
    job.save(update_fields=["progress"])
    install_network_guard(SHAREGPT_DOWNLOAD_BYTES if large_workload else DEFAULT_NETWORK_BYTES)
    try:
        perform(job)
        # Keep a cancellation requested during a study tick for the next tick.
        job.error = ""
        job.attempts = 0
        job.available_at = timezone.now() + timedelta(seconds=3)
        job.lease_until = None
        job.save(
            update_fields=[
                "state",
                "error",
                "progress",
                "result",
                "controller_run",
                "attempts",
                "available_at",
                "lease_until",
                "updated_at",
            ]
        )
    except Exception as exc:
        import json

        from .diagnostics import failure_details, failure_summary

        details = failure_details(exc)
        # Provider exception strings can contain secrets, signed URLs or prompts.
        # Only static messages for errors controlled by our boundary are exposed.
        from hf_model_watch.optimization.dspark import DSparkConfigurationError

        from .workload_cache import CACHE_FULL, CACHE_INVALID
        from .workloads import FORMAT_ERROR, MEMORY_ERROR

        safe = {
            "Model access or revision lookup failed. Check the model ID, "
            "revision and Hugging Face connection.",
            "The model lookup did not return a pinned commit.",
            FORMAT_ERROR, MEMORY_ERROR, CACHE_FULL, CACHE_INVALID,
            "Selected dataset files exceed the 1 GiB download limit, or sizes are unknown.",
            "Tokenizer exceeds the 128 MiB limit, or file sizes are unknown.",
            "Workload settings changed; validate a new dataset sample first.",
            "Dataset tokenizer differs from the selected pinned model. Validate a new sample.",
            "Selection changed. Prepare a new configuration.",
            "Expected nonempty conversations [{from, value}]",
            "Invalid ShareGPT role or empty text",
            "ShareGPT conversation has no human prompt",
            "Custom HF dataset/tokenizer loading or access failed; check access and dependencies",
            "Custom HF dataset iteration or tokenization failed",
            "Dataset exceeds the 10,000 row scan limit; choose a smaller split.",
            "Dataset row exceeds the 256 KiB limit.",
            "Dataset exceeds the 2,000,000 source token scan limit.",
            "Frozen workload exceeds the 16 MiB limit.",
            "Dataset has too few valid prompts; reduce prompt count or choose another split.",
            "Selected JSON data file was not found at this dataset revision.",
            "Download limit exceeded (1 GiB per workload operation).",
        }
        message = (
            str(exc)
            if (
                isinstance(exc, DSparkConfigurationError)
                or isinstance(exc, ValueError) and str(exc) in safe
            )
            else failure_summary(details)
        )
        job.progress["diagnostics"] = details
        Job.objects.filter(pk=job.pk).update(error=message, progress=job.progress)
        print(
            json.dumps({"job": str(job.id), "kind": job.kind, "failure": details}),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
