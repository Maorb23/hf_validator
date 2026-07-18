from __future__ import annotations

import argparse
import importlib.metadata
from typing import Any

from hf_model_watch.tracing import emit_trace_event

REQUIRED_PACKAGES = ("torch", "transformers", "accelerate", "peft", "bitsandbytes")
OPTIONAL_PACKAGES = ("datasets", "wandb", "sentencepiece", "protobuf")


def collect_gpu_capabilities(require_gpu: bool = False) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch

    cuda_available = bool(torch.cuda.is_available())
    if not torch.version.cuda:
        raise RuntimeError("installed Torch build has no CUDA support")
    if require_gpu and not cuda_available:
        raise RuntimeError("CUDA GPU is not available inside the container")
    metadata: dict[str, Any] = {
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "peft_version": peft.__version__,
        "bitsandbytes_version": bitsandbytes.__version__,
        "packages": {
            name: _package_version(name) for name in (*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES)
        },
    }
    if cuda_available:
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "gpu_count": int(torch.cuda.device_count()),
            }
        )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the worker GPU Python environment.")
    parser.add_argument("--build-check", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    try:
        metadata = collect_gpu_capabilities(require_gpu=args.require_gpu)
    except Exception as exc:
        emit_trace_event(
            "gpu_environment_check",
            "bootstrap",
            "failed",
            f"GPU Python environment check failed: {type(exc).__name__}: {exc}",
        )
        return 1
    emit_trace_event(
        "gpu_environment_check",
        "bootstrap",
        "completed",
        "GPU Python environment is ready",
        metadata={"build_check": args.build_check, **metadata},
    )
    return 0


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
