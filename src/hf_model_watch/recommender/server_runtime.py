from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download

from hf_model_watch.runtime.backend_transformers import GenerationResult


def build_server_command(
    framework: str, model_path: str, port: int, gpu_count: int = 1
) -> list[str]:
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if framework == "vllm":
        return [
            "vllm",
            "serve",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(gpu_count),
        ]
    if framework == "sglang":
        return [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp-size",
            str(gpu_count),
        ]
    if framework == "trt-llm":
        return [
            "trtllm-serve",
            "serve",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp_size",
            str(gpu_count),
        ]
    raise ValueError(f"Unsupported server framework: {framework}")


class OpenAIServerSession:
    """Run a pinned local server and use its completions endpoint for fixture prompts."""

    def __init__(self, model_id: str, revision: str, framework: str, gpu_count: int = 1) -> None:
        executable = {"vllm": "vllm", "trt-llm": "trtllm-serve"}.get(framework)
        if executable and not shutil.which(executable):
            raise RuntimeError(f"{executable} is missing from the framework image")
        if framework == "sglang" and importlib.util.find_spec("sglang") is None:
            raise RuntimeError("sglang is missing from the framework image")
        model_path = snapshot_download(repo_id=model_id, revision=revision)
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process = subprocess.Popen(
            build_server_command(framework, model_path, self.port, gpu_count),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        self.served_model = model_path
        try:
            self._wait_ready(300)
        except Exception:
            self.close()
            raise

    def _wait_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Framework server exited before becoming ready")
            try:
                with urlopen(f"{self.base_url}/v1/models", timeout=2) as response:
                    models = json.load(response)
                choices = models.get("data") or []
                if choices:
                    self.served_model = choices[0].get("id") or self.served_model
                return
            except Exception:
                time.sleep(2)
        raise TimeoutError("Framework server did not become ready")

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        body = json.dumps(
            {
                "model": self.served_model,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0,
            }
        ).encode()
        request = Request(
            f"{self.base_url}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urlopen(request, timeout=180) as response:
            result: dict[str, Any] = json.load(response)
        elapsed = time.perf_counter() - started
        text = str(result["choices"][0].get("text") or "")
        usage = result.get("usage") or {}
        return GenerationResult(
            prompt,
            text,
            int(usage.get("prompt_tokens") or len(prompt.split())),
            int(usage.get("completion_tokens") or len(text.split())),
            elapsed,
        )

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_server_gpu_memory() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return max(float(line.strip()) for line in result.stdout.splitlines()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
