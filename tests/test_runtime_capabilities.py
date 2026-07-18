from __future__ import annotations

import sys
from types import SimpleNamespace

from hf_model_watch.runtime.capabilities import collect_gpu_capabilities


def test_collect_gpu_capabilities_reports_device(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(index: int) -> str:
            return "H100"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            return (9, 0)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.8"),
        cuda=FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(__version__="0.test"))
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(__version__="0.test"))

    result = collect_gpu_capabilities(require_gpu=True)

    assert result["cuda_available"] is True
    assert result["gpu_name"] == "H100"
    assert result["gpu_capability"] == "9.0"
