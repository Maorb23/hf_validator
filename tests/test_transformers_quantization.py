from __future__ import annotations

import sys
from types import SimpleNamespace

from hf_model_watch.runtime.backend_transformers import load_model


def test_load_model_applies_nf4_quantization(monkeypatch) -> None:
    captured = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    class FakeModel:
        def eval(self) -> None:
            captured["eval"] = True

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return FakeModel()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    load_model("org/model", revision="abc", load_in_4bit=True)

    kwargs = captured["kwargs"]
    quantization = kwargs["quantization_config"].kwargs
    assert kwargs["device_map"] == "auto"
    assert quantization["load_in_4bit"] is True
    assert quantization["bnb_4bit_quant_type"] == "nf4"
    assert quantization["bnb_4bit_use_double_quant"] is True
    assert captured["eval"] is True
