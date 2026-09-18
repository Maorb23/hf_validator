from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from hf_model_watch.runtime.backend_transformers import (
    deterministic_generate,
    score_reference_completion,
)


class _CharacterTokenizer:
    eos_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_tensors: str,
    ) -> dict:
        values = ([1] if add_special_tokens else []) + [ord(character) for character in text]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, values, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "".join(chr(int(value)) for value in values)


class _ReferenceModel:
    config = SimpleNamespace(max_position_embeddings=64)

    def __init__(self, loss: float) -> None:
        self.loss = loss
        self.embedding = torch.nn.Embedding(256, 4)

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, *, input_ids, labels):
        assert input_ids.shape == labels.shape
        assert labels[0, :-1].eq(-100).all()
        return SimpleNamespace(loss=torch.tensor(self.loss))


def test_reference_loss_reports_cross_entropy_and_bits_per_byte() -> None:
    result = score_reference_completion(
        _ReferenceModel(math.log(2)),
        _CharacterTokenizer(),
        "prompt\n",
        "x",
    )

    assert result["reference_cross_entropy"] == pytest.approx(math.log(2))
    assert result["reference_perplexity"] == pytest.approx(2)
    assert result["reference_negative_log_likelihood"] == pytest.approx(math.log(2))
    assert result["reference_bits_per_byte"] == pytest.approx(1)
    assert result["reference_completion_tokens"] == 1
    assert result["reference_utf8_bytes"] == 1


class _GenerationModel:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.parameter

    def generate(self, input_ids, **kwargs):
        suffix = torch.tensor([[ord("o"), ord("k")]], dtype=input_ids.dtype)
        return torch.cat((input_ids, suffix), dim=1)


def test_generation_decodes_only_the_new_tokens(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = deterministic_generate(
        _GenerationModel(),
        _CharacterTokenizer(),
        prompt="prompt",
        max_new_tokens=2,
    )

    assert result.text == "ok"
    assert result.generated_tokens == 2
