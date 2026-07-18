from __future__ import annotations

import json

import pytest

from hf_model_watch.orchestration.job_payloads import WorkerJobPayload, load_worker_payload


def test_worker_payload_validates(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "model_id": "sshleifer/tiny-gpt2",
                "revision": "abc123",
                "config": {"worker": {"dry_run": True}},
                "artifact_prefix": "hf-model-watch",
                "skip_finetune": True,
                "finetune_smoke_only": False,
            }
        ),
        encoding="utf-8",
    )

    payload = load_worker_payload(path)

    assert payload.run_id == "run_1"
    assert payload.model_id == "sshleifer/tiny-gpt2"
    assert payload.skip_finetune is True


def test_worker_payload_missing_required_field_fails_cleanly(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"run_id": "run_1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid worker payload"):
        load_worker_payload(path)


def test_worker_payload_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        WorkerJobPayload.model_validate(
            {"run_id": "run_1", "model_id": "org/model", "surprise": True}
        )
