from __future__ import annotations

import json

from hf_model_watch.storage.schemas import (
    RunManifest,
    build_reproducibility_snapshot,
    initialize_run_artifacts,
    load_run_manifest,
)


def test_manifest_writes_valid_json(tmp_path) -> None:
    manifest = RunManifest(
        run_id="run_test",
        model_id="sshleifer/tiny-gpt2",
        revision="abc123",
        config_snapshot={"scan": {"max_models_to_scan": 1}},
    )

    run_dir = initialize_run_artifacts(tmp_path, manifest)
    manifest_path = run_dir / "run_manifest.json"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = load_run_manifest(manifest_path)

    assert data["run_id"] == "run_test"
    assert data["status"] == "started"
    assert data["completed_at"] is None
    assert loaded.model_id == "sshleifer/tiny-gpt2"
    assert (run_dir / "trace_events.jsonl").exists()
    assert (run_dir / "metadata" / "claims_vs_observed.json").exists()


def test_reproducibility_snapshot_contains_required_run_fields() -> None:
    snapshot = build_reproducibility_snapshot(
        "org/model",
        "abc123",
        {"finetune": {"seed": 42}},
        {"backend": "transformers", "dtype": "bf16", "quantization": None},
    )

    assert snapshot["model_id"] == "org/model"
    assert snapshot["revision"] == "abc123"
    assert snapshot["selected_backend"] == "transformers"
    assert snapshot["selected_dtype"] == "bf16"
    assert snapshot["random_seed"] == 42
    assert "torch" in snapshot["package_versions"]
