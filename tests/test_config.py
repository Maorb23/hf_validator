from __future__ import annotations

from hf_model_watch.config import load_config


def test_configs_load() -> None:
    config = load_config("config")

    assert config.scan.lookback_days == 2
    assert config.scan.max_models_to_scan == 200
    assert config.scan.allow_trust_remote_code is False
    assert config.serverless.gpu_type == "gpu-h100-sxm"
    assert config.storage["local_artifact_dir"] == "artifacts"
