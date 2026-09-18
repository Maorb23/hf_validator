from __future__ import annotations

import json
import sys

from hf_model_watch.recommender import probe


def test_probe_stages_show_hf_scan_and_missing_tavily_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(probe, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        probe,
        "hf_lookup",
        lambda **_kwargs: [
            {
                "model_id": "org/x",
                "revision": "abc",
                "license": "apache-2.0",
                "estimated_params_billion": 1.7,
                "context_length": 2048,
                "tokenizer_config": {},
                "missing_fields": ["card_excerpt"],
            }
        ],
    )
    monkeypatch.setattr(
        sys, "argv", ["probe", "hf", "--model-id", "org/x", "--output-dir", str(tmp_path)]
    )
    probe.main()
    hf_result = json.loads(capsys.readouterr().out)
    assert hf_result["model_id"] == "org/x"
    assert hf_result["missing_fields"] == ["card_excerpt"]
    assert (tmp_path / "hf_record.json").exists()

    monkeypatch.setattr(sys, "argv", ["probe", "tavily", "--output-dir", str(tmp_path)])
    probe.main()
    tavily_result = json.loads(capsys.readouterr().out)
    assert tavily_result["status"] == "missing_api_key"
    assert json.loads((tmp_path / "tavily_evidence.json").read_text()) == []
