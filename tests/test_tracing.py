from __future__ import annotations

import json

from hf_model_watch.tracing import emit_trace_event


def test_emit_trace_event_outputs_valid_json_line(capsys) -> None:
    emit_trace_event(
        event_type="unit_test",
        stage="test",
        status="completed",
        message="trace emitted",
        metadata={"model_id": "sshleifer/tiny-gpt2"},
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith("TRACE_EVENT ")

    payload = json.loads(line.removeprefix("TRACE_EVENT "))
    assert payload["event_type"] == "unit_test"
    assert payload["stage"] == "test"
    assert payload["status"] == "completed"
    assert payload["message"] == "trace emitted"
    assert "timestamp" in payload
    assert payload["metadata"]["model_id"] == "sshleifer/tiny-gpt2"
