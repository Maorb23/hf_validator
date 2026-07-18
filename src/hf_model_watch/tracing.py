from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def emit_trace_event(
    event_type: str,
    stage: str,
    status: str,
    message: str,
    metadata: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> None:
    """Emit one structured TRACE_EVENT line to stdout."""
    event: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "stage": stage,
        "status": status,
        "message": message,
    }
    if metadata is not None:
        event["metadata"] = metadata
    if metrics is not None:
        event["metrics"] = metrics
    if artifacts is not None:
        event["artifacts"] = artifacts

    print("TRACE_EVENT " + json.dumps(event, sort_keys=True, separators=(",", ":")))
