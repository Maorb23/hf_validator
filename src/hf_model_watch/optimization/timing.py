"""Execution timing evidence. Partial, nonfinite and missing phases are unknown."""

import math
from datetime import datetime

TIMINGS = ("queue_wait_s", "startup_to_ready_s", "benchmark_s")


def seconds(value):
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def elapsed(start, end):
    try:
        return seconds(
            (
                datetime.fromisoformat(end.replace("Z", "+00:00"))
                - datetime.fromisoformat(start.replace("Z", "+00:00"))
            ).total_seconds()
        )
    except (TypeError, ValueError, AttributeError):
        return None


def trial_timings(execution):
    phases = execution.get("phase_status", {})
    measured = execution.get("timings_s", {})

    def phase(name):
        # An interrupted startup never reached ready; a partial benchmark is not
        # a complete benchmark duration. Older records can omit phase_status.
        return seconds(measured.get(name)) if phases.get(name, "complete") == "complete" else None

    return {
        "queue_wait_s": seconds(measured.get("queue_wait")),
        "startup_to_ready_s": phase("launch_to_ready"),
        "benchmark_s": phase("benchmark"),
    }


def framework_timings(trials):
    # Totals describe all recorded attempts, never an invented average or GPU time.
    return {
        key: sum(t.timings[key] for t in trials)
        if trials and all(t.timings.get(key) is not None for t in trials)
        else None
        for key in TIMINGS
    }
