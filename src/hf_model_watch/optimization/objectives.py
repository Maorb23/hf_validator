"""Display metadata; native optimizer validation runs at the execution boundary."""

from itertools import combinations

from .contracts import DIRECTIONS

LABELS = {
    "output_throughput": "Output throughput (tokens/second)",
    "p95_e2e_latency_ms": "95th-percentile end-to-end latency (ms)",
    "p95_ttft_ms": "95th-percentile time to first token (ms)",
}


def objective_options() -> list[dict]:
    options = []
    for size in range(1, len(LABELS) + 1):
        for metrics in combinations(LABELS, size):
            value = ",".join(metrics)
            names = list(metrics)
            options.append({
                "value": value,
                "label": ("Pareto · " if size > 1 else "")
                + " + ".join(LABELS[name] for name in names),
                "directions": [DIRECTIONS[name] for name in names],
            })
    return options
