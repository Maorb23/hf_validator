from unittest.mock import Mock

import pytest

from hf_model_watch.optimization.contracts import StudyContext
from hf_model_watch.optimization.timing import elapsed, framework_timings, trial_timings
from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter, _client_args


def test_trial_identity_timings_and_rejected_attempts_survive_results():
    context = StudyContext(
        model_id="org/model",
        revision="a" * 40,
        gpu_type="gpu-h100-sxm",
        concurrency=1,
        workload={"input_tokens": 16, "output_tokens": 8},
        objectives=["mean_ttft_ms"],
    )
    manifest = dict(
        run_id="run",
        framework="vllm",
        semantic_context=context.model_dump(),
        model=context.model_id,
        gpu_type=context.gpu_type,
        gpu_count=1,
        workload=_client_args(context),
        objectives=context.objectives,
        constraints=None,
        resolved_revision=context.revision,
        study_summary={
            "trial_states": {"0": "COMPLETE", "1": "FAIL", "2": "PRUNED"},
            "failed_trials": [{"number": 1, "reason": "startup failed"}],
        },
        test_results=[
            {
                "optimization": {"trial_number": 0},
                "config": {"batch": 4},
                "results": {"mean_ttft_ms": 10, "completed": 100},
                "metadata": {
                    "execution": {
                        "timings_s": {"launch_to_ready": 6.5, "benchmark": 2},
                        "phase_status": {"launch_to_ready": "complete", "benchmark": "complete"},
                    }
                },
            }
        ],
    )
    controller = Mock()
    controller.results.return_value = dict(
        run_id="run", state="completed", manifests={"vllm": manifest}
    )
    summary = OptimizationAdapter(controller, {}).results("run", context)
    assert [(t.trial_id, t.status) for t in summary.trials] == [
        ("0", "COMPLETE"),
        ("1", "FAIL"),
        ("2", "PRUNED"),
    ]
    assert summary.candidates[0].trial_id == "0"
    assert summary.candidates[0].timings == {
        "queue_wait_s": None,
        "startup_to_ready_s": 6.5,
        "benchmark_s": 2,
    }
    assert summary.trials[1].reason == "startup failed"
    assert summary.framework_timings["vllm"]["startup_to_ready_s"] is None
    assert framework_timings(summary.trials[:1])["benchmark_s"] == 2


@pytest.mark.parametrize("value", [None, -1, True, float("nan"), float("inf")])
def test_unknown_invalid_and_partial_timings_never_become_zero(value):
    assert trial_timings({"timings_s": {"benchmark": value}})["benchmark_s"] is None
    assert (
        trial_timings(
            {"timings_s": {"launch_to_ready": 5}, "phase_status": {"launch_to_ready": "partial"}}
        )["startup_to_ready_s"]
        is None
    )


def test_queue_elapsed_rejects_missing_reversed_and_naive_timestamps():
    assert elapsed("2026-10-26T10:00:00Z", "2026-10-26T10:00:02.500+00:00") == 2.5
    assert elapsed("2026-10-26T10:00:00Z", "2026-10-26T09:59:59Z") is None
    assert elapsed(None, None) is None
    assert elapsed("2026-10-26T10:00:00", "2026-10-26T10:00:02Z") is None
    assert trial_timings({"timings_s": {"benchmark": 0}})["benchmark_s"] == 0
