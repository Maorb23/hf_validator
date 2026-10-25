"""Presentation of execution outcomes, independent of controller lifecycle state."""


def trial_failures(result):
    summary = result.get("summary", {})
    failures = []
    errors = summary.get("framework_errors", {})
    for framework in sorted(set(summary.get("trial_state_counts", {})) | set(errors)):
        counts = summary.get("trial_state_counts", {}).get(framework, {})
        failed = counts.get("FAIL", 0)
        if failed or framework in errors:
            failures.append(
                {
                    "framework": framework,
                    "failed_trials": failed,
                    "completed_trials": counts.get("COMPLETE", 0),
                    "reason": errors.get(framework)
                    or (
                        f"{failed} benchmark trial(s) failed validation or execution. "
                        "Open the framework log for the recorded reason."
                    ),
                }
            )
    return failures


def study_outcome(job):
    if job.kind != "study" or job.state != "completed":
        return job.state
    candidates = job.result.get("summary", {}).get("candidates", [])
    if trial_failures(job.result):
        return "partial_success" if candidates else "no_valid_results"
    return job.state


def display_progress(job):
    progress = dict(job.progress)
    if job.kind == "study" and job.state in {"completed", "dry_run"}:
        # Legacy runs can retain a transient provider error after recovery.
        # Keep diagnostics as history, but never present it as a current error.
        error = progress.pop("provider_error", None)
        details = progress.pop("provider_diagnostics", None)
        if error or details:
            progress["recovered_provider_error"] = {"message": error, "details": details}
    return progress
