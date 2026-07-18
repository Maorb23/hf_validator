from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from hf_model_watch.orchestration.prefect_flow import prefect_controller_flow

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the hf-model-watch Prefect controller.")
    parser.add_argument(
        "--name",
        default=os.getenv("HF_MODEL_WATCH_PREFECT_DEPLOYMENT", "hf-model-watch-controller-local"),
        help="Prefect deployment name.",
    )
    parser.add_argument(
        "--cron",
        default=os.getenv("HF_MODEL_WATCH_PREFECT_CRON", "0 7,19 * * *"),
        help="Cron schedule for controller runs.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("HF_MODEL_WATCH_CONFIG_DIR", "config"),
        help="Config directory passed to the controller flow.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=_bool_env("HF_MODEL_WATCH_PREFECT_DRY_RUN", True),
        help="Submit dry-run Nebius jobs from scheduled runs.",
    )
    parser.add_argument(
        "--real-run",
        action="store_false",
        dest="dry_run",
        help="Submit real Nebius jobs from scheduled runs.",
    )
    args = parser.parse_args()

    if not hasattr(prefect_controller_flow, "serve"):
        raise RuntimeError("Prefect is not installed; run `uv add prefect` first.")

    prefect_controller_flow.serve(
        name=args.name,
        cron=args.cron,
        parameters={"config_dir": args.config_dir, "dry_run": args.dry_run},
    )


if __name__ == "__main__":
    main()
