from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from hf_model_watch.orchestration.result_sync_flow import prefect_result_sync_flow

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Nebius result synchronization flow.")
    parser.add_argument(
        "--name",
        default=os.getenv(
            "HF_MODEL_WATCH_RESULT_SYNC_DEPLOYMENT",
            "hf-model-watch-result-sync-local",
        ),
    )
    parser.add_argument(
        "--cron",
        default=os.getenv("HF_MODEL_WATCH_RESULT_SYNC_CRON", "*/10 * * * *"),
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("HF_MODEL_WATCH_CONFIG_DIR", "config"),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("HF_MODEL_WATCH_RESULT_SYNC_CACHE", "outputs/synced-nebius"),
    )
    args = parser.parse_args()

    if not hasattr(prefect_result_sync_flow, "serve"):
        raise RuntimeError("Prefect is not installed")
    prefect_result_sync_flow.serve(
        name=args.name,
        cron=args.cron,
        parameters={
            "config_dir": args.config_dir,
            "cache_dir": args.cache_dir,
            "force": False,
        },
    )


if __name__ == "__main__":
    main()
