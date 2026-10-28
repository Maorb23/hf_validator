"""Container supervisor: migrate the mounted database, then run both CPU services."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP_TIMEOUT = 20


def stop(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + STOP_TIMEOUT
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.1)
    for process in processes:
        # Also kill surviving Gunicorn children in a process group whose leader exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise(migration, services):
    processes = []

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        process = subprocess.Popen(migration, cwd=ROOT, start_new_session=True)
        processes.append(process)
        if process.wait() != 0:
            return 1
        for command in services:
            processes.append(subprocess.Popen(command, cwd=ROOT, start_new_session=True))
        while True:
            if any(p.poll() is not None for p in processes[1:]):
                print("A web service exited; stopping the deployment for recovery.", flush=True)
                return 1
            time.sleep(0.2)
    finally:
        stop(processes)


def main():
    os.umask(0o077)
    # Refuse ephemeral state if Railway has not attached the configured volume.
    if not os.path.ismount("/data"):
        raise SystemExit("A persistent volume must be mounted at /data.")
    private = Path(os.environ["INFIRO_PRIVATE_ROOT"]).resolve()
    database = Path(os.environ["INFIRO_DATABASE"]).resolve()
    if not private.is_relative_to("/data") or not database.is_relative_to(private):
        raise SystemExit("Private root and database must be on the /data volume.")
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A deployment-wide lease also excludes concurrent migrations/HTTP writers.
    with (private / ".web-service.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another deployment is using this volume.") from None
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infiro_site.settings")
        import django
        django.setup()
        from infiro_platform.security import cipher
        cipher()  # Fail before migrations or serving if the persistent key is invalid.
        port = int(os.environ.get("PORT", "8000"))
        if not 1 <= port <= 65535:
            raise SystemExit("PORT must be a valid TCP port.")
        return supervise(
            [sys.executable, "web/manage.py", "migrate", "--noinput"],
            [
                [sys.executable, "-m", "gunicorn", "--chdir", "web",
                 "infiro_site.wsgi:application", "--bind", f"0.0.0.0:{port}",
                 "--workers", "1", "--threads", "4", "--timeout", "120",
                 "--graceful-timeout", "15", "--error-logfile", "-"],
                [sys.executable, "web/manage.py", "platform_worker"],
            ],
        )


if __name__ == "__main__":
    sys.exit(main())
