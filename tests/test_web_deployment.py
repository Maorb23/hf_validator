"""Offline deployment checks; no provider requests and no GPU jobs."""
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def wait_file(path, process):
    deadline = time.monotonic() + 15
    while not path.exists():
        assert process.poll() is None, "Supervisor exited before services started"
        assert time.monotonic() < deadline, "Service startup timed out"
        time.sleep(0.05)


@pytest.mark.parametrize("failure", ["migration", "service", "signal"])
def test_supervisor_orders_startup_and_stops_peers(tmp_path, failure):
    migrated = tmp_path / "migrated"
    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    migration = [sys.executable, "-c", (
        "from pathlib import Path; import sys; "
        f"Path({str(migrated)!r}).touch(); sys.exit({1 if failure == 'migration' else 0})"
    )]
    worker = [sys.executable, "-c", f'''
from pathlib import Path
import signal, time
assert Path({str(migrated)!r}).exists()
def stop(*args):
    Path({str(stopped)!r}).touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path({str(ready)!r}).touch()
while True: time.sleep(.05)
''']
    peer = [sys.executable, "-c", (
        "import time; time.sleep(1); "
        + ("raise SystemExit(0)" if failure == "service" else "time.sleep(60)")
    )]
    script = ("from deploy.web_start import supervise; "
              f"raise SystemExit(supervise({migration!r}, {[worker, peer]!r}))")
    process = subprocess.Popen([sys.executable, "-c", script], cwd=ROOT)
    try:
        if failure != "migration":
            wait_file(ready, process)
        if failure == "signal":
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=25) == (0 if failure == "signal" else 1)
        assert stopped.exists() == (failure != "migration")
        if failure == "migration":
            assert not ready.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_database_artifacts_and_credentials_survive_fresh_processes(tmp_path):
    from cryptography.fernet import Fernet

    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'web'}",
           "DJANGO_SETTINGS_MODULE": "infiro_site.settings",
           "INFIRO_ENV": "production", "DJANGO_SECRET_KEY": "deployment-test-secret-" * 4,
           "DJANGO_ALLOWED_HOSTS": "testserver,healthcheck.railway.app",
           "DJANGO_TRUST_PROXY_SSL_HEADER": "1",
           "INFIRO_PRIVATE_ROOT": str(tmp_path),
           "INFIRO_DATABASE": str(tmp_path / "platform.sqlite3"),
           "INFIRO_CREDENTIAL_KEYS": Fernet.generate_key().decode(),
           "INFIRO_LIVE_ENABLED": "0"}
    setup = '''
import django
django.setup()
from django.core.management import call_command
call_command('migrate', interactive=False, verbosity=0)
from django.contrib.auth import get_user_model
from infiro_platform.models import Connection
from infiro_platform.security import seal
from infiro_platform.demo import seed_prepared
from infiro_platform.services import approve, perform_study
user = get_user_model().objects.create_user('deployment', password='offline-test-password')
c = Connection(owner=user, kind='huggingface', name='offline')
c.ciphertext = seal(c, {'token': 'offline-credential'})
c.save()
p = seed_prepared(user)
j = approve(user, p.id, p.digest, True)
perform_study(j)
j.save()
assert j.state == 'dry_run'
'''
    verify = '''
import django
django.setup()
import hashlib
from django.conf import settings
from django.test import Client
from infiro_platform.models import Connection, Dataset, Prepared, Job, Artifact
from infiro_platform.security import unseal, private_dir
assert unseal(Connection.objects.get()) == {'token': 'offline-credential'}
assert Dataset.objects.get().status == 'ready'
assert Prepared.objects.get().status == 'prepared'
assert Job.objects.get(kind='study').state == 'dry_run'
for artifact in Artifact.objects.all():
    path = private_dir(artifact.owner_id, artifact.job_id) / artifact.relative_path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256
client = Client()
assert client.get('/healthz', HTTP_HOST='healthcheck.railway.app').status_code == 200
assert client.get('/').status_code == 301
assert client.get('/', secure=True).status_code == 200
assert client.get('/app/', secure=True).status_code == 302
assert client.login(username='deployment', password='offline-test-password')
assert client.get('/app/', secure=True).status_code == 200
assert client.get('/app/api/bootstrap', secure=True).status_code == 200
assert client.get('/private/platform.sqlite3', secure=True).status_code == 404
'''
    for script in (setup, verify, verify):
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr

    # Exercise the real startup processes twice with the same persisted state.
    for _ in range(2):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        commands = [
            [sys.executable, "-m", "gunicorn", "--chdir", "web",
             "infiro_site.wsgi:application", "--bind", f"127.0.0.1:{port}",
             "--workers", "1", "--graceful-timeout", "2"],
            [sys.executable, "web/manage.py", "platform_worker"],
        ]
        migration = [sys.executable, "web/manage.py", "migrate", "--noinput"]
        script = ("from deploy.web_start import supervise; "
                  f"raise SystemExit(supervise({migration!r}, {commands!r}))")
        server = subprocess.Popen([sys.executable, "-c", script], cwd=ROOT, env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 20
            while True:
                assert server.poll() is None
                try:
                    request = Request(f"http://127.0.0.1:{port}/healthz",
                                      headers={"Host": "healthcheck.railway.app"})
                    with urlopen(request, timeout=1) as response:
                        assert response.status == 200
                    break
                except (URLError, TimeoutError):
                    assert time.monotonic() < deadline
                    time.sleep(.1)
            request = Request(f"http://127.0.0.1:{port}/",
                              headers={"Host": "testserver", "X-Forwarded-Proto": "https"})
            with urlopen(request, timeout=5) as response:
                assert response.status == 200
            result = subprocess.run([sys.executable, "-c", verify], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
        finally:
            server.send_signal(signal.SIGTERM)
            assert server.wait(timeout=25) == 0

    # Follow the documented MultiFernet rotation, then remove the old key.
    new_key = Fernet.generate_key().decode()
    env["INFIRO_CREDENTIAL_KEYS"] = new_key + "," + env["INFIRO_CREDENTIAL_KEYS"]
    rotation = """
import django
django.setup()
from django.db import transaction
from infiro_platform.models import Connection
from infiro_platform.security import cipher, unseal
with transaction.atomic():
    for connection in Connection.objects.all():
        unseal(connection)
        connection.ciphertext = cipher().rotate(connection.ciphertext.encode()).decode()
        connection.save(update_fields=['ciphertext'])
        unseal(connection)
"""
    subprocess.run([sys.executable, "-c", rotation], cwd=ROOT, env=env, check=True, timeout=30)
    env["INFIRO_CREDENTIAL_KEYS"] = new_key
    subprocess.run([sys.executable, "-c", verify], cwd=ROOT, env=env, check=True, timeout=30)


def test_startup_refuses_an_unmounted_volume(monkeypatch):
    from deploy.web_start import main

    monkeypatch.setattr("deploy.web_start.os.path.ismount", lambda path: False)
    with pytest.raises(SystemExit, match="persistent volume must be mounted"):
        main()


def test_startup_refuses_ephemeral_database(monkeypatch):
    from deploy.web_start import main

    monkeypatch.setattr("deploy.web_start.os.path.ismount", lambda path: True)
    monkeypatch.setenv("INFIRO_PRIVATE_ROOT", "/data/private")
    monkeypatch.setenv("INFIRO_DATABASE", "/tmp/platform.sqlite3")
    with pytest.raises(SystemExit, match="must be on the /data volume"):
        main()


def test_optimizer_snapshot_checksum_and_scope():
    import hashlib
    import tarfile

    checksum, relative = (ROOT / "vendor/llm-optimizer/SHA256SUMS").read_text().split()
    archive = ROOT / relative
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum
    with tarfile.open(archive) as source:
        paths = [Path(member.name) for member in source.getmembers()]
    assert any(path.name == "LICENSE" for path in paths)
    allowed = {"src", "pyproject.toml", "README.md", "LICENSE"}
    assert all(len(path.parts) == 1 or path.parts[1] in allowed for path in paths)
    assert not any(path.suffix in {".sqlite", ".sqlite3", ".pem", ".key"} for path in paths)
