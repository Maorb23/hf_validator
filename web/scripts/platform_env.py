"""Create a persistent local environment file outside version control; print no keys."""

import argparse
import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--data-dir", type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
data_dir = args.data_dir.resolve()
data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
if any(c in str(data_dir) for c in "\n\r'\""):
    raise SystemExit("Choose a data directory without quotes or newlines.")
try:
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    raise SystemExit(
        "Environment file already exists; keep it to retain access to encrypted credentials."
    ) from None
with os.fdopen(descriptor, "w") as output:
    output.write("DJANGO_SECRET_KEY=" + secrets.token_urlsafe(64) + "\n")
    output.write("INFIRO_CREDENTIAL_KEYS=" + Fernet.generate_key().decode() + "\n")
    output.write("INFIRO_DATABASE='" + str(data_dir / "platform.sqlite3") + "'\n")
    output.write("INFIRO_PRIVATE_ROOT='" + str(data_dir) + "'\n")
    output.write("INFIRO_LIVE_ENABLED=0\n")
print(
    "Created a local environment file with persistent keys. Keep it private "
    "and back it up separately from the database."
)
