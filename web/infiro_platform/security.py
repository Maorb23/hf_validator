"""Credential envelopes and outbound network policy; no plaintext serialization."""

import hashlib
import ipaddress
import json
import re
import socket
from urllib.parse import urlsplit

from cryptography.fernet import Fernet, MultiFernet
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

PROVIDERS = {
    "nebius_inference": {"secrets": {"api_key"}, "required": {"api_key"}, "metadata": {"endpoint"}},
    "nebius_jobs": {
        "secrets": {"iam_token", "private_key"},
        "required": set(),
        "metadata": {"project_id", "auth_method", "service_account_id", "public_key_id"},
    },
    "huggingface": {"secrets": {"token"}, "required": {"token"}, "metadata": {"secret_version_id"}},
    "storage": {
        "secrets": {"access_key_id", "secret_access_key"},
        "required": {"access_key_id", "secret_access_key"},
        "metadata": {"bucket", "region", "endpoint", "secret_version_id"},
    },
    "tavily": {"secrets": {"api_key"}, "required": {"api_key"}, "metadata": set()},
}


def cipher():
    if not settings.INFIRO_CREDENTIAL_KEYS:
        raise ImproperlyConfigured(
            "Configure persistent INFIRO_CREDENTIAL_KEYS before storing credentials."
        )
    try:
        return MultiFernet(
            [Fernet(k.strip().encode()) for k in settings.INFIRO_CREDENTIAL_KEYS.split(",")]
        )
    except Exception:
        raise ImproperlyConfigured("INFIRO_CREDENTIAL_KEYS must contain Fernet keys.") from None


def seal(connection, values):
    envelope = {
        "owner": connection.owner_id,
        "id": str(connection.id),
        "kind": connection.kind,
        "values": values,
    }
    return cipher().encrypt(json.dumps(envelope).encode()).decode()


def unseal(connection):
    envelope = json.loads(cipher().decrypt(connection.ciphertext.encode()))
    if (envelope["owner"], envelope["id"], envelope["kind"]) != (
        connection.owner_id,
        str(connection.id),
        connection.kind,
    ):
        raise ValueError("Credential identity mismatch")
    return envelope["values"]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def public_ip(address):
    ip = ipaddress.ip_address(address.split("%")[0])
    if (
        not ip.is_global
        or ip.is_multicast
        or ip.is_reserved
        or (getattr(ip, "ipv4_mapped", None) and not ip.ipv4_mapped.is_global)
    ):
        raise ValueError(
            "Private, local, reserved and link-local network destinations are blocked."
        )


def endpoint(value):
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        raise ValueError("Invalid public HTTPS endpoint.") from None
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or port not in {None, 443}
        or "\\" in value
        or any(ord(c) < 33 for c in value)
    ):
        raise ValueError("Endpoints must be credential-free public HTTPS URLs on port 443.")
    try:
        addresses = socket.getaddrinfo(url.hostname, 443, type=socket.SOCK_STREAM)
    except OSError:
        raise ValueError("Endpoint DNS lookup failed.") from None
    if not addresses:
        raise ValueError("Endpoint has no public destination.")
    for address in addresses:
        public_ip(address[4][0])
    return value.rstrip("/")


def validate_connection(kind, metadata, values):
    spec = PROVIDERS.get(kind)
    if not spec or not isinstance(metadata, dict) or not isinstance(values, dict):
        raise ValueError("Choose a supported connection type.")
    if (
        set(metadata) - spec["metadata"]
        or set(values) - spec["secrets"]
        or not spec["required"] <= set(values)
    ):
        raise ValueError("Connection fields do not match this provider.")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 8192 for v in values.values()):
        raise ValueError("Credential fields must contain nonempty text.")
    if any(not isinstance(v, str) or len(v) > 512 for v in metadata.values()):
        raise ValueError("Invalid provider settings.")
    metadata = {k: v.strip() for k, v in metadata.items()}
    if kind == "nebius_jobs":
        method = metadata.get("auth_method", "iam_token")
        if method not in {"iam_token", "service_account"}:
            raise ValueError("Choose service-account or IAM-token authentication.")
        if set(values) != ({"private_key"} if method == "service_account" else {"iam_token"}):
            raise ValueError("Provide only the credential for the selected authentication method.")
        if method == "service_account":
            from .nebius_auth import private_key

            for key in ("service_account_id", "public_key_id"):
                if not re.fullmatch(r"[a-zA-Z0-9-]{5,100}", metadata.get(key, "")):
                    raise ValueError("Service account ID and public key ID are required.")
            private_key(values["private_key"])
        elif {"service_account_id", "public_key_id"} & set(metadata):
            raise ValueError("Service account settings require service-account authentication.")
    if metadata.get("secret_version_id") and not re.fullmatch(
        r"mbse[a-zA-Z0-9]*-?[a-zA-Z0-9]+", metadata["secret_version_id"]
    ):
        raise ValueError("Enter a SecretStash secret or version ID (starting with mbse).")
    if kind == "nebius_inference":
        metadata["endpoint"] = endpoint(
            metadata.get("endpoint") or "https://api.tokenfactory.nebius.com/v1"
        )
    if kind == "nebius_jobs" and not re.fullmatch(
        r"[a-zA-Z0-9-]{5,100}", metadata.get("project_id", "")
    ):
        raise ValueError("A Nebius project ID is required; inference keys do not authorize jobs.")
    if kind == "storage":
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", metadata.get("bucket", ""))
            or ".." in metadata["bucket"]
        ):
            raise ValueError("Enter a valid S3 bucket name.")
        if not re.fullmatch(r"[a-z0-9-]{2,50}", metadata.get("region", "")):
            raise ValueError("Storage region is required.")
        if metadata.get("endpoint"):
            metadata["endpoint"] = endpoint(metadata["endpoint"])
    return metadata


def private_dir(owner_id, job_id):
    # Both values are database identities, never path strings supplied by a user.
    root = settings.INFIRO_PRIVATE_ROOT / "users" / str(int(owner_id)) / "jobs" / str(job_id)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def workload_cache_dir(owner_id):
    root = settings.INFIRO_PRIVATE_ROOT / "users" / str(int(owner_id)) / "sharegpt-cache"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def install_network_guard(max_bytes=128 * 1024 * 1024):
    """Child-only guard checks the actual socket peer, including every redirect.

    DNS answers are rechecked at connect time, preventing DNS rebinding. Reads are
    bounded across HTTP clients, SDKs and dataset downloads, including TLS reads.
    """
    import ssl

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    received = [0]

    def check(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            host, port = address[:2]
            if port != 443:
                raise ValueError("Outbound services must use HTTPS port 443.")
            answers = socket.getaddrinfo(host, port, sock.family, socket.SOCK_STREAM)
            for answer in answers:
                public_ip(answer[4][0])
            # Pass a validated literal peer to the native connect; never allow a
            # second hostname lookup between validation and the connection.
            return answers[0][4]
        return address

    def connect(sock, address):
        return original_connect(sock, check(sock, address))

    def connect_ex(sock, address):
        return original_connect_ex(sock, check(sock, address))

    def bounded(method):
        def read(*args, **kwargs):
            result = method(*args, **kwargs)
            received[0] += len(result) if isinstance(result, bytes) else (result or 0)
            if received[0] > max_bytes:
                label = (
                    "1 GiB per workload operation"
                    if max_bytes > 128 * 1024**2
                    else "128 MiB per operation"
                )
                raise ValueError(f"Download limit exceeded ({label}).")
            return result

        return read

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.recv = bounded(socket.socket.recv)
    socket.socket.recv_into = bounded(socket.socket.recv_into)
    ssl.SSLSocket.read = bounded(ssl.SSLSocket.read)
