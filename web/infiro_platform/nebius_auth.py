"""Explicit, in-memory Nebius credentials; never consult CLI profiles or key paths."""

import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


def private_key(value):
    try:
        raw = value.strip().encode()
        if not raw.startswith(b"-----BEGIN "):
            raw = base64.b64decode(raw, validate=True)
        key = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(key, RSAPrivateKey) or key.key_size < 2048:
            raise ValueError
        return key
    except Exception:
        raise ValueError(
            "Enter a valid RSA private key (PEM or base64 PEM), at least 2048 bits."
        ) from None


def make_sdk(metadata, values):
    from nebius.base.service_account.service_account import ServiceAccount
    from nebius.sdk import SDK

    if metadata.get("auth_method", "iam_token") == "service_account":
        credentials = ServiceAccount(
            private_key(values["private_key"]),
            metadata["public_key_id"],
            metadata["service_account_id"],
        )
    else:
        credentials = values["iam_token"]
    # The SDK exchanges service-account assertions and refreshes expiring tokens.
    return SDK(credentials=credentials)
