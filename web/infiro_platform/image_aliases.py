"""Verify short Nebius registry tags without consulting local Docker credentials."""

import hashlib
import re
from urllib.parse import urlsplit
from urllib.request import parse_http_list, parse_keqv_list

import httpx


def verified_alias(pinned, aliases, sdk):
    alias = aliases.get(pinned)
    if not alias or len(alias) > 64:
        raise ValueError("Configure a worker image tag of at most 64 characters for this digest.")
    match = re.fullmatch(
        r"(cr\.[a-z0-9-]+\.nebius\.cloud)/([a-zA-Z0-9_./-]+):([a-zA-Z0-9_.-]+)", alias
    )
    if not match or ".." in match[2].split("/"):
        raise ValueError("Worker image aliases must use the private Nebius registry.")
    host, repository, tag = match.groups()
    expected = pinned.rsplit("@", 1)[-1]
    token = sdk.get_token_sync(timeout=30).token
    headers = {
        "Accept": ", ".join(
            (
                "application/vnd.docker.distribution.manifest.v2+json",
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
                "application/vnd.oci.image.index.v1+json",
            )
        )
    }
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        url = f"https://{host}/v2/{repository}/manifests/{tag}"
        response = client.get(url, headers=headers, auth=("iam", token))
        if response.status_code == 401:
            challenge = response.headers.get("www-authenticate", "")
            if not challenge.lower().startswith("bearer "):
                raise ValueError("Worker image registry authentication failed.")
            fields = parse_keqv_list(parse_http_list(challenge[7:]))
            realm = fields.get("realm", "")
            target = urlsplit(realm)
            if target.scheme != "https" or target.netloc != host:
                raise ValueError(
                    "Worker image registry returned an unexpected authentication endpoint."
                )
            auth_response = client.get(
                realm,
                params={
                    "service": fields.get("service", host),
                    "scope": f"repository:{repository}:pull",
                },
                auth=("iam", token),
            )
            if auth_response.status_code != 200:
                raise ValueError("Worker image registry authentication failed.")
            registry_token = auth_response.json().get("token") or auth_response.json().get(
                "access_token"
            )
            if not isinstance(registry_token, str) or not registry_token:
                raise ValueError("Worker image registry returned no access token.")
            response = client.get(
                url, headers=headers | {"Authorization": f"Bearer {registry_token}"}
            )
        if response.status_code != 200:
            raise ValueError("Unable to verify the worker image tag in the registry.")
        actual = "sha256:" + hashlib.sha256(response.content).hexdigest()
        if actual != expected:
            raise ValueError(
                "Worker image tag no longer matches the reviewed digest. Submission stopped."
            )
    return alias
