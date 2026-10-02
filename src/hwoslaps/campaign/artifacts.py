"""Shared byte-level artifact identities for campaign execution."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path) -> str:
    """Hash an artifact without loading its entire payload into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
