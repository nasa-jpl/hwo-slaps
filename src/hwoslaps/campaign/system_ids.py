"""The bare ``sysNNNN`` identifier shared by catalog, producers and harvester."""

from __future__ import annotations

import re


class SystemIdError(ValueError):
    """Raised when a value carries no ``sysNNNN`` identifier."""


def bare_system_id(value: str) -> str:
    """Return the last ``sysNNNN`` token in ``value``, such as a ladder run name."""
    matches = re.findall(r"sys\d{4}", str(value))
    if not matches:
        raise SystemIdError(f"No sysNNNN identifier in {value!r}")
    return matches[-1]
