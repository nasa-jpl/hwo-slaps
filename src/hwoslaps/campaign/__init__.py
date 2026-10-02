"""Reusable immutable campaign execution and adaptive ladder primitives.

Study population and release contracts live in source-only ``studies``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CampaignError", "JobResult", "freeze_campaign", "harvest_campaign",
    "run_campaign", "validate_campaign_manifest",
]


def __getattr__(name: str) -> Any:
    """Load executor APIs lazily so importing campaigns has no worker effects."""
    if name in __all__:
        from . import s1_lite
        return getattr(s1_lite, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
