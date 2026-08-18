"""Campaign orchestration for HWO-SLAPS production runs."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

__all__ = [
    "CampaignError",
    "JobResult",
    "freeze_campaign",
    "harvest_campaign",
    "run_campaign",
    "validate_campaign_manifest",
]

if TYPE_CHECKING:
    from .s1_lite import (
        CampaignError,
        JobResult,
        freeze_campaign,
        harvest_campaign,
        run_campaign,
        validate_campaign_manifest,
    )


def __getattr__(name: str) -> Any:
    """Resolve campaign APIs without eager executor-module imports."""
    if name in __all__:
        from . import s1_lite

        return getattr(s1_lite, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
