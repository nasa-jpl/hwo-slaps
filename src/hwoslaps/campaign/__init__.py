"""Campaign orchestration for HWO-SLAPS production runs."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

__all__ = [
    "CampaignError",
    "DesignFreezeError",
    "JobResult",
    "Stage0Error",
    "build_stage0_campaign",
    "design_freeze_digest",
    "freeze_campaign",
    "harvest_campaign",
    "load_design_freeze",
    "run_campaign",
    "sample_pool",
    "validate_campaign_manifest",
    "validate_design_freeze",
    "validate_stage0_manifest",
    "verify_bound_artifacts",
    "write_stage0_campaign",
]

_S1_LITE_NAMES = frozenset({
    "CampaignError",
    "JobResult",
    "freeze_campaign",
    "harvest_campaign",
    "run_campaign",
    "validate_campaign_manifest",
})

_DESIGN_FREEZE_NAMES = frozenset({
    "DesignFreezeError",
    "design_freeze_digest",
    "load_design_freeze",
    "validate_design_freeze",
    "verify_bound_artifacts",
})

_STAGE0_NAMES = frozenset({
    "Stage0Error",
    "build_stage0_campaign",
    "sample_pool",
    "validate_stage0_manifest",
    "write_stage0_campaign",
})

if TYPE_CHECKING:
    from .design_freeze import (
        DesignFreezeError,
        design_freeze_digest,
        load_design_freeze,
        validate_design_freeze,
        verify_bound_artifacts,
    )
    from .s1_lite import (
        CampaignError,
        JobResult,
        freeze_campaign,
        harvest_campaign,
        run_campaign,
        validate_campaign_manifest,
    )
    from .stage0 import (
        Stage0Error,
        build_stage0_campaign,
        sample_pool,
        validate_stage0_manifest,
        write_stage0_campaign,
    )


def __getattr__(name: str) -> Any:
    """Resolve campaign APIs without eager executor-module imports."""
    if name in _S1_LITE_NAMES:
        from . import s1_lite

        return getattr(s1_lite, name)
    if name in _DESIGN_FREEZE_NAMES:
        from . import design_freeze

        return getattr(design_freeze, name)
    if name in _STAGE0_NAMES:
        from . import stage0

        return getattr(stage0, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
