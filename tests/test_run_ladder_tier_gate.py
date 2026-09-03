"""Tests for the production and caller-supplied ladder tier gates."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT/"scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_ladder as runner  # noqa: E402


def _staged_config(tier: str) -> dict:
    """Return a minimal staged configuration for tier-gate tests."""
    return {
        "ladder": {
            "tier": tier,
            "golden": True,
            "parent_overlap": False,
            "psf_state": "science35",
            "kernel": "k999",
            "engine": "jax",
            "mask_mode": "all_pixels",
            "node_spacing_arcsec": 0.05,
            "threshold": "q_F >= 10",
        },
        "modeling": {"fit_psf": {"mode": "matched"}},
    }


def test_default_production_gate_rejects_validation():
    """The default production gate rejects the validation tier."""
    with pytest.raises(ValueError, match="not one of"):
        runner._verify_ladder_block(_staged_config("validation"))


def test_explicit_gate_accepts_validation():
    """A caller-supplied gate accepts the declared validation tier."""
    config = _staged_config("validation")
    allowed_tiers = ("parent", "selected", "validation")
    assert runner._verify_ladder_block(
        config, allowed_tiers=allowed_tiers
    ) is config["ladder"]


def test_explicit_gate_still_rejects_unknown_tier():
    """A caller-supplied gate still rejects an undeclared tier."""
    config = _staged_config("unknown")
    allowed_tiers = ("parent", "selected", "validation")
    with pytest.raises(ValueError, match="not one of"):
        runner._verify_ladder_block(config, allowed_tiers=allowed_tiers)
