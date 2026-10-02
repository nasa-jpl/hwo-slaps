"""Reusable bounded profile-optimization settings.

Procedure identifiers retain their historical values for result provenance;
settings do not depend on a particular release or observing campaign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

PROCEDURE_VERSION = "fresh_nonlinear_v7_lbfgsb_v2"
OBJECTIVE_VERSION = "consistent_sampling_v2"


@dataclass(frozen=True)
class FreshProfileSettings:
    """Bounded local-profile settings for a residual objective."""

    original_start_count: int = 8
    start_separation_normalized_l2: float = 0.05
    start_separation_posterior_sigma: float = 1.0
    maxiter: int = 500
    ftol: float = 0.0
    gtol: float = 1.0e-10
    maxls: int = 50
    repeat_maxiter: int = 1000
    repeat_ftol: float = 0.0
    repeat_gtol: float = 1.0e-12
    support_log_likelihood_tolerance: float = 0.1
    repeat_log_likelihood_tolerance: float = 0.1
    minimum_distinct_original_start_support: int = 2
    scalar_residual_tolerance: float = 1.0e-4
    version: str = PROCEDURE_VERSION

    def __post_init__(self) -> None:
        if self.original_start_count < 1:
            raise ValueError("original_start_count must be positive")
        if self.start_separation_normalized_l2 <= 0:
            raise ValueError("start separation must be positive")
        if self.start_separation_posterior_sigma <= 0:
            raise ValueError("posterior-sigma start separation must be positive")
        if self.maxiter < 1 or self.repeat_maxiter < 1 or self.maxls < 1:
            raise ValueError("optimizer iteration and line-search limits must be positive")
        if self.ftol < 0 or self.gtol < 0 or self.repeat_ftol < 0 or self.repeat_gtol < 0:
            raise ValueError("optimizer tolerances must be non-negative")
        if self.support_log_likelihood_tolerance < 0 or self.repeat_log_likelihood_tolerance < 0:
            raise ValueError("support tolerances must be non-negative")
        if self.minimum_distinct_original_start_support < 1:
            raise ValueError("minimum support must be positive")
        if self.scalar_residual_tolerance <= 0:
            raise ValueError("scalar residual tolerance must be positive")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FreshProfileSettings":
        """Construct settings from a flat or nested optimization mapping."""
        if mapping is None:
            return cls()
        current = mapping.get("current_search_optimization", mapping)
        if not isinstance(current, Mapping):
            raise ValueError("current_search_optimization must be a mapping")
        gate = current.get("support_gate", {})
        repeat = current.get("tighter_repeat", {})
        if not isinstance(gate, Mapping) or not isinstance(repeat, Mapping):
            raise ValueError("support_gate and tighter_repeat must be mappings")
        values = {
            "original_start_count": current.get("original_start_count", cls.original_start_count),
            "start_separation_normalized_l2": current.get(
                "start_separation_normalized_l2", cls.start_separation_normalized_l2
            ),
            "start_separation_posterior_sigma": current.get(
                "start_separation_posterior_sigma", cls.start_separation_posterior_sigma
            ),
            "maxiter": current.get("maxiter", cls.maxiter),
            "ftol": current.get("ftol", cls.ftol),
            "gtol": current.get("gtol", cls.gtol),
            "maxls": current.get("maxls", cls.maxls),
            "repeat_maxiter": repeat.get("maxiter", cls.repeat_maxiter),
            "repeat_ftol": repeat.get("ftol", cls.repeat_ftol),
            "repeat_gtol": repeat.get("gtol", cls.repeat_gtol),
            "support_log_likelihood_tolerance": gate.get(
                "log_likelihood_tolerance", cls.support_log_likelihood_tolerance
            ),
            "repeat_log_likelihood_tolerance": gate.get(
                "repeat_log_likelihood_tolerance", cls.repeat_log_likelihood_tolerance
            ),
            "minimum_distinct_original_start_support": gate.get(
                "minimum_distinct_original_starts",
                cls.minimum_distinct_original_start_support,
            ),
            "scalar_residual_tolerance": current.get(
                "scalar_residual_tolerance", cls.scalar_residual_tolerance
            ),
            "version": str(mapping.get("procedure_version", PROCEDURE_VERSION)),
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedure_version": self.version,
            "optimizer": "L-BFGS-B",
            "parameterization": "normalized_box_0_1",
            "original_start_count": self.original_start_count,
            "start_separation_normalized_l2": self.start_separation_normalized_l2,
            "maxiter": self.maxiter,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "maxls": self.maxls,
            "tighter_repeat": {
                "maxiter": self.repeat_maxiter,
                "ftol": self.repeat_ftol,
                "gtol": self.repeat_gtol,
            },
            "support_gate": {
                "minimum_distinct_original_starts": self.minimum_distinct_original_start_support,
                "log_likelihood_tolerance": self.support_log_likelihood_tolerance,
                "repeat_log_likelihood_tolerance": self.repeat_log_likelihood_tolerance,
            },
            "scalar_residual_tolerance": self.scalar_residual_tolerance,
            "fallback_solver": False,
            "best_finite_retention": True,
        }
