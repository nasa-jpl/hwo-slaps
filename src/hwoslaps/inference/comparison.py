"""Readable scientific compatibility of a forecast reference and nonlinear fit.

The truth-preserving comparison digest remains useful across model-PSF arms.
This report checks fitted assumptions separately. It states the linearization
approximation and excludes differing statistic targets, physical inputs,
constraints or noise from strict validation. Legacy inputs remain unassessed.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from ..identity import json_ready
from .result import ForecastReference


@dataclass(frozen=True)
class ComparisonReport:
    checks: Mapping[str, Mapping[str, Any]]

    @property
    def differences(self) -> tuple[str, ...]:
        return tuple(name for name, check in self.checks.items() if check["status"] == "different")

    @property
    def unassessed(self) -> tuple[str, ...]:
        return tuple(name for name, check in self.checks.items() if check["status"] == "not_assessed")

    @property
    def compatible(self) -> bool:
        return not self.differences and not self.unassessed

    def to_mapping(self) -> dict[str, Any]:
        return json_ready({"compatible": self.compatible, "differences": self.differences,
                           "not_assessed": self.unassessed, "checks": self.checks})


def compare_forecast(reference: ForecastReference, *, hypothesis: Any, data: Mapping[str, Any],
                     fitted_parameters: tuple[str, ...], comparison_digest: str) -> ComparisonReport:
    """Compare recorded physical and fitted assumptions without redefining digests."""
    checks = {}

    def compare(name: str, forecast: Any, fit: Any, reason: str) -> None:
        status = "not_assessed" if forecast is None or fit is None else (
            "match" if json_ready(forecast) == json_ready(fit) else "different")
        checks[name] = {"status": status, "forecast": forecast, "fit": fit, "reason": reason}

    compare("physical_node", [reference.mass_msun, reference.position_yx_arcsec],
            [hypothesis.mass_msun, hypothesis.position_yx_arcsec], "Physical mass and position in y,x arcsec.")
    compare("truth_configuration", reference.comparison_digest, comparison_digest,
            "Truth-preserving configuration identity, excluding the model PSF.")
    compare("model_psf", reference.model_kernel.to_mapping(), data.get("model_kernel"),
            "Actual model kernel bytes, shape and angular sampling.")
    compare("mask", reference.mask_digest, data.get("mask", {}).get("digest"), "Selected detector pixels.")
    compare("nuisance_parameters", sorted(reference.nuisance_names), sorted(fitted_parameters),
            "All nuisance directions must be fitted, including background and PSF directions.")
    inputs, fitted = reference.scientific_inputs or {}, data.get("comparison_inputs", {})
    for name in ("trial_model", "truth_psf", "fitted_psf", "detector_geometry", "response", "noise", "nuisance_constraints"):
        compare(name, inputs.get(name), fitted.get(name), {
            "trial_model": "Halo recipe, mass definition, plane/redshifts and cosmology.",
            "truth_psf": "Truth kernel bindings and angular sampling.",
            "fitted_psf": "Normalized kernels actually used by the forecast and nonlinear convolver.",
            "detector_geometry": "Detector shape and angular pixel scale.",
            "response": "Image response operator and light-profile sampling.",
            "noise": "Normalized noise values, whitening covariance and fiducial variance choice.",
            "nuisance_constraints": "Gaussian penalties, flat directions, and finite uniform fit boxes differ.",
        }[name])
    mode = fitted.get("statistic", {}).get("amplitude")
    statistic_status = ("not_assessed" if not inputs.get("statistic") or mode is None else
                        "different" if reference.metric == "q_mismatch" or mode in ("local_search", "freed")
                        else "approximation")
    checks["statistic_conventions"] = {
        "status": statistic_status,
        "forecast": inputs.get("statistic"), "fit": fitted.get("statistic"),
        "reason": "A q_mismatch forecast profiles a signed signal amplitude, whereas the nonlinear roles fit a "
                  "physical halo. A position/mass search also differs from a fixed-node forecast. These targets "
                  "require diagnostic comparison. A fixed-template q_asimov comparison retains the first-order "
                  "approximation; compatibility establishes neither numerical agreement nor calibrated significance."}
    return ComparisonReport(checks)
