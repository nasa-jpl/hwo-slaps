"""Readable scientific compatibility of a forecast reference and nonlinear fit.

The truth-preserving comparison digest remains useful across model-PSF arms.
This report checks fitted assumptions separately. Linearization and q definitions
are reported as approximations; differing physical inputs, constraints or noise
exclude strict validation. Legacy records without the inputs are not assessed.
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
    checks["statistic_conventions"] = {
        "status": "approximation" if inputs.get("statistic") and fitted.get("statistic") else "not_assessed",
        "forecast": inputs.get("statistic"), "fit": fitted.get("statistic"),
        "reason": "Linear profiled q is compared with a nonlinear role likelihood ratio. Fixed/fitted signal "
                  "amplitude, first-order nuisance response and one-sided conventions remain explicit approximations; "
                  "compatibility does not establish numerical agreement or a calibrated search significance."}
    return ComparisonReport(checks)
