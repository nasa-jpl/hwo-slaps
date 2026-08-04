"""Runtime smoke tests for PyAutoLens model conversion."""

from __future__ import annotations

pytest_plugins = []

import pytest

pytest.importorskip("autolens")
pytest.importorskip("autofit")

from hwoslaps.modeling.nonlinear.autolens_model_builder import (  # noqa: E402
    autofit_model_from_spec,
    smooth_model_spec_from_config,
    subhalo_model_spec_from_trial,
)
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial  # noqa: E402


def _config() -> dict:
    return {
        "lensing": {
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "centre": [0.0, 0.0],
                    "einstein_radius": 1.0,
                    "ell_comps": [0.1, 0.0],
                },
            },
            "source_galaxy": {
                "redshift": 0.6,
                "light": {
                    "type": "Exponential",
                    "centre": [-0.03, 0.08],
                    "ell_comps": [0.1, 0.2],
                    "intensity": 2.0,
                    "effective_radius": 0.11,
                },
            },
        }
    }


def test_runtime_autofit_model_conversion_for_smooth_and_nfw_subhalo():
    """Convert smooth and NFW subhalo specs into AutoFit models."""
    trial = SubhaloTrial(
        case_id="nfw",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.2, -0.1),
        model="NFW",
        profile_class="NFWSph",
        lens_redshift=0.2,
        source_redshift=0.6,
        kappa_s=0.01,
        scale_radius_arcsec=0.2,
    )

    smooth_model = autofit_model_from_spec(smooth_model_spec_from_config(_config()))
    subhalo_model = autofit_model_from_spec(
        subhalo_model_spec_from_trial(_config(), trial)
    )

    assert smooth_model.total_free_parameters > 0
    assert subhalo_model.total_free_parameters >= smooth_model.total_free_parameters
