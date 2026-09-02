"""Tests for freed subhalo specifications and validator execution."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import autolens as al
import numpy as np
import pytest

from hwoslaps.lensing.mass_models import (
    concentration_mass_relation,
    nfw_lensing_parameters,
)
from hwoslaps.modeling.nonlinear.autolens_model_builder import (
    DEFAULT_PRIOR_WIDTHS,
    autofit_model_from_spec,
    fixed_point_model_spec_from_trial,
    smooth_model_spec_from_config,
    subhalo_model_spec_from_trial,
)
from hwoslaps.modeling.nonlinear.autolens_runner import (
    AutoLensFitRunner,
    NonlinearSearchSettings,
    analysis_key_from,
)
from hwoslaps.modeling.nonlinear.dataset_builder import (
    NonlinearDatasetMetadata,
)
from hwoslaps.modeling.nonlinear.mass_mapping import (
    build_mass_mapping_context,
)
from hwoslaps.modeling.nonlinear.output_schema import NonlinearFitSummary
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial
from hwoslaps.modeling.nonlinear.validator import NonlinearMetricValidator
from hwoslaps.psf.utils import make_pyauto_convolver, make_pyauto_kernel


def _config():
    """Return a complete canonical NFW nonlinear configuration."""
    return {
        "global_seed": 1,
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
            "subhalo": {
                "enabled": True,
                "model": "NFW",
                "mass": 1.0e7,
                "concentration": {
                    "model": "moline2017_eq7",
                    "x_sub": 1.0,
                    "h": None,
                },
            },
            "cosmology": "Planck15",
        },
        "observation": {"throughput": 1.0},
    }


def _trial(model="NFW"):
    """Return a canonical trial for one supported subhalo model."""
    profile_class = {
        "NFW": "NFWSph",
        "SIS": "IsothermalSph",
        "PointMass": "PointMass",
    }[model]
    return SubhaloTrial(
        case_id="freed-case",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.2, -0.1),
        model=model,
        profile_class=profile_class,
        lens_redshift=0.2,
        source_redshift=0.6,
        kappa_s=0.01 if model == "NFW" else None,
        scale_radius_arcsec=0.2 if model == "NFW" else None,
        einstein_radius_arcsec=0.01 if model != "NFW" else None,
    )


def _metadata():
    """Return compact validation dataset metadata."""
    return NonlinearDatasetMetadata(
        dataset_kind="asimov",
        data_units="electron_rate",
        background_treatment="subtract_known",
        sky_dark_background_adu=0.0,
        mask_name="all_pixels",
        n_unmasked_pixels=25,
        psf_truth_label="truth",
        psf_fit_label="fit",
    )


class _Native:
    """Minimal native-array wrapper for identity hashing."""

    def __init__(self, values):
        self.native = np.asarray(values, dtype=float)


def _identity_dataset():
    """Return a minimal dataset accepted by analysis-key hashing."""
    return SimpleNamespace(
        data=_Native([[1.0]]),
        noise_map=_Native([[0.1]]),
        psf=_Native([[1.0]]),
    )


def _smooth_analysis_key(dataset, config, priors_config=None):
    """Return the validator's smooth-only analysis identity."""
    spec = smooth_model_spec_from_config(config, priors_config=priors_config)
    resolved_widths = dict(DEFAULT_PRIOR_WIDTHS)
    if priors_config:
        resolved_widths.update(priors_config)
    metadata = dict(spec.metadata)
    metadata.update(
        {
            "fit_mode": "smooth",
            "resolved_prior_widths": resolved_widths,
        }
    )
    return analysis_key_from(dataset, _metadata(), metadata)


def test_freed_spec_parameters_overrides_and_context_guards():
    """Free centre and mass with explicit bounds and reject mismatches."""
    config = _config()
    context = build_mass_mapping_context(config)
    spec = subhalo_model_spec_from_trial(
        config,
        _trial(),
        fit_mode="freed",
        mass_context=context,
    )
    subhalo = spec.galaxies["lens"].components["subhalo"]
    assert subhalo.class_name == "NFWMCRSubhaloSph"
    assert set(subhalo.parameters) == {
        "centre_0",
        "centre_1",
        "log10_m200",
        "mapping_context",
    }
    assert subhalo.parameters["centre_0"].lower == pytest.approx(0.05)
    assert subhalo.parameters["centre_0"].upper == pytest.approx(0.35)
    assert subhalo.parameters["log10_m200"].lower == pytest.approx(6.0)
    assert subhalo.parameters["log10_m200"].upper == pytest.approx(8.5)

    overridden = subhalo_model_spec_from_trial(
        config,
        _trial(),
        priors_config={"subhalo_freed_centre_window_arcsec": 0.2},
        fit_mode="freed",
        mass_context=context,
    )
    assert overridden.galaxies["lens"].components["subhalo"].parameters[
        "centre_0"
    ].lower == pytest.approx(0.0)

    with pytest.raises(ValueError, match="build_mass_mapping_context"):
        subhalo_model_spec_from_trial(config, _trial(), fit_mode="freed")
    wrong_model = _trial("SIS")
    with pytest.raises(ValueError, match="model"):
        subhalo_model_spec_from_trial(
            config,
            wrong_model,
            fit_mode="freed",
            mass_context=context,
        )
    wrong_redshift = SubhaloTrial(
        **{
            **_trial().to_dict(),
            "lens_redshift": 0.21,
        }
    )
    with pytest.raises(ValueError, match="redshift"):
        subhalo_model_spec_from_trial(
            config,
            wrong_redshift,
            fit_mode="freed",
            mass_context=context,
        )
    wrong_source_redshift = SubhaloTrial(
        **{
            **_trial().to_dict(),
            "source_redshift": 0.61,
        }
    )
    with pytest.raises(ValueError, match="source redshift"):
        subhalo_model_spec_from_trial(
            config,
            wrong_source_redshift,
            fit_mode="freed",
            mass_context=context,
        )
    with pytest.raises(ValueError, match="must be None"):
        subhalo_model_spec_from_trial(
            config,
            _trial(),
            fit_mode="fixed_template",
            mass_context=context,
        )


class _StubAnalysis:
    """Return a controlled fixed-template point likelihood."""

    def __init__(self, value=-5.0):
        self.value = value
        self.last_instance = None

    def log_likelihood_function(self, instance):
        self.last_instance = instance
        return self.value


class _StubRunner:
    """Return deterministic fit summaries while recording calls."""

    def __init__(self, subhalo_log_l=-4.0):
        self.settings = NonlinearSearchSettings(
            n_live_smooth=3,
            n_live_subhalo_fixed=5,
            n_live_subhalo_search=7,
        )
        self.subhalo_log_l = subhalo_log_l
        self.calls = []
        self.analysis = _StubAnalysis()

    def make_analysis(self, dataset, model_metadata=None):
        return self.analysis

    def run_model(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs["role"]
        return NonlinearFitSummary(
            model_role=role,
            fit_mode=kwargs["fit_mode"],
            status="success",
            log_likelihood_max=(
                -10.0 if role == "smooth" else self.subhalo_log_l
            ),
            analysis_key=kwargs["analysis_key"],
            n_live=kwargs["n_live"],
        )


def test_validate_fixed_template_delegates_without_behavior_change():
    """Keep fixed-template statuses and likelihood extraction unchanged."""
    first_runner = _StubRunner()
    delegated = NonlinearMetricValidator(first_runner).validate_fixed_template(
        _identity_dataset(),
        _metadata(),
        _config(),
        _trial(),
    )
    second_runner = _StubRunner()
    direct = NonlinearMetricValidator(second_runner).validate_case(
        _identity_dataset(),
        _metadata(),
        _config(),
        _trial(),
        fit_mode="fixed_template",
    )
    assert delegated.smooth_fit.status == direct.smooth_fit.status
    assert delegated.subhalo_fit.status == direct.subhalo_fit.status
    assert delegated.smooth_fit.log_likelihood_max == -10.0
    assert delegated.subhalo_fit.log_likelihood_max == -4.0
    assert first_runner.calls[1]["n_live"] == 5


def test_freed_invariant_records_point_and_flags_low_search():
    """Record the direct fixed point and flag an artificially low maximum."""
    context = build_mass_mapping_context(_config())
    runner = _StubRunner(subhalo_log_l=-6.0)
    result = NonlinearMetricValidator(runner).validate_case(
        _identity_dataset(),
        _metadata(),
        _config(),
        _trial(),
        fit_mode="freed",
        mass_context=context,
    )
    assert result.diagnostics["log_l_fixed_template_point"] == pytest.approx(
        -5.0
    )
    assert "freed_below_fixed_template" in result.quality_flags
    assert runner.calls[1]["n_live"] == 7


def test_smooth_reuse_skips_smooth_search_and_sets_flag():
    """Reuse a supplied denominator and run only the subhalo search."""
    runner = _StubRunner()
    dataset = _identity_dataset()
    smooth = NonlinearFitSummary(
        model_role="smooth",
        fit_mode="fixed_template",
        status="success",
        log_likelihood_max=-10.0,
        analysis_key=_smooth_analysis_key(dataset, _config()),
    )
    result = NonlinearMetricValidator(runner).validate_case(
        dataset,
        _metadata(),
        _config(),
        _trial(),
        fit_mode="fixed_template",
        smooth_result=smooth,
    )
    assert [call["role"] for call in runner.calls] == ["subhalo"]
    assert "smooth_reused" in result.quality_flags


def _tiny_asimov_dataset(config):
    """Build a tiny finite-noise Asimov dataset with an NFW subhalo."""
    cosmology = al.cosmo.Planck15()
    concentration = concentration_mass_relation(
        1.0e7,
        model="moline2017_eq7",
        x_sub=1.0,
        h=0.6774,
    )
    kappa_s, scale_radius = nfw_lensing_parameters(
        1.0e7,
        concentration,
        0.2,
        0.6,
        cosmology,
    )
    lens = al.Galaxy(
        redshift=0.2,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.0,
            ell_comps=(0.1, 0.0),
        ),
        subhalo=al.mp.NFWSph(
            centre=(0.2, -0.1),
            kappa_s=kappa_s,
            scale_radius=scale_radius,
        ),
    )
    light = config["lensing"]["source_galaxy"]["light"]
    source = al.Galaxy(
        redshift=0.6,
        light=al.lp.Exponential(
            centre=tuple(light["centre"]),
            ell_comps=tuple(light["ell_comps"]),
            intensity=light["intensity"],
            effective_radius=light["effective_radius"],
        ),
    )
    grid = al.Grid2D.uniform(shape_native=(9, 9), pixel_scales=0.2)
    image = al.Tracer(galaxies=[lens, source]).image_2d_from(grid)
    data = al.Array2D.no_mask(
        values=np.asarray(image.native),
        pixel_scales=0.2,
    )
    noise = al.Array2D.full(
        fill_value=0.2,
        shape_native=(9, 9),
        pixel_scales=0.2,
    )
    return al.Imaging(
        data=data,
        noise_map=noise,
        psf=make_pyauto_convolver(
            make_pyauto_kernel(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                pixel_scales=0.2,
            )
        ),
        over_sample_size_lp=1,
    )


def test_real_two_core_freed_validation_populates_recovery(tmp_path):
    """Complete a tiny two-core freed validation with recovery output."""
    config = _config()
    context = build_mass_mapping_context(config)
    settings = NonlinearSearchSettings(
        n_live_smooth=5,
        n_live_subhalo_search=5,
        number_of_cores=2,
        maxcall=80,
        seed=11,
        path_prefix="item7-smoke",
    )
    runner = AutoLensFitRunner(settings, output_dir=tmp_path)
    result = NonlinearMetricValidator(runner).validate_case(
        _tiny_asimov_dataset(config),
        _metadata(),
        config,
        _trial(),
        fit_mode="freed",
        mass_context=context,
        analysis_key="item7e2e00000001",
    )
    assert result.smooth_fit.status == "success", result.smooth_fit.error
    assert result.subhalo_fit.status == "success", result.subhalo_fit.error
    assert result.subhalo_fit.n_like_max_reached is not None
    assert result.subhalo_recovery is not None
    assert np.isfinite(result.subhalo_recovery.log10_m200_ml)


GOLDEN_SMOOTH = {
    "model_type": "smooth",
    "galaxies": {
        "lens": {
            "name": "lens",
            "redshift": {
                "kind": "fixed",
                "value": 0.2,
                "lower": None,
                "upper": None,
            },
            "components": {
                "mass": {
                    "class_name": "Isothermal",
                    "parameters": {
                        "centre_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.005,
                            "upper": 0.005,
                        },
                        "centre_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.005,
                            "upper": 0.005,
                        },
                        "einstein_radius": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.99,
                            "upper": 1.01,
                        },
                        "ell_comps_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.08,
                            "upper": 0.12000000000000001,
                        },
                        "ell_comps_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.02,
                            "upper": 0.02,
                        },
                    },
                },
            },
        },
        "source": {
            "name": "source",
            "redshift": {
                "kind": "fixed",
                "value": 0.6,
                "lower": None,
                "upper": None,
            },
            "components": {
                "light": {
                    "class_name": "Exponential",
                    "parameters": {
                        "centre_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.04,
                            "upper": -0.019999999999999997,
                        },
                        "centre_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.07,
                            "upper": 0.09,
                        },
                        "ell_comps_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.05,
                            "upper": 0.15000000000000002,
                        },
                        "ell_comps_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.15000000000000002,
                            "upper": 0.25,
                        },
                        "intensity": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 1.0,
                            "upper": 3.0,
                        },
                        "effective_radius": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.077,
                            "upper": 0.14300000000000002,
                        },
                    },
                },
            },
        },
    },
    "fit_mode": "smooth",
    "metadata": {"builder": "smooth_model_spec_from_config"},
}


GOLDEN_FIXED = {
    "model_type": "subhalo",
    "galaxies": {
        "lens": {
            "name": "lens",
            "redshift": {
                "kind": "fixed",
                "value": 0.2,
                "lower": None,
                "upper": None,
            },
            "components": {
                "mass": {
                    "class_name": "Isothermal",
                    "parameters": {
                        "centre_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.005,
                            "upper": 0.005,
                        },
                        "centre_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.005,
                            "upper": 0.005,
                        },
                        "einstein_radius": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.99,
                            "upper": 1.01,
                        },
                        "ell_comps_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.08,
                            "upper": 0.12000000000000001,
                        },
                        "ell_comps_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.02,
                            "upper": 0.02,
                        },
                    },
                },
                "subhalo": {
                    "class_name": "NFWSph",
                    "parameters": {
                        "centre_0": {
                            "kind": "fixed",
                            "value": 0.2,
                            "lower": None,
                            "upper": None,
                        },
                        "centre_1": {
                            "kind": "fixed",
                            "value": -0.1,
                            "lower": None,
                            "upper": None,
                        },
                        "kappa_s": {
                            "kind": "fixed",
                            "value": 0.01,
                            "lower": None,
                            "upper": None,
                        },
                        "scale_radius": {
                            "kind": "fixed",
                            "value": 0.2,
                            "lower": None,
                            "upper": None,
                        },
                    },
                },
            },
        },
        "source": {
            "name": "source",
            "redshift": {
                "kind": "fixed",
                "value": 0.6,
                "lower": None,
                "upper": None,
            },
            "components": {
                "light": {
                    "class_name": "Exponential",
                    "parameters": {
                        "centre_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.04,
                            "upper": -0.019999999999999997,
                        },
                        "centre_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.07,
                            "upper": 0.09,
                        },
                        "ell_comps_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.05,
                            "upper": 0.15000000000000002,
                        },
                        "ell_comps_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.15000000000000002,
                            "upper": 0.25,
                        },
                        "intensity": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 1.0,
                            "upper": 3.0,
                        },
                        "effective_radius": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.077,
                            "upper": 0.14300000000000002,
                        },
                    },
                },
            },
        },
    },
    "fit_mode": "fixed_template",
    "metadata": {
        "builder": "subhalo_model_spec_from_trial",
        "trial_case_id": "golden",
        "mass_profile_source": "hwo_slaps_forward_model",
    },
}


GOLDEN_LOCAL = {
    "model_type": "subhalo",
    "galaxies": {
        "lens": {
            "name": "lens",
            "redshift": {
                "kind": "fixed",
                "value": 0.2,
                "lower": None,
                "upper": None,
            },
            "components": {
                "mass": {
                    "class_name": "Isothermal",
                    "parameters": {
                        "centre_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.005,
                            "upper": 0.005,
                        },
                        "centre_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.005,
                            "upper": 0.005,
                        },
                        "einstein_radius": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.99,
                            "upper": 1.01,
                        },
                        "ell_comps_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.08,
                            "upper": 0.12000000000000001,
                        },
                        "ell_comps_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.02,
                            "upper": 0.02,
                        },
                    },
                },
                "subhalo": {
                    "class_name": "NFWSph",
                    "parameters": {
                        "centre_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.17,
                            "upper": 0.23,
                        },
                        "centre_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.13,
                            "upper": -0.07,
                        },
                        "kappa_s": {
                            "kind": "fixed",
                            "value": 0.01,
                            "lower": None,
                            "upper": None,
                        },
                        "scale_radius": {
                            "kind": "fixed",
                            "value": 0.2,
                            "lower": None,
                            "upper": None,
                        },
                    },
                },
            },
        },
        "source": {
            "name": "source",
            "redshift": {
                "kind": "fixed",
                "value": 0.6,
                "lower": None,
                "upper": None,
            },
            "components": {
                "light": {
                    "class_name": "Exponential",
                    "parameters": {
                        "centre_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": -0.04,
                            "upper": -0.019999999999999997,
                        },
                        "centre_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.07,
                            "upper": 0.09,
                        },
                        "ell_comps_0": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.05,
                            "upper": 0.15000000000000002,
                        },
                        "ell_comps_1": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.15000000000000002,
                            "upper": 0.25,
                        },
                        "intensity": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 1.0,
                            "upper": 3.0,
                        },
                        "effective_radius": {
                            "kind": "uniform",
                            "value": None,
                            "lower": 0.077,
                            "upper": 0.14300000000000002,
                        },
                    },
                },
            },
        },
    },
    "fit_mode": "local_search",
    "metadata": {
        "builder": "subhalo_model_spec_from_trial",
        "trial_case_id": "golden",
        "mass_profile_source": "hwo_slaps_forward_model",
    },
}


def test_legacy_model_spec_payloads_match_baseline_literals():
    """Pin smooth, fixed-template, and local-search payloads byte-stably."""
    config = _config()
    trial = SubhaloTrial(
        case_id="golden",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.2, -0.1),
        model="NFW",
        profile_class="NFWSph",
        lens_redshift=0.2,
        source_redshift=0.6,
        kappa_s=0.01,
        scale_radius_arcsec=0.2,
    )
    assert smooth_model_spec_from_config(config).to_dict() == GOLDEN_SMOOTH
    assert subhalo_model_spec_from_trial(
        config,
        trial,
        fit_mode="fixed_template",
    ).to_dict() == GOLDEN_FIXED
    assert subhalo_model_spec_from_trial(
        config,
        trial,
        fit_mode="local_search",
    ).to_dict() == GOLDEN_LOCAL


def test_fixed_point_uses_exact_clipped_prior_targets():
    """Evaluate C1 at the exact clipped slope target."""
    config = deepcopy(_config())
    mass = config["lensing"]["lens_galaxy"]["mass"]
    mass["type"] = "PowerLaw"
    mass["slope"] = 1.21
    priors = {
        "lens_slope_sigma": 0.2,
    }
    runner = _StubRunner()
    NonlinearMetricValidator(runner).validate_case(
        _identity_dataset(),
        _metadata(),
        config,
        _trial(),
        fit_mode="freed",
        priors_config=priors,
        mass_context=build_mass_mapping_context(config),
    )
    instance = runner.analysis.last_instance
    assert instance.galaxies.lens.mass.slope == 1.21

    spec = fixed_point_model_spec_from_trial(
        config,
        _trial(),
        priors_config=priors,
    )
    model = autofit_model_from_spec(spec)
    assert model.prior_count == 0
    pinned = model.instance_from_prior_medians()
    assert pinned.galaxies.lens.mass.slope == 1.21


def test_smooth_reuse_rejects_mismatched_analysis_key():
    """Reject a denominator summary from a different smooth analysis."""
    smooth = NonlinearFitSummary(
        model_role="smooth",
        fit_mode="smooth",
        status="success",
        log_likelihood_max=-10.0,
        analysis_key="wrong-key",
    )
    with pytest.raises(ValueError, match="wrong-key.*expected smooth"):
        NonlinearMetricValidator(_StubRunner()).validate_case(
            _identity_dataset(),
            _metadata(),
            _config(),
            _trial(),
            smooth_result=smooth,
        )


def test_smooth_analysis_key_is_distinct_and_stable_across_fit_modes():
    """Give smooth fits a stable key distinct from subhalo fit keys."""
    dataset = _identity_dataset()
    fixed_runner = _StubRunner()
    NonlinearMetricValidator(fixed_runner).validate_case(
        dataset,
        _metadata(),
        _config(),
        _trial(),
        fit_mode="fixed_template",
    )
    freed_runner = _StubRunner()
    NonlinearMetricValidator(freed_runner).validate_case(
        dataset,
        _metadata(),
        _config(),
        _trial(),
        fit_mode="freed",
        mass_context=build_mass_mapping_context(_config()),
    )
    fixed_smooth_key = fixed_runner.calls[0]["analysis_key"]
    freed_smooth_key = freed_runner.calls[0]["analysis_key"]
    assert fixed_smooth_key == freed_smooth_key
    assert fixed_smooth_key != fixed_runner.calls[1]["analysis_key"]
    assert freed_smooth_key != freed_runner.calls[1]["analysis_key"]


class _RecoveryFailureRunner(_StubRunner):
    """Return a successful fit while surfacing a callback warning."""

    def run_model(self, **kwargs):
        summary = super().run_model(**kwargs)
        callback = kwargs.get("result_callback")
        if callback is not None:
            try:
                callback(SimpleNamespace(), kwargs["model"])
            except Exception as exc:
                summary.warnings.append(f"result_callback failed: {exc}")
        return summary


def test_recovery_extraction_failure_is_visible():
    """Flag missing freed recovery while retaining successful fit status."""
    runner = _RecoveryFailureRunner()
    result = NonlinearMetricValidator(runner).validate_case(
        _identity_dataset(),
        _metadata(),
        _config(),
        _trial(),
        fit_mode="freed",
        mass_context=build_mass_mapping_context(_config()),
    )
    assert result.subhalo_fit.status == "success"
    assert any(
        "result_callback failed" in warning
        for warning in result.subhalo_fit.warnings
    )
    assert "recovery_extraction_failed" in result.quality_flags
    assert result.subhalo_recovery is None
