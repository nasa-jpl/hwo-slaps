"""Tests for the transformed clumpy fit-side light profile."""

from __future__ import annotations

import pickle

import autolens as al
import numpy as np
import pytest
import yaml

from hwoslaps.lensing.generator import _create_source_galaxy
from hwoslaps.modeling.nonlinear.autolens_model_builder import (
    autofit_model_from_spec,
    smooth_model_spec_from_config,
)
from hwoslaps.modeling.nonlinear.autolens_runner import (
    AutoLensFitRunner,
    NonlinearSearchSettings,
)
from hwoslaps.modeling.nonlinear.clumpy_profiles import (
    ClumpyTemplateContext,
    ClumpyTransformedSource,
)
from hwoslaps.psf.utils import make_pyauto_convolver, make_pyauto_kernel


def _config():
    """Load the canonical clumpy-source scene."""
    with open(
        "configs/scenes/scene2_clumpy.yaml",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


def test_clumpy_composite_matches_truth_source_image():
    """Match the generator source image exactly at truth parameters."""
    config = _config()
    truth = _create_source_galaxy(config["lensing"]["source_galaxy"])
    model = autofit_model_from_spec(smooth_model_spec_from_config(config))
    fitted = model.instance_from_prior_medians().galaxies.source
    grid = al.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.03)
    np.testing.assert_allclose(
        np.asarray(fitted.image_2d_from(grid)),
        np.asarray(truth.image_2d_from(grid)),
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_clumpy_transform_semantics_are_joint_and_analytic():
    """Scale all intensities, sizes, and relative clump offsets jointly."""
    context = ClumpyTemplateContext(
        host=(0.1, -0.2, 2.0, 0.1, 1.5),
        host_centre=(-0.03, 0.08),
        clumps=((0.07, -0.04, 0.0, 0.0, 3.0, 0.02, 1.0),),
        context_hash="test",
    )
    profile = ClumpyTransformedSource(
        centre=(0.2, -0.1),
        flux_scale=3.0,
        size_scale=2.0,
        host_ell_comps=(0.1, -0.2),
        host_intensity=2.0,
        host_effective_radius=0.1,
        host_sersic_index=1.5,
        template_context=context,
    )
    assert profile.host_profile.centre == pytest.approx((0.2, -0.1))
    assert profile.host_profile.intensity == pytest.approx(6.0)
    assert profile.host_profile.effective_radius == pytest.approx(0.2)
    clump = profile.clump_profiles[0]
    assert clump.centre == pytest.approx((0.34, -0.18))
    assert clump.intensity == pytest.approx(9.0)
    assert clump.effective_radius == pytest.approx(0.04)

    grid = al.Grid2D.uniform(shape_native=(13, 13), pixel_scales=0.03)
    direct = profile.host_profile.image_2d_from(grid)
    direct += clump.image_2d_from(grid)
    np.testing.assert_allclose(
        np.asarray(profile.image_2d_from(grid)),
        np.asarray(direct),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "mode,expected",
    [("rigid", 4), ("host_free", 9), ("fully_free", 19)],
)
def test_clumpy_mode_parameter_counts_and_fully_free_fields(mode, expected):
    """Count each parameterization and pin the four free clump fields."""
    spec = smooth_model_spec_from_config(
        _config(),
        clumpy_fit_parameterization=mode,
    )
    model = autofit_model_from_spec(spec)
    assert model.galaxies.source.prior_count == expected
    if mode == "fully_free":
        for name in ("clump_0", "clump_1", "clump_2"):
            component = spec.galaxies["source"].components[name]
            free = {
                parameter
                for parameter, prior in component.parameters.items()
                if prior.kind != "fixed"
            }
            assert free == {
                "centre_0",
                "centre_1",
                "intensity",
                "effective_radius",
            }
            assert component.parameters["ell_comps_0"].kind == "fixed"
            assert component.parameters["sersic_index"].kind == "fixed"


def test_clumpy_composite_runs_real_analysis_likelihood(tmp_path):
    """Evaluate a finite non-JAX AnalysisImaging likelihood."""
    config = _config()
    model = autofit_model_from_spec(smooth_model_spec_from_config(config))
    instance = model.instance_from_prior_medians()
    grid = al.Grid2D.uniform(shape_native=(15, 15), pixel_scales=0.08)
    tracer = al.Tracer(
        galaxies=[instance.galaxies.lens, instance.galaxies.source]
    )
    image = tracer.image_2d_from(grid)
    data = al.Array2D.no_mask(
        values=np.asarray(image.native),
        pixel_scales=0.08,
    )
    noise = al.Array2D.full(
        fill_value=0.1,
        shape_native=(15, 15),
        pixel_scales=0.08,
    )
    dataset = al.Imaging(
        data=data,
        noise_map=noise,
        psf=make_pyauto_convolver(
            make_pyauto_kernel(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                pixel_scales=0.08,
            )
        ),
        over_sample_size_lp=1,
    )
    runner = AutoLensFitRunner(NonlinearSearchSettings(), str(tmp_path))
    analysis = runner.make_analysis(dataset)
    assert np.isfinite(analysis.log_likelihood_function(instance))


def test_clumpy_composite_and_model_pickle_round_trip():
    """Pickle both an instantiated composite and its AutoFit model."""
    model = autofit_model_from_spec(smooth_model_spec_from_config(_config()))
    instance = model.instance_from_prior_medians().galaxies.source.light
    restored_instance = pickle.loads(pickle.dumps(instance))
    restored_model = pickle.loads(pickle.dumps(model))
    assert isinstance(restored_instance, ClumpyTransformedSource)
    assert restored_instance.template_context == instance.template_context
    assert restored_model.total_free_parameters == model.total_free_parameters
