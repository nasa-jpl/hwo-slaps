"""Image geometry is Cartesian; smooth cell centres and interpolation knots differ."""

import json

import numpy as np
import pytest

from hwoslaps.inference.api import prepare_case
from hwoslaps.inference.settings import FitSpec

pytestmark = pytest.mark.backend
DEVICES = [False, pytest.param(True, marks=pytest.mark.xtx_gpu)]


def _samples(shape=(8, 10)):
    rows, cols = np.indices(shape, dtype=float)
    sb = np.exp(-0.5 * (((rows - 2.7) / 1.1) ** 2 + ((cols - 5.6) / 1.4) ** 2))
    return sb / (0.2**2 * sb.sum())


@pytest.mark.parametrize("gpu", DEVICES, ids=["cpu", "gpu"])
@pytest.mark.parametrize("free_geometry", [False, True], ids=["fixed-geometry", "free-geometry"])
def test_image_profile_smooth_centre_derivative_matches_independent_spline(gpu, free_geometry):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.scene.image_profile import ImageLightProfile

    ensure_jax_x64()
    assert jax.default_backend() == ("gpu" if gpu else "cpu")
    # The even template puts the exact sky centre strictly inside a bilinear cell.
    centre = np.array([0.3, -0.2])
    grid = al.Grid2DIrregular(values=np.array([centre, centre + [0.031, 0.047], centre + [-0.023, 0.012]]))
    weights = np.array([1.0, -0.7, 0.4])
    truth = np.array([*centre, 37.3, 0.92, 1.15]) if free_geometry else np.array([1.15])

    def scalar(parameters, xp):
        y, x, rotation, size, flux = parameters if free_geometry else (*centre, 37.3, 0.92, parameters[0])
        profile = ImageLightProfile(centre=(y, x), rotation_deg=rotation, pixel_scale_arcsec=0.2,
                                   sb=_samples(), total_flux=1.7, flux_scale=flux, size_scale=size)
        return xp.asarray(weights) @ xp.asarray(profile.image_2d_from(grid=grid, xp=xp).array)

    value, gradient = jax.jit(jax.value_and_grad(lambda p: scalar(p, jnp)))(truth)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert float(value) == pytest.approx(scalar(truth, np), rel=1e-13, abs=1e-13)
    differences = []
    for index in range(len(truth)):
        step = np.zeros(len(truth))
        step[index] = 1e-6
        differences.append((scalar(truth + step, np) - scalar(truth - step, np)) / 2e-6)
    np.testing.assert_allclose(gradient, differences, rtol=2e-7, atol=1e-9)


@pytest.mark.parametrize("gpu", DEVICES, ids=["cpu", "gpu"])
def test_image_interpolation_knot_retains_distinct_one_sided_slopes(gpu):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.scene.image_profile import ImageLightProfile

    ensure_jax_x64()
    assert jax.default_backend() == ("gpu" if gpu else "cpu")
    # An odd template puts the centre on a real interpolation knot. The Cartesian
    # correction must retain its bilinear slope jump, rather than smooth it away.
    grid = al.Grid2DIrregular(values=np.array([[0.0, 0.0]]))
    sb = _samples((9, 11))

    def value(y, xp):
        profile = ImageLightProfile(centre=(y, 0.0), rotation_deg=0.0, pixel_scale_arcsec=0.2,
                                   sb=sb, total_flux=1.0, flux_scale=1.0, size_scale=1.0)
        return xp.sum(profile.image_2d_from(grid=grid, xp=xp).array)

    step = 1e-7
    centre = value(0.0, np)
    left = (centre - value(-step, np)) / step
    right = (value(step, np) - centre) / step
    derivative = float(jax.jit(jax.grad(lambda y: value(y, jnp)))(0.0))
    assert np.isfinite(derivative)
    assert derivative == pytest.approx(left, rel=2e-8, abs=1e-9)
    assert abs(left - right) > 0.1
    assert abs(derivative - 0.5 * (left + right)) > 0.05


@pytest.mark.parametrize("gpu", DEVICES, ids=["cpu", "gpu"])
def test_image_public_objective_exact_smooth_centre_gradient(gpu, prepared_forecast_factory, tmp_path):
    import jax
    from hwoslaps.simulation import simulate

    assert jax.default_backend() == ("gpu" if gpu else "cpu")
    asset = tmp_path / "smooth-centre.npz"
    np.savez(asset, sb=_samples(), pixel_scale_arcsec=np.array(0.2),
             metadata_json=np.array(json.dumps({"format_version": 1, "provenance": {"test": "smooth-centre"}})))
    prepared = prepared_forecast_factory({"scene": {"grid": {"shape": [41, 41], "over_sample_size": 1},
        "source": {"light": {"light": {"ell_comps": [0.0, 0.0]}}},
        "lens": {"light": {"glow": {"type": "Image", "centre": [0.0, 0.0], "asset_path": str(asset),
            "total_flux": 1.7, "flux_scale": 1.15, "size_scale": 0.92, "rotation_deg": 37.3}}}},
        "forecast": {"nuisances": {"fixed": ["lens.mass.*", "source.*", "lens.light.glow.flux_scale",
            "lens.light.glow.size_scale", "lens.light.glow.rotation_deg"], "background_offset": False}}})
    trial = prepared.hypothesis(1e8, (0.4, -0.6))
    observation = simulate(prepared, subhalo=trial, noise_seed=None)
    case = prepare_case(prepared, trial, observation, fit=FitSpec(mode="fixed_template"), use_jax=True)
    independent = prepare_case(prepared, trial, observation, fit=FitSpec(mode="fixed_template"), use_jax=False)
    model = case.model("smooth")
    assert len(model.parameter_names) == 2 and all(".centre." in name for name in model.parameter_names)
    objective = case.objective("smooth")
    point = (model.truth - objective.lower) / (objective.upper - objective.lower)
    value, gradient = objective.value_and_gradient(point)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert value == pytest.approx(0.5 * independent.chi_squared("smooth", model.truth), rel=1e-10, abs=1e-10)
    for step_size in (1e-5, 3e-6):
        differences = []
        for index in range(len(point)):
            step = np.zeros(len(point))
            step[index] = step_size
            differences.append((independent.chi_squared("smooth", objective.to_physical(point + step))
                                - independent.chi_squared("smooth", objective.to_physical(point - step)))
                               / (4 * step_size))
        np.testing.assert_allclose(gradient, differences, rtol=2e-6, atol=1e-6)
