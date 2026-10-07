"""Reference and JAX rendering conventions at the prepared forecast boundary."""

from copy import deepcopy

import numpy as np
import pytest

pytestmark = pytest.mark.backend
SCENES = ("matched", "kernel_mismatch", "image", "lens_light", "two_source_components")


def scene_mapping(mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel):
    if scenario == "kernel_mismatch":
        kernel = tiny_gaussian_kernel.copy()
        kernel[3, 3] *= 1.1
        path = tmp_path / "model.npy"
        np.save(path, kernel / kernel.sum())
        mapping["psf"]["model"] = {"kind": "kernel", "path": str(path), "pixel_scale_arcsec": 0.05}
    elif scenario == "image":
        mapping["scene"]["source"]["light"] = {"light": {"type": "Image", "asset_path": str(image_asset),
            "centre": [-0.03, 0.08], "flux_scale": 1.0, "size_scale": 1.2, "rotation_deg": 10.0, "total_flux": 1.0}}
    elif scenario == "lens_light":
        mapping["scene"]["lens"]["light"] = {"bulge": {"type": "Exponential", "centre": [0.0, 0.0],
            "ell_comps": [0.07, 0.03], "intensity": 0.8, "effective_radius": 0.4}}
    elif scenario == "two_source_components":
        second = deepcopy(mapping["scene"]["source"]["light"]["light"])
        second.update(centre=[0.04, -0.06], effective_radius=0.18, intensity=0.3, ell_comps=[0.08, 0.03])
        mapping["scene"]["source"]["light"]["second"] = second
    return mapping


def assert_engines(mapping):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    with prepare_forecast(mapping) as reference, prepare_forecast(mapping, execution=Execution(engine="jax")) as jax:
        expected = forecast(reference, masses_msun=[1.0e8, 3.0e8])
        actual = forecast(jax, masses_msun=[1.0e8, 3.0e8])
    for name in ("fisher_raw", "fisher_profiled", "amplitude_hat", "amplitude_spurious"):
        if getattr(expected, name) is not None:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=5.0e-6, atol=0.0)
    return expected, actual


@pytest.mark.parametrize("scenario", SCENES)
def test_jax_engine_matches_reference(minimal_mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel, monkeypatch):
    if scenario == "kernel_mismatch":
        from hwoslaps.optics import providers
        from hwoslaps.optics.optical_psf import OpticalPSF

        def unexpected_optical_construction(*args, **kwargs):
            raise AssertionError("external detector kernels must not construct optical PSFs or pupils")

        monkeypatch.setattr(OpticalPSF, "__init__", unexpected_optical_construction)
        monkeypatch.setattr(providers, "build_pupil", unexpected_optical_construction)
    results = assert_engines(scene_mapping(minimal_mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel))
    if scenario == "kernel_mismatch":
        for result in results:
            assert np.all(np.isfinite(result.q_mismatch))
            assert np.any(result.q_spurious > 0.), "external kernel mismatch must have a spurious response"


@pytest.mark.xtx_gpu
@pytest.mark.parametrize("scenario", SCENES)
def test_jax_gpu_engine_matches_reference(minimal_mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel):
    assert_engines(scene_mapping(minimal_mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel))


@pytest.mark.parametrize("engine", ["reference", "jax"])
def test_mass_sequence_rows_equal_fresh_evaluations(minimal_mapping, engine):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    execution = Execution(engine=engine)
    masses = [3.0e8, 1.0e8, 3.0e8]
    with prepare_forecast(minimal_mapping, execution=execution) as prepared:
        rows = forecast(prepared, masses_msun=masses)
    for index, mass in enumerate(masses):
        with prepare_forecast(minimal_mapping, execution=execution) as fresh:
            expected = forecast(fresh, masses_msun=[mass])
        np.testing.assert_array_equal(rows.fisher_profiled[index], expected.fisher_profiled[0])


def test_light_evaluator_refuses_unsupported_profile():
    import autogalaxy as ag
    from hwoslaps.fisher.engines.jax_profiles import build_light_evaluator

    with pytest.raises(ValueError, match="Gaussian"):
        build_light_evaluator(ag.lp.Gaussian(), np.array([[0.1, 0.1], [0.2, 0.3]]))


def test_fast_fft_length_is_smallest_seven_smooth_bound():
    from hwoslaps.fisher.engines.fft import next_fast_length

    def smooth(value):
        for factor in (2, 3, 5, 7):
            while value % factor == 0:
                value //= factor
        return value == 1
    for target in [1, 11, 17, 31, 73, 107, 257, 499]:
        expected = next(value for value in range(target, 2 * target + 1) if smooth(value))
        assert next_fast_length(target) == expected


def test_padded_fft_convolution_matches_direct_real_space():
    from scipy.signal import convolve2d
    from hwoslaps.fisher.engines.fft import convolution_fft_shape

    image = np.random.default_rng(31).normal(size=(11, 13))
    kernel = np.array([[0.0, 0.1, 0.0], [0.2, 0.4, 0.2], [0.0, 0.1, 0.0]])
    shape = convolution_fft_shape(image.shape, kernel.shape)
    full = np.fft.irfft2(np.fft.rfft2(image, s=shape) * np.fft.rfft2(kernel, s=shape), s=shape)
    actual = full[1:12, 1:14]
    np.testing.assert_allclose(actual, convolve2d(image, kernel, mode="same"), rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("tiny_support", [False, True])
def test_image_evaluator_refuses_reference_convention_drift(image_asset, tiny_support):
    import autolens as al
    from hwoslaps.fisher.engines.jax_profiles import build_light_evaluator, verification_points
    from hwoslaps.scene.image_source import load_image_asset
    from hwoslaps.scene.image_profile import ImageLightProfile

    class ShiftedConvention(ImageLightProfile):
        def image_2d_from(self, grid, **kwargs):
            return 1.01 * super().image_2d_from(grid=grid, **kwargs)

    if tiny_support:
        samples = np.zeros((8, 8))
        samples[3, 3] = 1.0e16
        profile = ShiftedConvention(centre=(0.1, 0.2), rotation_deg=19.0, pixel_scale_arcsec=1.0e-8,
                                    sb=samples, total_flux=1.0, flux_scale=1.0, size_scale=1.0)
        macro = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        old_points = np.random.default_rng(0).uniform([-1.0, -1.0], [1.0, 1.0], size=(128, 2))
        assert not np.any(np.asarray(profile.image_2d_from(grid=al.Grid2DIrregular(values=old_points))))
        points = verification_points(macro, [profile])
        assert np.any(np.asarray(profile.image_2d_from(grid=al.Grid2DIrregular(values=points))))
    else:
        profile = ShiftedConvention.from_asset(load_image_asset(image_asset), centre=(0.0, 0.0), rotation_deg=0.0,
                                              total_flux=1.0, flux_scale=1.0, size_scale=1.0)
        points = np.array([[0.0, 0.0], [0.01, 0.02], [-0.01, 0.0]])
    with pytest.raises(ValueError, match="convention drift"):
        build_light_evaluator(profile, points)


@pytest.mark.parametrize("samples, r_min, r_max", [
    (samples, r_min, r_max)
    for r_min, r_max in ((1.0e-6, 2.0), (1.0e-5, 40.0), (3.0e-8, 300.0))
    for samples in (8192, 32768, 131072)
])
def test_affine_log_lookup_equals_general_interpolation_at_knots_and_neighbours(samples, r_min, r_max):
    import jax
    import jax.numpy as jnp
    from hwoslaps.fisher.engines.radial import affine_log_grid_parameters, interpolate_log_grid

    jax.config.update("jax_enable_x64", True)
    knots = np.log(np.logspace(np.log10(r_min), np.log10(r_max), samples))
    values = (np.sin(np.linspace(-2.0, 3.0, samples))
              + 0.17 * np.cos(np.linspace(0.0, 19.0, samples) ** 1.3))
    queries = np.concatenate((knots, np.nextafter(knots, -np.inf), np.nextafter(knots, np.inf),
                              0.5 * (knots[:-1] + knots[1:]),
                              [knots[0] - 1.0, knots[-1] + 1.0, -np.inf, np.inf, np.nan]))
    affine = affine_log_grid_parameters(knots)
    assert affine is not None
    assert affine[0] == float(knots[0]) and np.isfinite(affine[1]) and affine[1] > 0.0
    actual = interpolate_log_grid(jnp.asarray(queries), jnp.asarray(knots), jnp.asarray(values), affine)
    expected = jnp.interp(jnp.asarray(queries), jnp.asarray(knots), jnp.asarray(values))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2.0e-14, atol=2.0e-15, equal_nan=True)
    # Continuity alone cannot distinguish which interval owns an exact knot.
    # Its public derivative must use the right-hand slope, like jnp.interp.
    direct = lambda query: interpolate_log_grid(query, jnp.asarray(knots), jnp.asarray(values), affine)
    slopes = jax.jit(jax.vmap(jax.grad(direct)))(jnp.asarray(knots[1:-1]))
    np.testing.assert_allclose(np.asarray(slopes), np.diff(values)[1:] / np.diff(knots)[1:],
                               rtol=2.0e-14, atol=2.0e-15)

    # The original zero/tiny-value, scalar shape, knot dtype and query dtype contracts.
    small_knots = np.log(np.logspace(-6.0, np.log10(5.0), 129))
    small_values = np.array([0.0, *np.geomspace(1.0e-300, 1.0e-12, 128)])
    small_affine = affine_log_grid_parameters(small_knots)
    assert small_affine is not None
    small_direct = lambda query: interpolate_log_grid(query, jnp.asarray(small_knots),
                                                     jnp.asarray(small_values), small_affine)
    for dtype in (np.float32, np.float64):
        small_queries = jnp.asarray([small_knots[0], 0.5 * (small_knots[3] + small_knots[4]),
                                     small_knots[-1], -np.inf, np.inf, np.nan], dtype=dtype)
        scalar = small_direct(small_queries[1])
        assert np.asarray(scalar).shape == () and scalar.dtype == jnp.asarray(small_values).dtype
        small_expected = jnp.interp(small_queries, jnp.asarray(small_knots), jnp.asarray(small_values))
        for result in (small_direct(small_queries), jax.jit(small_direct)(small_queries),
                       jax.jit(jax.vmap(small_direct))(small_queries)):
            np.testing.assert_allclose(np.asarray(result), np.asarray(small_expected),
                                       equal_nan=True, rtol=2.0e-14, atol=0.0)
    irregular = np.array([0.0, 0.01, 0.05, 0.9, 1.0])
    assert affine_log_grid_parameters(irregular) is None
    duplicated_last = np.array([*small_knots[:-1], small_knots[-2], small_knots[-1]])
    perturbed = small_knots.copy()
    perturbed[len(perturbed) // 2] += 0.75 * np.median(np.diff(perturbed))
    for refused in (np.array([0.0, np.nan]), np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                    small_knots.astype(np.float32), small_knots[:1], small_knots.reshape(-1, 1),
                    duplicated_last, perturbed):
        assert affine_log_grid_parameters(refused) is None
    np.testing.assert_array_equal(
        np.asarray(interpolate_log_grid(jnp.array([0.03, 0.8]), jnp.asarray(irregular), jnp.arange(5.0), None)),
        np.asarray(jnp.interp(jnp.array([0.03, 0.8]), jnp.asarray(irregular), jnp.arange(5.0))))


def test_radial_extent_covers_far_domain_and_refuses_extrapolation():
    from hwoslaps.fisher.engines.radial import check_coverage, radial_grid

    coordinates = np.array([[1.0, 1.0], [-1.0, -1.0]])
    radial = radial_grid(coordinates, (0.0, 0.0), 100.0, samples=8192)
    assert radial.r_max > 100.0 + np.sqrt(2.0)
    check_coverage(radial, np.array([[0.0, 100.0]]), (0.0, 0.0))
    with pytest.raises(ValueError, match="radial"):
        check_coverage(radial, np.array([[0.0, 2.0 * radial.r_max]]), (0.0, 0.0))


def test_dense_covariance_uses_host_projection_and_matches_reference(minimal_mapping, tmp_path):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    minimal_mapping["scene"]["grid"]["shape"] = [11, 11]
    minimal_mapping["scene"]["grid"]["pixel_scale_arcsec"] = 0.15
    minimal_mapping["psf"]["truth"]["pixel_scale_arcsec"] = 0.15
    with prepare_forecast(minimal_mapping) as diagonal:
        sigma = diagonal.sigma_adu.reshape(-1)
    covariance = np.diag(sigma**2)
    covariance += 0.001 * (np.outer(sigma, sigma) - np.diag(sigma**2))
    path = tmp_path / "covariance.npy"
    np.save(path, covariance)
    minimal_mapping["forecast"]["noise_covariance"] = str(path)
    with prepare_forecast(minimal_mapping) as reference, prepare_forecast(minimal_mapping, execution=Execution(engine="jax")) as jax:
        expected = forecast(reference, masses_msun=[1.0e8])
        actual = forecast(jax, masses_msun=[1.0e8])
        assert actual.provenance["engine"]["projection"] == "dense"
        assert actual.provenance["noise_covariance"] == reference.record["file_digests"][str(path)]
    np.testing.assert_allclose(actual.fisher_profiled, expected.fisher_profiled, rtol=5.0e-6)


@pytest.mark.parametrize("geometry", ["off_plane_hypothesis", "other_plane_perturber"])
def test_jax_engine_refuses_geometry_it_cannot_render(minimal_mapping, geometry):
    from hwoslaps.fisher.api import Execution, prepare_forecast

    if geometry == "off_plane_hypothesis":
        minimal_mapping["scene"]["subhalo"] = {"type": "NFW", "redshift": 0.3,
                                                 "concentration": {"kind": "fixed", "value": 10.0}}
    else:
        minimal_mapping["scene"]["perturbers"] = {"halos": [{"type": "PointMass", "redshift": 0.3,
                                                            "mass_msun": 1.0e8, "centre": [0.1, 0.7]}]}
    with pytest.raises(ValueError, match="two-plane scene"):
        prepare_forecast(minimal_mapping, execution=Execution(engine="jax"))


@pytest.mark.xtx_gpu
def test_mass_ladder_reuses_compiled_shapes(minimal_mapping, caplog):
    import jax
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    minimal_mapping["forecast"]["positions"] = {"kind": "grid", "spacing_arcsec": 0.2, "half_width_arcsec": 0.6}
    with prepare_forecast(minimal_mapping, execution=Execution(engine="jax", batch_size=16)) as prepared:
        with jax.log_compiles(True), caplog.at_level("WARNING"):
            forecast(prepared, masses_msun=[1.0e8])
            assert any("Compiling " in record.getMessage() for record in caplog.records)
            caplog.clear()
            forecast(prepared, masses_msun=[3.0e8, 1.0e7, 8.0e8])
            assert not any("Compiling " in record.getMessage() for record in caplog.records)
