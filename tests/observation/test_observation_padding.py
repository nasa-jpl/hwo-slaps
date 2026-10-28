"""Small detector-footprint invariants for the native response and emitting sky."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import convolve2d

from hwoslaps.identity import json_ready
from hwoslaps.scene.rendering import RenderGeometry, render_scene_light
from hwoslaps.observation.expected import convolve_scene_light
from hwoslaps.scene.spec import GridSpec, LightGroup
from hwoslaps.optics.kernels import DetectorPSF, KernelBinding

pytestmark = pytest.mark.backend


def _kernel(shape, *, tilt=0.0):
    y, x = np.indices(shape, dtype=float)
    y, x = y - shape[0] // 2, x - shape[1] // 2
    values = np.exp(-0.5 * ((y - 0.6 + tilt * x) ** 2 / 2.3**2 + (x + 0.4) ** 2 / 2.0**2))
    return values / values.sum()


def padded_mapping(mapping, tmp_path, *, shape=(25, 31), off_image=False, unequal=False):
    mapping = deepcopy(mapping)
    truth = tmp_path / "padding-truth.npy"
    model = tmp_path / "padding-model.npy"
    np.save(truth, _kernel((9, 5), tilt=0.3))
    np.save(model, _kernel((3, 13), tilt=-0.4))
    mapping["scene"]["grid"] = {"shape": list(shape), "pixel_scale_arcsec": 0.1, "over_sample_size": 2}
    mapping["scene"]["lens"]["mass"]["mass"]["centre"] = [0.013, -0.021]
    mapping["scene"]["lens"]["light"] = {"glow": {"type": "Sersic", "centre": [1.4, 0.0] if off_image else [0.0, 0.0],
        "ell_comps": [0.0, 0.0], "intensity": 5.0 if off_image else 1.0,
        "effective_radius": 0.06 if off_image else 2.0, "sersic_index": 0.5}}
    mapping["scene"]["source"]["light"]["light"].update(effective_radius=0.7, intensity=1.0)
    mapping["psf"] = {"truth": {"kind": "kernel", "path": str(truth), "pixel_scale_arcsec": 0.1}}
    if unequal:
        mapping["psf"]["model"] = {"kind": "kernel", "path": str(model), "pixel_scale_arcsec": 0.1}
    mapping["forecast"]["positions"] = {"kind": "explicit", "positions_yx": [[0.411, -0.613]]}
    mapping["forecast"]["nuisances"] = {"fixed": ["*"], "background_offset": False}
    return mapping


@pytest.mark.parametrize("off_image", [False, True], ids=["broad-light", "off-image-emitter"])
def test_common_crop_preserves_lens_lensed_source_rates_and_noise(minimal_mapping, tmp_path, off_image):
    from hwoslaps.simulation import simulate

    small_mapping = padded_mapping(minimal_mapping, tmp_path, off_image=off_image)
    large_mapping = deepcopy(small_mapping)
    large_mapping["scene"]["grid"]["shape"] = [45, 51]
    small = simulate(small_mapping, subhalo=None, noise_seed=None)
    large = simulate(large_mapping, subhalo=None, noise_seed=None)
    crop = (slice(10, 35), slice(10, 41))
    assert small.grid.shape == (25, 31) and small.pixel_scale_arcsec == 0.1
    for plane in ("lens", "source"):
        np.testing.assert_allclose(small.light_rate_by_plane_e_per_s[plane],
                                   large.light_rate_by_plane_e_per_s[plane][crop], rtol=2e-12, atol=2e-14)
    for name in ("light_rate_e_per_s", "expected_adu", "noise_map_adu"):
        np.testing.assert_allclose(getattr(small, name), getattr(large, name)[crop], rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(small.light_rate_e_per_s,
                                  sum(small.light_rate_by_plane_e_per_s.values()))
    np.testing.assert_array_equal(small.counts_e(), small.exposure.counts_e(small.light_rate_e_per_s))
    if off_image:
        assert np.max(small.light_rate_by_plane_e_per_s["lens"][0]) > 0.01


def test_flat_light_is_preserved_to_the_detector_edge():
    import autoarray as aa

    grid = GridSpec((8, 11), 0.1, 2)
    group = LightGroup(plane="source", sed=None, components=("flat",))
    class FlatTracer:
        def image_2d_from(self, grid):
            return aa.Array2D(values=np.full(grid.mask.shape_native, 2.5), mask=grid.mask)
    scene = SimpleNamespace(spec=SimpleNamespace(grid=grid), pixel_scale_arcsec=0.1,
                            light_groups={"source": group}, tracer=FlatTracer())
    kernel = DetectorPSF.from_array(_kernel((9, 5)), 0.1, normalize=False)
    rates = convolve_scene_light(scene, KernelBinding.uniform(kernel, ["source"]))
    np.testing.assert_allclose(rates["source"], 2.5, rtol=2e-15, atol=2e-15)


def test_union_support_and_signed_wavefront_derivative_use_real_exterior(minimal_mapping, tmp_path):
    from hwoslaps.fisher.api import prepare_forecast

    mapping = padded_mapping(minimal_mapping, tmp_path, unequal=True)
    with prepare_forecast(mapping) as prepared:
        geometry = prepared.renderer.geometry
        assert geometry.padding == (4, 6) and geometry.shape == (33, 43)
        assert prepared.observation.grid.shape == (25, 31)
        _, images = render_scene_light(prepared.scene, geometry)
        derivative = np.arange(39, dtype=float).reshape(3, 13) - 19.0
        actual = prepared.renderer.derivative_adu(prepared.scene, prepared.psfs.model_kernels, [derivative])
        per_group = prepared.renderer.derivative_adu(prepared.scene, prepared.psfs.model_kernels,
            {key: derivative for key in prepared.scene.light_groups})
        np.testing.assert_allclose(actual, per_group, rtol=1e-13, atol=1e-8)
        total = images["lens"] + images["source"]
        reference = geometry.crop(convolve2d(total, derivative, mode="same"))
        np.testing.assert_allclose(actual, prepared.observation.exposure.signal_adu(reference),
                                   rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(prepared.renderer.mean_adu(prepared.scene, prepared.psfs.truth_kernels),
                                   prepared.observation.expected_adu, rtol=1e-13, atol=1e-11)


@pytest.mark.parametrize("gpu", [False, pytest.param(True, marks=pytest.mark.xtx_gpu)], ids=["cpu", "gpu"])
def test_padded_jax_forecast_and_nonlinear_model_share_reference_response(gpu, minimal_mapping, tmp_path):
    import jax
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast
    from hwoslaps.inference.api import prepare_case
    from hwoslaps.inference.settings import FitSpec

    assert jax.default_backend() == ("gpu" if gpu else "cpu")
    mapping = padded_mapping(minimal_mapping, tmp_path, unequal=True)
    mapping["forecast"]["nuisances"]["fixed"] = ["lens.mass.*", "source.*", "lens.light.glow.centre_*",
        "lens.light.glow.ell_comp_*", "lens.light.glow.effective_radius", "lens.light.glow.sersic_index"]
    with prepare_forecast(mapping, execution=Execution(engine="reference")) as reference, \
         prepare_forecast(mapping, execution=Execution(engine="jax")) as accelerated:
        left, right = forecast(reference, masses_msun=[1e8]), forecast(accelerated, masses_msun=[1e8])
        for name in ("fisher_raw", "fisher_profiled", "q_mismatch", "amplitude_hat", "amplitude_spurious"):
            np.testing.assert_allclose(getattr(left, name), getattr(right, name), rtol=5e-6, atol=1e-8, err_msg=name)
        trial = reference.hypothesis(1e8, (0.411, -0.613))
        case = prepare_case(reference, trial, reference.observation, fit=FitSpec(mode="fixed_template"), use_jax=True)
        independent = prepare_case(reference, trial, reference.observation, fit=FitSpec(mode="fixed_template"), use_jax=False)
        response = left.provenance["comparison_inputs"]["response"]
        assert json_ready(response) == json_ready(case.data.record["comparison_inputs"]["response"])
        assert response["operator"] == "native_binned_psf_padded_v1"
        assert response["light_integration"]["source"]["policy"] == "uniform_subpixel_mean"
        model = case.model("smooth")
        instance = case.autofit_models["smooth"].instance_from_vector(vector=model.truth.tolist())
        fitted = case.analysis.fit_from(instance)
        rates = reference.renderer.light_rate(reference.scene, reference.psfs.model_kernels)
        np.testing.assert_allclose(np.asarray(fitted.model_data), rates[case.data.mask], rtol=2e-12, atol=2e-13)
        assert case.data.record["shape"] == (25, 31)
        assert tuple(case.data.imaging.shape_native) == (33, 43)
        objective = case.objective("smooth")
        point = (model.truth - objective.lower) / (objective.upper - objective.lower)
        value, gradient = objective.value_and_gradient(point)
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        assert len(point) == 1
        step = np.array([1e-5])
        oracle = (independent.chi_squared("smooth", objective.to_physical(point + step))
                  - independent.chi_squared("smooth", objective.to_physical(point - step))) / 4e-5
        assert gradient[0] == pytest.approx(oracle, rel=2e-6, abs=1e-6)
