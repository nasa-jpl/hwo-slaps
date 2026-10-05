"""Resolved nuisance order and detector-mean derivatives."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.backend
P1 = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity" / "engine" / "p1_optical_matched.yaml"


@pytest.mark.parametrize("light_type", ["Exponential", "Image"])
def test_nuisance_order_follows_registry_and_fixed_patterns(light_type, minimal_mapping, image_asset):
    from hwoslaps.config.checks import ConfigError
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    lens_names = ("lens.mass.mass.centre_y", "lens.mass.mass.centre_x", "lens.mass.mass.einstein_radius",
                  "lens.mass.mass.ell_comp_1", "lens.mass.mass.ell_comp_2")
    if light_type == "Image":
        minimal_mapping["scene"]["source"]["light"]["light"] = {
            "type": "Image", "asset_path": str(image_asset), "centre": [-0.03, 0.08],
            "total_flux": 1.0, "rotation_deg": 12.0}
        source_names = ("source.light.light.centre_y", "source.light.light.centre_x",
                        "source.light.light.flux_scale", "source.light.light.size_scale",
                        "source.light.light.rotation_deg")
        minimal_mapping["forecast"]["nuisances"] = {"priors": {"source.light.light.flux_scale": 0.02}}
    else:
        source_names = ("source.light.light.centre_y", "source.light.light.centre_x",
                        "source.light.light.ell_comp_1", "source.light.light.ell_comp_2",
                        "source.light.light.intensity", "source.light.light.effective_radius")
    expected = lens_names + source_names + ("observation.background_offset_adu",)
    with prepare_forecast(minimal_mapping, execution=Execution(engine="reference")) as prepared:
        assert prepared.nuisances.names == expected
        if light_type == "Image":
            amplitude = prepared.nuisances.names.index("source.light.light.flux_scale")
            assert prepared.nuisances.parameters[amplitude].prior_sigma == 0.02
            np.testing.assert_array_equal(prepared.nuisances.prior_precision,
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2500.0, 0.0, 0.0, 0.0])
    minimal_mapping["forecast"]["nuisances"] = {"fixed": ["lens.mass.*", "source.light.*.centre_*"], "background_offset": False}
    with prepare_forecast(minimal_mapping, execution=Execution(engine="reference")) as prepared:
        assert prepared.nuisances.names == source_names[2:]
    if light_type == "Exponential":
        minimal_mapping["forecast"]["nuisances"] = {"fixed": ["*"], "background_offset": False, "wavefront": None}
        with prepare_forecast(minimal_mapping, execution=Execution(engine="reference")) as prepared:
            assert prepared.nuisances.names == ()
            assert prepared.nuisances.images.shape == (0, *prepared.mask.shape)
            assert prepared.nuisances.prior_precision.shape == (0,)
            result = forecast(prepared, masses_msun=[1.0e8])
            np.testing.assert_array_equal(result.fisher_profiled, result.fisher_raw)
    minimal_mapping["forecast"]["nuisances"]["fixed"] = ["typo.*"]
    with pytest.raises(ConfigError, match="matches no parameter"):
        prepare_forecast(minimal_mapping, execution=Execution(engine="reference"))


def test_steps_resolve_by_name_kind_and_default(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast

    minimal_mapping["forecast"]["nuisances"] = {"steps": {"position": 0.002, "lens.mass.mass.centre_y": 0.004,
                                                                   "amplitude": 0.03}}
    with prepare_forecast(minimal_mapping) as prepared:
        steps = {parameter.name: parameter.step for parameter in prepared.nuisances.parameters}
        assert steps["lens.mass.mass.centre_y"] == 0.004
        assert steps["lens.mass.mass.centre_x"] == 0.002
        assert steps["lens.mass.mass.einstein_radius"] == 0.001
        assert steps["source.light.light.intensity"] == 0.03
        assert steps["source.light.light.effective_radius"] == 0.0012


def test_amplitude_derivative_equals_unit_amplitude_image(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast

    with prepare_forecast(minimal_mapping) as prepared:
        index = prepared.nuisances.names.index("source.light.light.intensity")
        intensity = prepared.scene.spec.source.light[0].values["intensity"]
        expected = (prepared.mean_model_adu - prepared.observation.exposure.background_adu) / intensity
        np.testing.assert_allclose(prepared.nuisances.images[index], expected, rtol=1.0e-9,
                                   atol=1.0e-12 * np.max(np.abs(expected)))
        background = prepared.nuisances.names.index("observation.background_offset_adu")
        np.testing.assert_array_equal(prepared.nuisances.images[background], np.ones(prepared.mask.shape))


def test_steps_leaving_parameter_domain_raise(minimal_mapping):
    from hwoslaps.config.checks import ConfigError
    from hwoslaps.fisher.api import prepare_forecast

    minimal_mapping["forecast"]["nuisances"] = {"steps": {"source.light.light.intensity": 1.0}}
    with pytest.raises(ConfigError, match="intensity.*half-step"):
        prepare_forecast(minimal_mapping)


def test_wavefront_derivative_equals_full_render_difference():
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.optics.kernels import KernelBinding
    from hwoslaps.optics.wavefront import WavefrontMode

    with prepare_forecast(P1) as prepared:
        mode = WavefrontMode("zernikes", 4)
        provider = prepared.psfs.model.provider
        value = provider.coefficients.value(mode)
        plus = provider.kernel(provider.wavelengths_m[0], coefficients=provider.coefficients.replace(mode, value + 1.0))
        minus = provider.kernel(provider.wavelengths_m[0], coefficients=provider.coefficients.replace(mode, value - 1.0))
        groups = tuple(prepared.scene.light_groups)
        expected = (prepared.renderer.mean_adu(prepared.scene, KernelBinding.uniform(plus, groups))
                    - prepared.renderer.mean_adu(prepared.scene, KernelBinding.uniform(minus, groups))) / 2.0
        actual = prepared.nuisances.images[prepared.nuisances.names.index("psf.zernikes[4]")]
        np.testing.assert_allclose(actual, expected, rtol=1.0e-8, atol=1.0e-8 * np.max(np.abs(expected)))
