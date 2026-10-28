"""The expected observation of a rendered scene: convolution, light planes, the product, its draws."""

import numpy as np
import pytest
from scipy.signal import convolve2d

from hwoslaps.instrument import Detector
from hwoslaps.observation.expected import Exposure, convolve_light
from hwoslaps.observation.noise import draw_noisy_adu
from hwoslaps.observation.observation import observe
from hwoslaps.optics.kernels import DetectorPSF, KernelBinding
from hwoslaps.scene.builder import build_scene, native_sampling_variation
from hwoslaps.scene.cosmology import Cosmology, parse_cosmology
from hwoslaps.scene.spec import LightGroup, parse_scene

pytestmark = pytest.mark.backend

PIXEL_SCALE = 0.1
SOURCE = {"source": LightGroup(plane="source", sed=None, components=("disk",))}
EXPOSURE = Exposure(Detector(1.0, 0.2, 0.002), exposure_time_s=900.0, sky_rate_e_per_s=1.0)
SOURCE_LIGHT = {"type": "Exponential", "centre": [-0.03, 0.08], "ell_comps": [0.14516129, 0.25142673],
                "intensity": 2.0, "effective_radius": 0.11}


def _gaussian_kernel(size, width):
    axis = np.arange(size) - size // 2
    values = np.exp(-0.5 * (axis[:, None] ** 2 + axis[None, :] ** 2) / width ** 2)
    return values / values.sum()


def _binding(values, groups=("source",), pixel_scale=PIXEL_SCALE):
    return KernelBinding.uniform(DetectorPSF.from_array(values, pixel_scale, normalize=False), groups)


def _scene(*, lens_light):
    mapping = {
        "grid": {"shape": [32, 32], "pixel_scale_arcsec": PIXEL_SCALE, "over_sample_size": 4},
        "lens": {"redshift": 0.2,
                 "mass": {"main": {"type": "Isothermal", "centre": [0.0, 0.0], "einstein_radius": 1.0,
                                   "ell_comps": [0.1, 0.0]}},
                 "light": ({"bulge": {"type": "Exponential", "centre": [0.0, 0.0], "ell_comps": [0.05, 0.0],
                                      "intensity": 0.5, "effective_radius": 0.4}} if lens_light else {})},
        "source": {"redshift": 0.6, "light": {"disk": SOURCE_LIGHT}},
        "subhalo": {"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.0, "h": None}},
    }
    return build_scene(parse_scene(mapping), Cosmology(parse_cosmology({"name": "Planck15"})), subhalo=None)


def test_point_source_is_spread_by_the_kernel_about_its_pixel():
    image = np.zeros((7, 7))
    image[3, 3] = 1.0
    kernel = np.array([[0.0, 0.05, 0.0], [0.1, 0.5, 0.2], [0.0, 0.15, 0.0]])

    rates = convolve_light({"source": image}, SOURCE, _binding(kernel), PIXEL_SCALE)

    expected = np.zeros((7, 7))
    expected[2:5, 2:5] = kernel
    assert list(rates) == ["source"]
    np.testing.assert_allclose(rates["source"], expected, rtol=0.0, atol=1e-12)
    assert float(rates["source"].sum()) == pytest.approx(1.0, abs=1e-12)


def test_planes_are_convolved_separately_and_summed():
    rng = np.random.default_rng(20261005)
    images = {key: rng.random((32, 32)) for key in ("lens", "source:a", "source:b")}
    groups = {"lens": LightGroup(plane="lens", sed=None, components=("bulge",)),
              "source:a": LightGroup(plane="source", sed=None, components=("a",)),
              "source:b": LightGroup(plane="source", sed=None, components=("b",))}
    lens_kernel, source_kernel = _gaussian_kernel(7, 0.8), _gaussian_kernel(7, 1.6)
    source_psf = DetectorPSF.from_array(source_kernel, PIXEL_SCALE, normalize=False)
    binding = KernelBinding.from_groups({"lens": DetectorPSF.from_array(lens_kernel, PIXEL_SCALE, normalize=False),
                                         "source:a": source_psf, "source:b": source_psf})
    assert len(binding.kernels) == 2

    rates = convolve_light(images, groups, binding, PIXEL_SCALE)

    assert list(rates) == ["lens", "source"]
    for plane, image, kernel in (("lens", images["lens"], lens_kernel),
                                 ("source", images["source:a"] + images["source:b"], source_kernel)):
        reference = convolve2d(image, kernel, mode="same")
        np.testing.assert_allclose(rates[plane], reference, rtol=1e-12, atol=1e-14 * reference.max(),
                                   err_msg=plane)

    def through_source_kernel(light, light_groups):
        binding = KernelBinding.uniform(source_psf, list(light_groups))
        return convolve_light(light, light_groups, binding, PIXEL_SCALE)

    source_groups = {key: groups[key] for key in ("source:a", "source:b")}
    assert list(through_source_kernel({key: images[key] for key in source_groups}, source_groups)) == ["source"]
    # Groups sharing a kernel are summed and convolved once: the bytes of one convolution of their sum,
    # which convolving each group and summing the results does not reproduce.
    once = through_source_kernel({"source": images["source:a"] + images["source:b"]},
                                 {"source": LightGroup(plane="source", sed=None, components=("a", "b"))})["source"]
    each = [through_source_kernel({key: images[key]}, {key: groups[key]})["source"] for key in source_groups]
    assert not np.array_equal(each[0] + each[1], once)
    np.testing.assert_array_equal(rates["source"], once)


def test_convolved_rates_below_round_off_are_refused():
    y, x = np.indices((64, 64), dtype=float)
    blob = np.exp(-0.5 * (((y - 30.0) / 2.0) ** 2 + ((x - 33.0) / 2.5) ** 2))
    compact = np.where(blob > 1.0e-3, blob, 0.0)

    rate = convolve_light({"source": compact}, SOURCE, _binding(_gaussian_kernel(21, 3.3)), PIXEL_SCALE)["source"]

    assert np.all(np.isfinite(rate))
    assert float(rate.min()) >= -1.0e-10 * float(np.abs(rate).max())
    assert np.all(EXPOSURE.noise_map_adu(rate) > 0.0)
    negative = np.zeros((9, 9))
    negative[4, 4] = -0.5
    with pytest.raises(ValueError, match="round-off") as caught:
        convolve_light({"source": negative}, SOURCE, _binding(np.array([[1.0]])), PIXEL_SCALE)
    assert "source-plane" in str(caught.value)


def test_kernel_sampling_must_match_the_grid():
    image = np.ones((9, 9))
    with pytest.raises(ValueError, match="arcsec per pixel"):
        convolve_light({"source": image}, SOURCE, _binding(np.array([[1.0]]), pixel_scale=0.2), PIXEL_SCALE)
    rates = convolve_light({"source": image}, SOURCE, _binding(np.array([[1.0]]), pixel_scale=PIXEL_SCALE + 1e-13),
                           PIXEL_SCALE)
    np.testing.assert_array_equal(rates["source"], image)


@pytest.mark.parametrize("lens_light", [False, True], ids=["source-only", "lens-light"])
def test_expected_observation_is_seedless_and_read_only(lens_light, monkeypatch):
    scene = _scene(lens_light=lens_light)
    binding = KernelBinding.uniform(DetectorPSF.from_array(_gaussian_kernel(5, 1.0), PIXEL_SCALE, normalize=False),
                                    list(scene.light_groups))

    sampling = {key: 0.375 + index / 100 for index, key in enumerate(scene.light_groups)}

    def forbidden(*args, **kwargs):
        raise AssertionError("expected observation attempted a random draw")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr("hwoslaps.observation.observation.draw_noisy_adu", forbidden)
    observation = observe(scene, binding, EXPOSURE, config_digest=None, photometry=None, sampling=sampling)

    assert observation.kind == "expected"
    assert observation.data_adu is observation.expected_adu
    assert observation.noise_seed is None and observation.subhalo is None
    assert observation.psfs.single is binding.single
    by_plane = observation.light_rate_by_plane_e_per_s
    if lens_light:
        assert list(by_plane) == ["lens", "source"]
        np.testing.assert_array_equal(observation.light_rate_e_per_s, by_plane["lens"] + by_plane["source"])
    else:
        assert list(by_plane) == ["source"]
        assert by_plane["source"] is observation.light_rate_e_per_s
    rate = observation.light_rate_e_per_s
    mean = ((rate * 900.0 + 900.0) + 0.002 * 900.0) / 1.0
    variance = (np.maximum(rate, 0.0) * 900.0 + 0.002 * 900.0) + 900.0 + 0.2**2
    np.testing.assert_allclose(observation.expected_adu, mean, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(observation.noise_map_adu, np.sqrt(variance), rtol=0.0, atol=1e-12)
    for array in (observation.data_adu, observation.noise_map_adu, observation.light_rate_e_per_s,
                  *by_plane.values()):
        assert array.shape == (32, 32)
        with pytest.raises(ValueError):
            array[0, 0] = 1.0
    with pytest.raises(TypeError):
        by_plane["lens"] = observation.light_rate_e_per_s
    assert set(observation.sampling) == set(scene.light_groups)
    assert observation.sampling == sampling
    source_sampling = sampling["source"]
    sampling["source"] = 99.0
    assert observation.sampling["source"] == source_sampling
    assert all(np.isfinite(value) and value >= 0.0 for value in observation.sampling.values())
    with pytest.raises(TypeError):
        observation.sampling["source"] = 0.0


def test_draw_moves_only_detector_noise():
    scene = _scene(lens_light=False)
    sampling = native_sampling_variation(scene)
    expected = observe(scene, _binding(_gaussian_kernel(5, 1.0)), EXPOSURE, config_digest=None,
                       photometry=None, sampling=sampling)

    first, again, other = expected.draw(5), expected.draw(5), expected.draw(6)

    np.testing.assert_array_equal(first.data_adu, again.data_adu)
    np.testing.assert_array_equal(first.data_adu, draw_noisy_adu(expected.counts_e(), EXPOSURE, 5))
    assert not np.array_equal(first.data_adu, other.data_adu)
    for noisy, seed in ((first, 5), (again, 5), (other, 6)):
        assert (noisy.kind, noisy.noise_seed) == ("noisy", seed)
        assert noisy.expected_adu is expected.expected_adu
        assert noisy.noise_map_adu is expected.noise_map_adu
        assert noisy.light_rate_e_per_s is expected.light_rate_e_per_s
        assert noisy.sampling == expected.sampling
        assert not noisy.data_adu.flags.writeable
    with pytest.raises(ValueError, match="noisy observation"):
        first.draw(7)
    for seed in (-1, True, 1.5, "1"):
        with pytest.raises(ValueError, match="seed"):
            expected.draw(seed)


@pytest.mark.parametrize("sampling, message", [
    ({}, "sampling covers"),
    ({"source": 0.0, "lens": 0.0}, "sampling covers"),
    ({"source": -0.01}, "sampling\\['source'\\]"),
    ({"source": np.nan}, "sampling\\['source'\\]"),
    ({"source": np.inf}, "sampling\\['source'\\]"),
    ({"source": True}, "sampling\\['source'\\]"),
])
def test_fiducial_sampling_must_cover_groups_with_finite_non_negative_values(sampling, message):
    scene = _scene(lens_light=False)
    with pytest.raises(ValueError, match=message):
        observe(scene, _binding(_gaussian_kernel(5, 1.0)), EXPOSURE, config_digest=None,
                photometry=None, sampling=sampling)
