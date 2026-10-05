"""Spectral PSF schema, runtime guards and optical-model basis ownership."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.config.schema import parse_config
from hwoslaps.optics.providers import CROSS_RULES, PSF_TABLE, build_model_psf, build_psf_provider, parse_psf


def _sampled(optical, count):
    result = {key: copy.deepcopy(value) for key, value in optical.items() if key != "wavelength_nm"}
    result["wavelength_samples"] = count
    return result


def _run_cross(psf, bandpass=None, scale=0.03, *, source_sed=True, lens_sed=True):
    light = {"type": "Exponential", "sed": {"kind": "flat_fnu"}}
    source, lens = copy.deepcopy(light), copy.deepcopy(light)
    if not source_sed:
        source["sed"] = None
    if not lens_sed:
        lens["sed"] = None
    values = {"psf": PSF_TABLE.read(psf, "psf"), "instrument": {"bandpass": bandpass},
              "scene": {"grid": {"pixel_scale_arcsec": scale},
                        "lens": {"light": {"bulge": lens}}, "source": {"light": {"disk": source}}}}
    for rule in CROSS_RULES:
        rule.check(values, "")


@pytest.mark.parametrize("count", [None, 0, -1, True, 1.5])
def test_optical_wavelength_keys_are_exclusive_and_sample_count_is_physical(p1_truth, count):
    mapping = {"truth": _sampled(p1_truth, count)}
    with pytest.raises(ConfigError):
        parse_psf(mapping)
    both = {"truth": {**p1_truth, "wavelength_samples": 2}}
    with pytest.raises(ConfigError, match="exactly one"):
        parse_psf(both)


@pytest.mark.parametrize("nodes", [(), (4e-7,), (6e-7, 4e-7), (4e-7, 4e-7), (np.nan, 6e-7), (False, 6e-7)])
def test_supplied_node_domains_are_checked_before_optical_construction(p1_truth, nodes):
    spec = parse_psf({"truth": _sampled(p1_truth, 2)}).truth
    with pytest.raises(ValueError):
        build_psf_provider(spec, pixel_scale_arcsec=0.03, wavelengths_m=nodes)
    with pytest.raises(ValueError, match="caller's bandpass nodes"):
        build_psf_provider(spec, pixel_scale_arcsec=0.03)


def test_chromatic_presence_and_shortest_node_sampling_rules(p1_truth, paper_pupil):
    band = {"kind": "top_hat", "min_nm": 450.0, "max_nm": 550.0, "throughput": 1.0}
    sampled = {"truth": _sampled(p1_truth, 2)}
    _run_cross(sampled, band)
    for arguments, path in [({"bandpass": None}, "instrument.bandpass"),
                            ({"bandpass": band, "source_sed": False}, "scene.source.light.disk.sed"),
                            ({"bandpass": band, "lens_sed": False}, "scene.lens.light.bulge.sed")]:
        with pytest.raises(ConfigError) as caught:
            _run_cross(sampled, **arguments)
        assert caught.value.path == path
    paper = {**_sampled(p1_truth, 21), "pupil": paper_pupil, "detector_oversampling": 3,
             "kernel_shape": [999, 999], "wavefront": {}}
    with pytest.raises(ConfigError, match="aliased kernel at 452.381") as caught:
        _run_cross({"truth": paper}, band, scale=0.00716)
    assert caught.value.path == "psf.truth"
    # The runtime uses supplied actual nodes, including when support came from a file.
    with pytest.raises(ValueError, match="aliased kernel at 452.381"):
        build_psf_provider(parse_psf({"truth": {**paper, "wavelength_samples": 1}}).truth,
                           pixel_scale_arcsec=0.00716, wavelengths_m=((450 + 100 / 42) / 1e9,))


def test_wavelength_overlay_switches_the_real_config_alternative(minimal_mapping, p1_truth):
    mapping = copy.deepcopy(minimal_mapping)
    mapping["psf"] = {"truth": copy.deepcopy(p1_truth)}
    base = parse_config(mapping)
    sampled = base.replace({"psf": {"truth": {"wavelength_nm": None, "wavelength_samples": 1}},
                            "instrument": {"bandpass": {"kind": "top_hat", "min_nm": 450., "max_nm": 550.,
                                                         "throughput": 1.}},
                            "scene": {"source": {"light": {"light": {"sed": {"kind": "flat_fnu"}}}}}})
    assert sampled.psf.truth.wavelength_m is None and sampled.psf.truth.wavelength_samples == 1
    assert base.psf.truth.wavelength_m == 500. / 1e9 and base.psf.truth.wavelength_samples is None
    back = sampled.replace({"psf": {"truth": {"wavelength_samples": None, "wavelength_nm": 500.}}})
    assert back.psf.truth == base.psf.truth


def test_optical_model_keys_and_basis_come_from_the_model_pupil(minimal_mapping, p1_truth, circular_pupil):
    circular = {**p1_truth, "pupil": circular_pupil, "wavefront": {}}
    hexagonal = {**p1_truth, "wavefront": {}}
    mapping = copy.deepcopy(minimal_mapping)
    mapping["forecast"]["nuisances"] = {"wavefront": {
        "modes": {"segment_hexikes": {"segments": [0], "nolls": [4]}}}}
    mapping["psf"] = {"truth": circular, "model": hexagonal}
    valid = parse_config(mapping)
    assert valid.psf.model.pupil.kind == "hex_segmented"
    mapping["psf"] = {"truth": hexagonal, "model": circular}
    with pytest.raises(ConfigError, match="has no segments") as caught:
        parse_config(mapping)
    assert caught.value.path == "forecast.nuisances.wavefront.modes.segment_hexikes"
    with pytest.raises(ConfigError, match="wavelength keys"):
        parse_psf({"truth": p1_truth, "model": {**p1_truth, "wavelength_nm": 600.}})
    with pytest.raises(ConfigError, match="wavelength keys"):
        parse_psf({"truth": p1_truth, "model": _sampled(p1_truth, 1)})
    with pytest.raises(ConfigError, match="needs an optical truth"):
        parse_psf({"truth": minimal_mapping["psf"]["truth"], "model": p1_truth})


@pytest.mark.backend
@pytest.mark.parametrize("sampled", [False, True], ids=["mono", "two-nodes"])
def test_optical_model_with_truth_keys_is_the_matched_limit(circular_pupil, sampled):
    truth = {"kind": "optical", "pupil": circular_pupil, "focal_length_m": 10., "kernel_shape": [11, 11],
             "detector_oversampling": 3, "wavefront": {"zernikes": {4: 8.}}}
    truth.update({"wavelength_samples": 2} if sampled else {"wavelength_nm": 500.})
    specs = parse_psf({"truth": truth, "model": copy.deepcopy(truth)})
    nodes = (4e-7, 6e-7) if sampled else None
    provider = build_psf_provider(specs.truth, pixel_scale_arcsec=0.03, wavelengths_m=nodes)
    model = build_model_psf(specs.model, provider, pixel_scale_arcsec=0.03)
    assert model.relation == "optical" and model.provider.wavelengths_m == provider.wavelengths_m
    for expected, actual in zip(provider.kernels(), model.provider.kernels()):
        np.testing.assert_array_equal(actual.kernel, expected.kernel)
        assert dict(actual.source) == dict(expected.source)
    with pytest.raises(ValueError, match="wavelength keys"):
        build_model_psf(replace(specs.model, wavelength_m=6e-7, wavelength_samples=None), provider,
                        pixel_scale_arcsec=0.03)


@pytest.mark.backend
def test_model_sampling_guards_its_own_optical_system(p1_truth, circular_pupil):
    truth = {**p1_truth, "pupil": circular_pupil, "wavefront": {}}
    model = {**p1_truth, "wavefront": {}, "detector_oversampling": 1}
    specs = parse_psf({"truth": truth, "model": model})
    with pytest.raises(ConfigError, match="under-resolved") as caught:
        _run_cross({"truth": truth, "model": model})
    assert caught.value.path == "psf.model"
    provider = build_psf_provider(specs.truth, pixel_scale_arcsec=0.03)
    with pytest.raises(ValueError, match="under-resolved"):
        build_model_psf(specs.model, provider, pixel_scale_arcsec=0.03)
