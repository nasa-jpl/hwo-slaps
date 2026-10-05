"""The ``psf`` section, its cross-section rules, the kernel provider and the model-PSF relations."""

import copy
import shutil
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.optics.kernels import DetectorPSF
from hwoslaps.optics.knowledge_error import WavefrontDrawSpec
from hwoslaps.optics.mode_priors import ModeWeightPriorSpec
from hwoslaps.optics.providers import (
    CROSS_RULES, PSF_TABLE, KernelFileSpec, KernelPSF, KnowledgeErrorModel, MatchedModel, WavefrontModel,
    build_model_psf, build_psf_provider, parse_psf,
)
from hwoslaps.optics.wavefront import WavefrontCoefficients

DELETE = object()
DRAW = {"prior": {"packaged": "jwst_wss_drift_v1"}, "amplitude_rms_nm": 10.0, "seed": 20261005}


def _edited(mapping, edits):
    result = copy.deepcopy(mapping)
    for dotted, value in edits.items():
        *parents, last = dotted.split(".")
        node = result
        for key in parents:
            node = node[key]
        if value is DELETE:
            del node[last]
        else:
            node[last] = value
    return result


def _draw_model(**draw):
    return {"kind": "knowledge_error", "draw": {**DRAW, **draw}}


@pytest.fixture
def kernel_file(tmp_path):
    path = tmp_path / "kernel.npy"
    np.save(path, np.pad([[1.0, 2.0, 1.0]], ((1, 1), (0, 0))))
    return str(path)


@pytest.fixture
def prior_file(tmp_path):
    path = tmp_path / "prior.yaml"
    with resources.as_file(resources.files("hwoslaps.optics").joinpath("priors", "jwst_wss_drift_v1.yaml")) as data:
        shutil.copyfile(data, path)
    return str(path)


PARSE_ROWS = {
    "unknown-top-key": ({"colour": "red"}, "psf.colour"),
    "unknown-truth-key": ({"truth.colour": "red"}, "psf.truth.colour"),
    "unknown-pupil-key": ({"truth.pupil.colour": "red"}, "psf.truth.pupil.colour"),
    "unknown-spider-key": ({"truth.pupil.spiders": {"count": 3, "width_m": 0.1, "colour": "red"}},
                           "psf.truth.pupil.spiders.colour"),
    "unknown-family": ({"truth.wavefront.pistons": {0: 1.0}}, "psf.truth.wavefront.pistons"),
    "unknown-draw-key": ({"model": _draw_model(colour=1)}, "psf.model.draw.colour"),
    "unknown-prior-key": ({"model": _draw_model(prior={"packaged": "jwst_wss_drift_v1", "colour": 1})},
                          "psf.model.draw.prior.colour"),
    "unknown-truth-kind": ({"truth.kind": "telescope"}, "psf.truth.kind"),
    "unknown-model-kind": ({"model": {"kind": "explicit"}}, "psf.model.kind"),
    "even-kernel-rows": ({"truth.kernel_shape": [16, 17]}, "psf.truth.kernel_shape[0]"),
    "even-kernel-columns": ({"truth.kernel_shape": [17, 16]}, "psf.truth.kernel_shape[1]"),
    "wavelength-required": ({"truth.wavelength_nm": DELETE}, "psf.truth"),
    "oversampling-required": ({"truth.detector_oversampling": DELETE}, "psf.truth.detector_oversampling"),
    "rings-required": ({"truth.pupil.rings": DELETE}, "psf.truth.pupil.rings"),
    "hex-key-on-circle": ({"truth.pupil": {"kind": "circular", "diameter_m": 7.2, "pixels": 128,
                                           "supersampling": 2, "rings": 2}}, "psf.truth.pupil.rings"),
    "no-centre-no-rings": ({"truth.pupil.central_segment": False, "truth.pupil.rings": 0}, "psf.truth.pupil.rings"),
    "aperture-outside-grid": ({"truth.pupil.diameter_m": 7.0}, "psf.truth.pupil.diameter_m"),
    "obscuration-one": ({"truth.pupil.obscuration_ratio": 1.0}, "psf.truth.pupil.obscuration_ratio"),
    "missing-segment": ({"truth.wavefront.segment_hexikes": {19: {1: 1.0}}},
                        "psf.truth.wavefront.segment_hexikes.19"),
    "draw-with-wavefront": ({"truth.draw": DRAW}, "psf.truth.draw"),
    "wavefront-and-offset": ({"model": {"kind": "wavefront", "wavefront": {"zernikes": {4: 1.0}},
                                        "offset": {"zernikes": {4: 1.0}}}}, "psf.model"),
    "neither-wavefront-nor-offset": ({"model": {"kind": "wavefront"}}, "psf.model"),
    "prior-without-source": ({"model": _draw_model(prior={})}, "psf.model.draw.prior"),
    "prior-file-missing": ({"model": _draw_model(prior={"path": "/no/such/prior.yaml"})},
                           "psf.model.draw.prior.path"),
    "unknown-packaged-prior": ({"model": _draw_model(prior={"packaged": "jwst_wss_static_v2"})},
                               "psf.model.draw.prior.packaged"),
    "power-law-range": ({"model": _draw_model(prior={"power_law": {
        "alpha": 1.0, "global_nolls": [6, 4], "segment_nolls": None, "segment_variance_fraction": 0.0}})},
        "psf.model.draw.prior.power_law.global_nolls"),
    "power-law-alpha": ({"model": _draw_model(prior={"power_law": {
        "global_nolls": [4, 6], "segment_nolls": None, "segment_variance_fraction": 0.0}})},
        "psf.model.draw.prior.power_law.alpha"),
    "draw-family": ({"model": _draw_model(family="pistons")}, "psf.model.draw.family"),
    "negative-amplitude": ({"model": _draw_model(amplitude_rms_nm=-1.0)}, "psf.model.draw.amplitude_rms_nm"),
}


@pytest.mark.parametrize("edits, path", list(PARSE_ROWS.values()), ids=list(PARSE_ROWS))
def test_psf_section_parse_table(p1_truth, edits, path):
    with pytest.raises(ConfigError) as caught:
        parse_psf(_edited({"truth": p1_truth}, edits))
    assert caught.value.path == path


def test_psf_section_rules_across_truth_model_and_files(p1_truth, circular_pupil, kernel_file, prior_file):
    kernel_truth = {"kind": "kernel", "path": kernel_file, "pixel_scale_arcsec": 0.03}
    circular = {**p1_truth, "pupil": circular_pupil, "wavefront": {}}
    rows = [
        ({"truth": kernel_truth, "model": {"kind": "wavefront", "offset": {"zernikes": {4: 1.0}}}}, "psf.model.kind"),
        ({"truth": kernel_truth, "model": _draw_model()}, "psf.model.kind"),
        ({"truth": {**kernel_truth, "array_key": "kernel"}}, "psf.truth.array_key"),
        ({"truth": {**circular, "wavefront": {"segment_hexikes": {0: {1: 1.0}}}}},
         "psf.truth.wavefront.segment_hexikes"),
        ({"truth": circular, "model": {"kind": "wavefront", "offset": {"segment_hexikes": {0: {1: 1.0}}}}},
         "psf.model.offset.segment_hexikes"),
        ({"truth": circular, "model": _draw_model()}, "psf.model.draw.family"),
        ({"truth": {**circular, "draw": {**DRAW, "family": "segment"}}}, "psf.truth.draw.family"),
        ({"truth": p1_truth, "model": _draw_model(prior={"packaged": "jwst_wss_drift_v1", "path": prior_file})},
         "psf.model.draw.prior"),
    ]
    for mapping, path in rows:
        with pytest.raises(ConfigError) as caught:
            parse_psf(mapping)
        assert caught.value.path == path
    spec = parse_psf({"truth": circular, "model": _draw_model(family="global", prior={"path": prior_file})})
    assert spec.model.draw == WavefrontDrawSpec(ModeWeightPriorSpec("path", None, Path(prior_file), None), 10.0,
                                                20261005, "global")


def test_psf_section_fills_defaults_and_reparses_to_an_equal_spec(p1_truth, kernel_file):
    mapping = {"truth": {**p1_truth, "pupil": {**p1_truth["pupil"], "spiders": {"count": 3, "width_m": 0.1}}},
               "model": _draw_model()}
    values = PSF_TABLE.read(mapping, "psf")
    assert PSF_TABLE.read(values, "psf") == values
    assert parse_psf(values) == parse_psf(mapping)
    pupil = values["truth"]["pupil"]
    assert (pupil["obscuration_ratio"], pupil["central_segment"], pupil["spiders"]["angle_deg"]) == (0.0, True, 0.0)
    assert values["truth"]["draw"] is None and values["model"]["draw"]["family"] == "combined"
    spec = parse_psf(mapping)
    assert spec.truth.wavelength_m == 500.0 / 1e9 and spec.truth.kernel_shape == (17, 17)
    assert spec.model == KnowledgeErrorModel(WavefrontDrawSpec(
        ModeWeightPriorSpec("packaged", "jwst_wss_drift_v1", None, None), 10.0, 20261005, "combined"))
    assert PSF_TABLE.read({"truth": p1_truth}, "psf")["model"] == {"kind": "matched"}
    assert parse_psf({"truth": p1_truth}).model == MatchedModel()
    npz = kernel_file.replace(".npy", ".npz")
    np.savez(npz, kernel=np.load(kernel_file))
    model = parse_psf({"truth": p1_truth, "model": {"kind": "kernel", "path": npz, "pixel_scale_arcsec": 0.03}}).model
    assert model == KernelFileSpec(Path(npz), "kernel", 0.03, True, None)


def test_cross_rules_check_kernel_pixel_scales_and_truth_sampling(p1_truth, paper_pupil, kernel_file):
    kernel_truth = {"kind": "kernel", "path": kernel_file, "pixel_scale_arcsec": 0.03}
    paper = {**p1_truth, "pupil": paper_pupil, "detector_oversampling": 3, "kernel_shape": [999, 999],
             "wavefront": {}}

    def root(psf, scale):
        return {"scene": {"grid": {"pixel_scale_arcsec": scale}}, "psf": PSF_TABLE.read(psf, "psf")}

    for psf, scale in [({"truth": kernel_truth}, 0.03), ({"truth": kernel_truth}, 0.03 + 5e-13),
                       ({"truth": paper}, 0.00716)]:
        for rule in CROSS_RULES:
            rule.check(root(psf, scale), "")
    for psf, scale, path, message in [
        ({"truth": kernel_truth}, 0.02, "psf.truth.pixel_scale_arcsec", "never resampled"),
        ({"truth": p1_truth, "model": {**kernel_truth, "pixel_scale_arcsec": 0.031}}, 0.03,
         "psf.model.pixel_scale_arcsec", "never resampled"),
        ({"truth": paper}, 0.0074, "psf.truth", "aliased kernel at 500 nm"),
        ({"truth": {**paper, "detector_oversampling": 1}}, 0.00716, "psf.truth", "under-resolved kernel at 500 nm"),
    ]:
        with pytest.raises(ConfigError, match=message) as caught:
            for rule in CROSS_RULES:
                rule.check(root(psf, scale), "")
        assert caught.value.path == path


@pytest.mark.backend
def test_model_relations_build_their_providers(p1_truth, kernel_file):
    small = {**p1_truth, "pupil": {**p1_truth["pupil"], "pixels": 64}, "kernel_shape": [9, 9]}
    truth = build_psf_provider(parse_psf({"truth": small}).truth, pixel_scale_arcsec=0.03)

    matched = build_model_psf(MatchedModel(), truth, pixel_scale_arcsec=0.03)
    assert matched.provider is truth and matched.relation == "matched" and matched.knowledge_error is None

    replacement = WavefrontCoefficients.from_mapping({"zernikes": {6: 3.0}}, "wavefront")
    replaced = build_model_psf(WavefrontModel(replacement, None), truth, pixel_scale_arcsec=0.03)
    assert replaced.provider.coefficients == replacement and replaced.relation == "wavefront"
    assert replaced.provider.pupil is truth.pupil and replaced.provider.basis is truth.basis

    offset = WavefrontCoefficients.from_mapping({"segment_hexikes": {0: {4: 1.5}}, "zernikes": {6: 2.0}}, "offset")
    shifted = build_model_psf(WavefrontModel(None, offset), truth, pixel_scale_arcsec=0.03).provider.coefficients
    assert shifted.to_mapping() == {"segment_hexikes": {0: {4: 11.5}, 3: {5: 8.0}, 7: {6: 12.0}},
                                    "zernikes": {4: 5.0, 5: 5.0, 6: 2.0, 8: 5.0}}

    draw = parse_psf({"truth": small, "model": _draw_model()}).model
    error = build_model_psf(draw, truth, pixel_scale_arcsec=0.03)
    assert error.relation == "knowledge_error"
    assert error.provider.coefficients == error.knowledge_error.model
    assert error.knowledge_error.model == truth.coefficients.plus(error.knowledge_error.draw.coefficients)
    assert error.provider.kernel().kernel.tobytes() != truth.kernel().kernel.tobytes()

    kernel_spec = KernelFileSpec(Path(kernel_file), None, 0.03, True, None)
    kernel_model = build_model_psf(kernel_spec, truth, pixel_scale_arcsec=0.03)
    expected = DetectorPSF.from_array(np.load(kernel_file), 0.03, normalize=True).kernel_identity()
    assert isinstance(kernel_model.provider, KernelPSF) and kernel_model.relation == "kernel"
    assert kernel_model.provider.kernel().kernel_identity() == expected
    with pytest.raises(ValueError, match="never resampled"):
        build_model_psf(KernelFileSpec(Path(kernel_file), None, 0.031, True, None), truth, pixel_scale_arcsec=0.03)

    kernel_truth = build_psf_provider(kernel_spec, pixel_scale_arcsec=0.03)
    for model in (WavefrontModel(replacement, None), draw):
        with pytest.raises(ValueError, match="needs an optical truth"):
            build_model_psf(model, kernel_truth, pixel_scale_arcsec=0.03)


def test_kernel_provider_limits(kernel_file):
    psf = DetectorPSF.from_file(kernel_file, pixel_scale_arcsec=0.03, array_key=None, normalize=True,
                                file_sha256=None)
    provider = KernelPSF(psf)
    assert provider.kernel() is psf and provider.shape == (3, 3) and provider.pixel_scale_arcsec == 0.03
    assert (provider.wavelengths_m, provider.basis, provider.coefficients, provider.collecting_area_m2) == (
        None, None, None, None)
    with pytest.raises(ValueError, match="no spectral information"):
        provider.kernel(5e-7)
    with pytest.raises(ValueError, match="no wavefront basis"):
        provider.kernel(coefficients=WavefrontCoefficients.empty())
    assert provider.to_mapping() == {"provider": "kernel", "kernel": psf.kernel_identity().to_mapping(),
                                     "source": dict(psf.source)}


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0.0, -0.03, True, "0.03"])
def test_builders_refuse_a_scene_pixel_scale_that_is_not_finite_and_positive(kernel_file, scale):
    spec = KernelFileSpec(Path(kernel_file), None, 0.03, True, None)
    truth = build_psf_provider(spec, pixel_scale_arcsec=0.03)
    with pytest.raises(ValueError, match="pixel_scale_arcsec must be finite and positive"):
        build_psf_provider(spec, pixel_scale_arcsec=scale)
    with pytest.raises(ValueError, match="pixel_scale_arcsec must be finite and positive"):
        build_model_psf(MatchedModel(), truth, pixel_scale_arcsec=scale)
