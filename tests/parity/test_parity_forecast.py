"""Input-first comparison against independently generated submitted-paper anchors."""

import hashlib
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

FISHER_SCENES = ("p1_optical_matched", "p2_delta_knowledge_error", "p3_image_source_kernel",
                 "p4_subhalo_sis", "p4_subhalo_pointmass")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity"
STATISTICS = ("q_asimov", "fisher_raw", "fisher_profiled", "sigma_amplitude", "degradation")
MISMATCH = ("amplitude_hat", "q_mismatch", "z_mismatch", "amplitude_spurious", "q_spurious", "z_spurious")


def assert_inputs_match_paper(scene, preparation, manifest, paper_digest, final_name):
    entry = manifest["scenes"][scene]
    assert paper_digest.kernel(preparation.psfs.truth_kernels.single.kernel) == entry["truth_kernel_sha256"]
    assert paper_digest.kernel(preparation.psfs.model_kernels.single.kernel) == entry["fit_kernel_sha256"]
    names = entry["profiled_nuisance_names"]
    assert preparation.nuisances.names == tuple(final_name[scene][name] for name in names)
    np.testing.assert_array_equal(preparation.nuisances.prior_precision, entry["nuisance_prior_precision"])
    assert preparation.data_space.pixel_count == entry["pixels_unmasked"]
    digests = entry["diagnostic_digests"]
    for name, image in (("mu0_adu", preparation.mean_truth_adu), ("mu0_model_adu", preparation.mean_model_adu),
                        ("sigma_adu", preparation.sigma_adu), ("fisher_mask", preparation.mask.astype(float))):
        assert paper_digest.array(image) == digests[name], f"{scene} input {name}"
    for name, image in zip(names, preparation.nuisances.images):
        assert paper_digest.array(image) == digests["nuisance_images"][name], f"{scene} input {name}"
    if "fit_psf_delta" in entry:
        draw = preparation.psfs.model.knowledge_error.draw
        expected = entry["fit_psf_delta"]
        assert draw.spec.amplitude_rms_nm == expected["requested_amplitude_rms_nm"]
        assert draw.measured_rms_nm == expected["measured_draw_rms_nm"]
        assert draw.spec.seed == expected["seed"]
        assert draw.spec.family == expected["family"]
        coefficients = draw.coefficients.to_mapping()
        old = expected["draw_aberrations"]
        assert coefficients["segment_hexikes"] == {int(segment): {int(noll): value for noll, value in modes}
                                                   for segment, modes in old["segment_hexikes"]}
        assert coefficients["zernikes"] == {int(noll): value for noll, value in old["global_zernikes"]}


def assert_statistics_match_paper(scene, preparation, lane, manifest, *, rtol=None):
    from hwoslaps.fisher.api import forecast

    with np.load(FIXTURES / manifest["scenes"][scene]["fixture"], allow_pickle=False) as expected:
        result = forecast(preparation, masses_msun=expected["masses_msun"])
        np.testing.assert_array_equal(result.positions_yx, expected["positions_yx"])
        mismatch = f"{lane}__q_mismatch" in expected
        assert (result.q_mismatch is not None) is mismatch
        for name in STATISTICS + (MISMATCH if mismatch else ()):
            if rtol is None:
                np.testing.assert_array_equal(getattr(result, name), expected[f"{lane}__{name}"], err_msg=f"{scene} {name}")
            else:
                np.testing.assert_allclose(getattr(result, name), expected[f"{lane}__{name}"], rtol=rtol, atol=0.0,
                                           err_msg=f"{scene} {name}")


@pytest.mark.backend
@pytest.mark.parametrize("scene", FISHER_SCENES)
def test_reference_forecast_reproduces_paper(scene, prepared, manifest, paper_digest, final_name):
    preparation = prepared(scene, "reference")
    assert_inputs_match_paper(scene, preparation, manifest, paper_digest, final_name)
    assert_statistics_match_paper(scene, preparation, "reference", manifest)


@pytest.mark.backend
@pytest.mark.xtx_gpu
@pytest.mark.parametrize("scene", FISHER_SCENES)
def test_jax_forecast_reproduces_paper(scene, prepared, manifest, paper_digest, final_name):
    preparation = prepared(scene, "jax")
    assert_inputs_match_paper(scene, preparation, manifest, paper_digest, final_name)
    assert_statistics_match_paper(scene, preparation, "jax_gpu", manifest)


@pytest.mark.backend
@pytest.mark.parametrize("scene", FISHER_SCENES)
def test_jax_cpu_forecast_agrees_with_paper_reference(scene, prepared, manifest, paper_digest, final_name):
    preparation = prepared(scene, "jax")
    assert_inputs_match_paper(scene, preparation, manifest, paper_digest, final_name)
    assert_statistics_match_paper(scene, preparation, "reference", manifest, rtol=5.0e-6)


def test_packaged_prior_table_is_the_paper_table(manifest):
    path = "configs/psf_priors/jwst_wss_drift_v1.yaml"
    data = files("hwoslaps.optics").joinpath("priors", "jwst_wss_drift_v1.yaml").read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["inputs"][path]
