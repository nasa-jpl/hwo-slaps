"""Archived input identities plus independent corrected-policy forecast controls.

Historical output arrays stay immutable comparison evidence. Current q and signed
amplitudes must satisfy an independent augmented QR oracle on live physical inputs.
"""
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from current_oracles import (AugmentedQrOracle, assert_detector_response, assert_forecast_statistics,
                             historical_changes, write_receipt)

FISHER_SCENES = ("p1_optical_matched", "p2_delta_knowledge_error", "p3_image_source_kernel",
                 "p4_subhalo_sis", "p4_subhalo_pointmass")
FIXTURES = Path(__file__).resolve().parents[1]/"fixtures/paper_parity"
STATISTICS = ("q_asimov", "fisher_raw", "fisher_profiled", "sigma_amplitude", "degradation")
MISMATCH = ("amplitude_hat", "q_mismatch", "z_mismatch", "amplitude_spurious", "q_spurious", "z_spurious")


def assert_unchanged_inputs(scene, preparation, manifest, paper_digest, final_name):
    entry = manifest["scenes"][scene]
    assert paper_digest.kernel(preparation.psfs.truth_kernels.single.kernel) == entry["truth_kernel_sha256"]
    assert paper_digest.kernel(preparation.psfs.model_kernels.single.kernel) == entry["fit_kernel_sha256"]
    names = entry["profiled_nuisance_names"]
    assert preparation.nuisances.names == tuple(final_name[scene][name] for name in names)
    np.testing.assert_array_equal(preparation.nuisances.prior_precision, entry["nuisance_prior_precision"])
    assert preparation.data_space.pixel_count == entry["pixels_unmasked"]
    digests = entry["diagnostic_digests"]
    assert paper_digest.array(preparation.mask.astype(float)) == digests["fisher_mask"]
    changes = {name: {"historical_digest": digests[name], "current_digest": paper_digest.array(image)}
               for name, image in (("mu0_adu", preparation.mean_truth_adu),
                                   ("mu0_model_adu", preparation.mean_model_adu), ("sigma_adu", preparation.sigma_adu))}
    changes["nuisance_images"] = {name: {"historical_digest": digests["nuisance_images"][name],
                                        "current_digest": paper_digest.array(image)}
                                  for name, image in zip(names, preparation.nuisances.images, strict=True)}
    if "fit_psf_delta" in entry:
        draw, expected = preparation.psfs.model.knowledge_error.draw, entry["fit_psf_delta"]
        assert draw.spec.amplitude_rms_nm == expected["requested_amplitude_rms_nm"]
        assert draw.measured_rms_nm == expected["measured_draw_rms_nm"]
        assert draw.spec.seed == expected["seed"] and draw.spec.family == expected["family"]
        coefficients, old = draw.coefficients.to_mapping(), expected["draw_aberrations"]
        assert coefficients["segment_hexikes"] == {int(s): {int(n): v for n, v in modes} for s, modes in old["segment_hexikes"]}
        assert coefficients["zernikes"] == {int(n): v for n, v in old["global_zernikes"]}
    assert_detector_response(preparation)
    return changes


def _check(scene, engine, prepared, manifest, paper_digest, final_name, current_forecast_oracle, tmp_path, record_property):
    from hwoslaps.fisher.api import forecast
    preparation = prepared(scene, engine)
    inputs = assert_unchanged_inputs(scene, preparation, manifest, paper_digest, final_name)
    with np.load(FIXTURES/manifest["scenes"][scene]["fixture"], allow_pickle=False) as archive:
        # A declared bounded cohort, including both opposite corners and centre;
        # every archived mass is retained. Full archived arrays remain hash sealed.
        indices = np.unique([0, len(archive["positions_yx"])//2, len(archive["positions_yx"])-1])
        positions, masses = archive["positions_yx"][indices], archive["masses_msun"]
        result = forecast(preparation, masses_msun=masses, positions=positions)
        np.testing.assert_array_equal(result.positions_yx, positions)
        oracle, singular = current_forecast_oracle(scene, masses, positions)
        errors = assert_forecast_statistics(result, oracle)
        changes = historical_changes(result, archive, indices, STATISTICS+(MISMATCH if result.q_mismatch is not None else ()))
    route = result.provenance["engine"]
    assert route["native_binning"] == ("adaptive-pixel-integral" if scene.startswith("p4_") else "uniform-block-mean")
    if engine == "jax" and scene.startswith("p4_"):
        assert route["kind"] == "reference" and route["requested_kind"] == "jax" and route["jax_accelerated"] is False
    write_receipt(tmp_path, scene+"_"+engine, {"scene": scene, "cohort_indices": indices.tolist(),
        "masses_msun": masses.tolist(), "positions_yx": positions.tolist(), "input_digests": inputs,
        "smallest_relative_augmented_singular_value": singular, "qr_max_absolute_errors": errors,
        "historical_changes": changes, "engine": route}, record_property)


@pytest.mark.backend
@pytest.mark.parametrize("scene", FISHER_SCENES)
def test_reference_forecast_satisfies_current_qr_contract(scene, prepared, manifest, paper_digest, final_name,
                                                        current_forecast_oracle, tmp_path, record_property):
    _check(scene, "reference", prepared, manifest, paper_digest, final_name, current_forecast_oracle, tmp_path, record_property)


@pytest.mark.backend
@pytest.mark.xtx_gpu
@pytest.mark.parametrize("scene", FISHER_SCENES)
def test_jax_gpu_forecast_satisfies_current_qr_contract(scene, prepared, manifest, paper_digest, final_name,
                                                      current_forecast_oracle, tmp_path, record_property):
    _check(scene, "jax", prepared, manifest, paper_digest, final_name, current_forecast_oracle, tmp_path, record_property)


@pytest.mark.backend
@pytest.mark.parametrize("scene", FISHER_SCENES)
def test_jax_cpu_forecast_satisfies_current_qr_contract(scene, prepared, manifest, paper_digest, final_name,
                                                      current_forecast_oracle, tmp_path, record_property):
    _check(scene, "jax", prepared, manifest, paper_digest, final_name, current_forecast_oracle, tmp_path, record_property)


def test_archived_inputs_outputs_and_manifests_are_unchanged():
    sealed = json.loads((Path(__file__).parent/"historical_sha256.json").read_text())
    assert sealed["submitted_commit"] == "41621de8ca861e425e1caddb7a599d5c1032751f"
    assert set(sealed["sha256"]) == {p.relative_to(FIXTURES).as_posix() for p in FIXTURES.rglob("*") if p.is_file()}
    for name, digest in sealed["sha256"].items():
        assert hashlib.sha256((FIXTURES/name).read_bytes()).hexdigest() == digest, name


def test_qr_oracle_has_independent_analytic_controls():
    # Profiling J=e0 leaves s=e1 unchanged. A unit Gaussian prior instead
    # leaves half of e0's information through its data AND prior residuals.
    oracle = AugmentedQrOracle(np.array([[1.], [0.]]), np.array([0.]))
    residual = oracle.residual_coordinates([0., 2.])
    assert residual@residual == pytest.approx(4., rel=1e-15)
    oracle = AugmentedQrOracle(np.array([[1.], [0.]]), np.array([1.]))
    residual = oracle.residual_coordinates([2., 0.])
    assert residual@residual == pytest.approx(2., rel=1e-15)
    assert np.dot(residual, oracle.residual_coordinates([-3., 0.])) == pytest.approx(-3., rel=1e-15)


def test_packaged_prior_table_is_the_paper_table(manifest):
    data = files("hwoslaps.optics").joinpath("priors", "jwst_wss_drift_v1.yaml").read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["inputs"]["configs/psf_priors/jwst_wss_drift_v1.yaml"]
