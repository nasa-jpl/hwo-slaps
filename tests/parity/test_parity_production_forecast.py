"""Submitted-code oracle evaluated on genuine archived production inputs.

The archived arrays remain sealed historical evidence, not expected corrected outputs.
Historical per-node campaign q arrays were not archived.
The generator imports only verified 41621de Python files and preserves the
archived sys0043 scene, COSMOS asset, science35 kernel and detector/noise inputs.
The 8fa6209 nonlinear fixture remains a separate continuity anchor.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/paper_parity/production"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oracle():
    return json.loads((FIXTURES / "oracle.json").read_text())


def test_production_oracle_binds_submitted_code_and_original_inputs():
    oracle = _oracle()
    assert oracle["submitted_code"]["paper_commit"] == "41621de8ca861e425e1caddb7a599d5c1032751f"
    assert oracle["submitted_code"]["git_tree"] == "2f185e14053017b85b084f0387906b813b99ec1d"
    assert oracle["submitted_code"]["python_files_verified"] == 103
    assert _sha256(FIXTURES / "submitted_code_41621de.sha256") == oracle["submitted_code"]["python_manifest_sha256"]
    assert oracle["archive_provenance"]["production_commit"] == "28a41c232eaccd508293f9af82fa0cc10d4127c3"
    assert oracle["archive_provenance"]["historical_per_node_replay"] is False
    assert _sha256(FIXTURES / oracle["source_asset"]["file"]) == oracle["source_asset"]["sha256"]
    assert oracle["source_asset"]["sha256"] == "c67570fdc56068fc900e3428f6b6ff2d5ffce8f459858c302c6de1461d24ed21"
    assert _sha256(FIXTURES / oracle["engine_config"]) == oracle["engine_config_sha256"]
    assert _sha256(FIXTURES / oracle["fixture"]) == oracle["fixture_sha256"]


@pytest.fixture(scope="module")
def production_preparation():
    from dataclasses import replace
    from hwoslaps.fisher.api import Execution, prepare_forecast
    from hwoslaps.fisher.engines.base import make_engine
    from current_oracles import forecast_oracle
    oracle = _oracle()
    reference = prepare_forecast(FIXTURES/oracle["engine_config"], execution=Execution(engine="reference"))
    with np.load(FIXTURES/oracle["fixture"], allow_pickle=False) as archive:
        masses, positions = archive["masses_msun"], archive["positions_yx"]
    expected, singular = forecast_oracle(reference, masses, positions)
    accelerated = None
    def get(engine):
        nonlocal accelerated
        if engine == "reference":
            return reference
        if accelerated is None:
            execution = Execution(engine="jax")
            accelerated = replace(reference, engine=make_engine("jax", reference.engine.context, execution), execution=execution)
        return accelerated
    yield get, expected, singular
    if accelerated is not None:
        accelerated.close()
    reference.close()


def _current_production(engine, production_preparation, paper_digest, final_name, tmp_path, record_property):
    from hwoslaps.fisher.api import forecast
    from hwoslaps.simulation import simulate
    from current_oracles import assert_detector_response, assert_forecast_statistics, historical_changes, write_receipt
    oracle = _oracle()
    get, expected, singular = production_preparation
    preparation = get(engine)
    assert preparation.mean_truth_adu.shape == (522, 522)
    assert preparation.scene.spec.grid.pixel_scale_arcsec == 0.00716
    assert preparation.scene.spec.grid.over_sample_size == 4
    assert preparation.psfs.truth_kernels.single.kernel.shape == (999, 999)
    assert paper_digest.kernel(preparation.psfs.truth_kernels.single.kernel) == oracle["truth_kernel_sha256"]
    assert paper_digest.kernel(preparation.psfs.model_kernels.single.kernel) == oracle["fit_kernel_sha256"]
    assert preparation.nuisances.names == tuple(final_name["p3_image_source_kernel"][name] for name in oracle["profiled_nuisance_names"])
    np.testing.assert_array_equal(preparation.nuisances.prior_precision, oracle["nuisance_prior_precision"])
    assert preparation.data_space.pixel_count == oracle["pixels_unmasked"]
    assert paper_digest.array(preparation.mask.astype(float)) == oracle["diagnostic_digests"]["fisher_mask"]
    assert_detector_response(preparation)
    noisy = simulate(preparation, subhalo=None, noise_seed=oracle["noise_seed"])
    again = simulate(preparation, subhalo=None, noise_seed=oracle["noise_seed"])
    np.testing.assert_array_equal(noisy.data_adu, again.data_adu)
    inputs = {name: {"historical_digest": oracle["diagnostic_digests"][name], "current_digest": paper_digest.array(image)}
              for name, image in (("mu0_adu", preparation.mean_truth_adu), ("mu0_model_adu", preparation.mean_model_adu),
                                  ("sigma_adu", preparation.sigma_adu), ("noisy_baseline_adu", noisy.data_adu))}
    with np.load(FIXTURES/oracle["fixture"], allow_pickle=False) as archive:
        result = forecast(preparation, masses_msun=archive["masses_msun"], positions=archive["positions_yx"])
        np.testing.assert_array_equal(result.positions_yx, archive["positions_yx"])
        errors = assert_forecast_statistics(result, expected)
        changes = historical_changes(result, archive, np.arange(3), tuple(expected))
    if engine == "jax":
        assert result.provenance["engine"]["kind"] == "jax"
    write_receipt(tmp_path, "production_"+engine, {"scope": "Current engine on archived production inputs; not paper reproduction",
        "input_digests": inputs, "qr_max_absolute_errors": errors, "historical_changes": changes,
        "smallest_relative_augmented_singular_value": singular, "engine": result.provenance["engine"]}, record_property)


@pytest.mark.backend
def test_reference_production_inputs_satisfy_current_qr_oracle(production_preparation, paper_digest, final_name, tmp_path, record_property):
    _current_production("reference", production_preparation, paper_digest, final_name, tmp_path, record_property)


@pytest.mark.backend
@pytest.mark.xtx_gpu
def test_gpu_production_inputs_satisfy_current_qr_oracle(production_preparation, paper_digest, final_name, tmp_path, record_property):
    _current_production("jax", production_preparation, paper_digest, final_name, tmp_path, record_property)
