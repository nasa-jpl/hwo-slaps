"""Submitted-code oracle evaluated on genuine archived production inputs.

This is not a replay of historical per-node q arrays: those were not archived.
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


@pytest.mark.backend
def test_reference_forecast_matches_production_input_oracle(paper_digest, final_name):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast
    from hwoslaps.simulation import simulate

    oracle = _oracle()
    with prepare_forecast(FIXTURES / oracle["engine_config"],
                          execution=Execution(engine="reference", reference_workers=1)) as preparation:
        assert preparation.mean_truth_adu.shape == (522, 522)
        assert preparation.scene.spec.grid.pixel_scale_arcsec == 0.00716
        assert preparation.scene.spec.grid.over_sample_size == 4
        assert preparation.psfs.truth_kernels.single.kernel.shape == (999, 999)
        assert paper_digest.kernel(preparation.psfs.truth_kernels.single.kernel) == oracle["truth_kernel_sha256"]
        assert paper_digest.kernel(preparation.psfs.model_kernels.single.kernel) == oracle["fit_kernel_sha256"]
        assert oracle["truth_kernel_sha256"] == "429a3702e050178c68b44d4de49e6a61acb595e4e5649120ba31c56470aae88c"
        names = oracle["profiled_nuisance_names"]
        assert len(names) == 10
        mapping = final_name["p3_image_source_kernel"]
        assert preparation.nuisances.names == tuple(mapping[name] for name in names)
        np.testing.assert_array_equal(preparation.nuisances.prior_precision, oracle["nuisance_prior_precision"])
        assert preparation.data_space.pixel_count == oracle["pixels_unmasked"]
        digests = oracle["diagnostic_digests"]
        for name, image in (("mu0_adu", preparation.mean_truth_adu), ("mu0_model_adu", preparation.mean_model_adu),
                            ("sigma_adu", preparation.sigma_adu), ("fisher_mask", preparation.mask.astype(float))):
            assert paper_digest.array(image) == digests[name], name
        for name, image in zip(names, preparation.nuisances.images, strict=True):
            assert paper_digest.array(image) == digests["nuisance_images"][name], name
        noisy = simulate(preparation, subhalo=None, noise_seed=oracle["noise_seed"])
        assert paper_digest.array(noisy.data_adu) == digests["noisy_baseline_adu"]
        with np.load(FIXTURES / oracle["fixture"], allow_pickle=False) as expected:
            result = forecast(preparation, masses_msun=expected["masses_msun"], positions=expected["positions_yx"])
            np.testing.assert_array_equal(result.positions_yx, expected["positions_yx"])
            for name in ("q_asimov", "fisher_raw", "fisher_profiled", "sigma_amplitude", "degradation"):
                np.testing.assert_array_equal(getattr(result, name), expected["reference__" + name], err_msg=name)
