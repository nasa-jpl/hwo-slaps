"""Prepared-state lifecycle and public mass/position semantics."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.backend


@pytest.mark.parametrize("engine", ["reference", "jax"])
def test_mass_rows_ignore_masses_evaluated_earlier(minimal_mapping, engine):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    execution = Execution(engine=engine)
    with prepare_forecast(minimal_mapping, execution=execution) as reused:
        forecast(reused, masses_msun=[3.0e8, 8.0e8])
        actual = forecast(reused, masses_msun=[1.0e8])
        with pytest.raises(TypeError):
            forecast(reused)
    with prepare_forecast(minimal_mapping, execution=execution) as fresh:
        expected = forecast(fresh, masses_msun=[1.0e8])
    np.testing.assert_array_equal(actual.fisher_profiled, expected.fisher_profiled)
    np.testing.assert_array_equal(actual.fisher_raw, expected.fisher_raw)


def test_prepared_configuration_cannot_relabel_cached_science(minimal_mapping):
    from hwoslaps.config.schema import resolve_config
    from hwoslaps.fisher.api import forecast, prepare_forecast

    config = resolve_config(minimal_mapping)
    component = config.scene.source.light[0]
    with prepare_forecast(config) as prepared:
        before = forecast(prepared, masses_msun=[1.0e8])
        digest = prepared.record["config_digest"]
        altered = prepared.config.replace({"scene": {"source": {"light": {component.name: {"intensity": 100.0}}}}})
        assert altered.scene.source.light[0].values["intensity"] == 100.0
        copy = prepared.config.to_mapping()
        copy["scene"]["source"]["light"][component.name]["intensity"] = 200.0
        minimal_mapping["scene"]["source"]["light"][component.name]["intensity"] = 300.0
        after = forecast(prepared, masses_msun=[1.0e8])
        assert prepared.config.scene.source.light[0].values["intensity"] == 1.0
        assert after.provenance["config_digest"] == before.provenance["config_digest"] == digest
        np.testing.assert_array_equal(after.fisher_profiled, before.fisher_profiled)


@pytest.mark.parametrize("asset_kind", ["kernel", "image"])
def test_rewritten_referenced_file_is_refused_before_forecast(minimal_mapping, image_asset, asset_kind):
    from hwoslaps.fisher.api import forecast, prepare_forecast

    if asset_kind == "image":
        minimal_mapping["scene"]["source"]["light"] = {"light": {"type": "Image", "asset_path": str(image_asset),
            "centre": [0.0, 0.0], "flux_scale": 1.0, "size_scale": 1.0, "rotation_deg": 0.0, "total_flux": 1.0}}
        path = image_asset
    else:
        path = Path(minimal_mapping["psf"]["truth"]["path"])
    original = path.read_bytes()
    with prepare_forecast(minimal_mapping) as prepared:
        before = forecast(prepared, masses_msun=[1.0e8])
        try:
            path.write_bytes(original + b"changed")
            with pytest.raises(ValueError, match=str(path)):
                forecast(prepared, masses_msun=[1.0e8])
        finally:
            path.write_bytes(original)
        after = forecast(prepared, masses_msun=[1.0e8])
        np.testing.assert_array_equal(after.fisher_profiled, before.fisher_profiled)
        assert after.provenance["config_digest"] == before.provenance["config_digest"]


@pytest.mark.parametrize("engine", ["reference", "jax"])
def test_position_subsets_reproduce_full_map_rows_and_keep_domain(minimal_mapping, engine):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    minimal_mapping["forecast"]["positions"] = {"kind": "grid", "spacing_arcsec": 0.3, "half_width_arcsec": 0.6}
    with prepare_forecast(minimal_mapping, execution=Execution(engine=engine)) as prepared:
        full = forecast(prepared, masses_msun=[1.0e8])
        keep = np.arange(len(prepared.positions)) % 3 == 0
        subset = forecast(prepared, masses_msun=[1.0e8], positions=prepared.positions.select(keep))
        np.testing.assert_allclose(subset.fisher_profiled, full.fisher_profiled[:, keep], rtol=1.0e-12)
        assert subset.positions.domain_radius_arcsec == full.positions.domain_radius_arcsec
        with pytest.raises(ValueError, match="outside the prepared domain"):
            forecast(prepared, masses_msun=[1.0e8], positions=[[3.0, 0.0]])


def test_identical_model_kernel_is_the_matched_limit(minimal_mapping):
    from hwoslaps.fisher.api import forecast, prepare_forecast

    with prepare_forecast(minimal_mapping) as prepared:
        matched = forecast(prepared, masses_msun=[1.0e8])
        assert prepared.mean_model_adu is prepared.mean_truth_adu
        digest = prepared.record["config_digest"]
    minimal_mapping["psf"]["model"] = deepcopy(minimal_mapping["psf"]["truth"])
    with prepare_forecast(minimal_mapping) as prepared:
        mismatched = forecast(prepared, masses_msun=[1.0e8])
        assert prepared.record["config_digest"] != digest
    np.testing.assert_allclose(mismatched.q_mismatch, matched.q_asimov, rtol=1.0e-10)
    np.testing.assert_allclose(mismatched.amplitude_hat, 1.0, rtol=1.0e-10)
    np.testing.assert_allclose(mismatched.q_spurious, 0.0, atol=1.0e-20)


def test_execution_does_not_enter_configuration_digest(minimal_mapping):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    with prepare_forecast(minimal_mapping) as reference, prepare_forecast(minimal_mapping, execution=Execution(engine="jax")) as jax:
        assert reference.record["config_digest"] == jax.record["config_digest"]
        assert reference.record["comparison_digest"] == jax.record["comparison_digest"]
        for prepared in (reference, jax):
            result = forecast(prepared, masses_msun=[1.0e8])
            assert result.provenance["engine"]["reference_workers"] == 1
            assert result.provenance["engine"]["batch_size"] == 16


@pytest.mark.parametrize("amplitude", [0.0, 1.0])
def test_small_knowledge_error_has_matched_limit_and_quadratic_spurious_response(amplitude):
    from hwoslaps.config.schema import load_config
    from hwoslaps.fisher.api import forecast, prepare_forecast

    path = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity" / "engine" / "p2_delta_knowledge_error.yaml"
    config = load_config(path)
    altered = config.replace({"psf": {"model": {"draw": {"amplitude_rms_nm": amplitude}}}})
    with prepare_forecast(altered) as prepared:
        result = forecast(prepared, masses_msun=[1.0e8], positions=[[0.0, 0.4]])
    if amplitude == 0.0:
        matched = config.replace({"psf": {"model": {"kind": "matched"}}})
        with prepare_forecast(matched) as prepared:
            expected = forecast(prepared, masses_msun=[1.0e8], positions=[[0.0, 0.4]])
        np.testing.assert_allclose(result.q_mismatch, expected.q_asimov, rtol=1.0e-10)
        np.testing.assert_allclose(result.amplitude_hat, 1.0, rtol=1.0e-10)
        np.testing.assert_allclose(result.q_spurious, 0.0, atol=1.0e-20)
    elif amplitude == 1.0:
        half = config.replace({"psf": {"model": {"draw": {"amplitude_rms_nm": 0.5}}}})
        with prepare_forecast(half) as prepared:
            expected = forecast(prepared, masses_msun=[1.0e8], positions=[[0.0, 0.4]])
        np.testing.assert_allclose(result.q_spurious / expected.q_spurious, 4.0, rtol=0.15)


@pytest.mark.parametrize("changed", ["kernel", "convolver"])
def test_prepared_kernel_buffers_cannot_change_identity(minimal_mapping, changed):
    from hwoslaps.fisher.api import forecast, prepare_forecast

    with prepare_forecast(minimal_mapping) as prepared:
        kernel = prepared.psfs.truth_kernels.single
        if changed == "kernel":
            buffer = kernel.kernel
            buffer.setflags(write=True)
            buffer[3, 3] += 0.01
        else:
            import autoarray as aa
            buffer = np.array(kernel.convolver().kernel.native)
            buffer[3, 3] += 0.01
            kernel.convolver().kernel = aa.Array2D.no_mask(values=buffer, pixel_scales=kernel.pixel_scale_arcsec)
        with pytest.raises(ValueError, match="truth.*kernel.*changed"):
            forecast(prepared, masses_msun=[1.0e8])
        with pytest.raises(TypeError):
            prepared.record["config_digest"] = "changed"


@pytest.mark.parametrize("engine", ["reference", "jax"])
def test_closed_preparation_refuses_forecast(minimal_mapping, engine):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    prepared = prepare_forecast(minimal_mapping, execution=Execution(engine=engine))
    forecast(prepared, masses_msun=[1.0e8])
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        forecast(prepared, masses_msun=[1.0e8])


def test_rank_zero_nuisance_has_json_serializable_condition_metadata(minimal_mapping, tmp_path):
    from hwoslaps.fisher.api import forecast, prepare_forecast

    kernel_path = tmp_path / "delta.npy"
    np.save(kernel_path, np.ones((1, 1)))
    minimal_mapping["psf"]["truth"]["path"] = str(kernel_path)
    minimal_mapping["scene"]["grid"].update(shape=[3, 3], over_sample_size=1)
    minimal_mapping["scene"]["lens"]["mass"]["mass"]["ell_comps"] = [0.0, 0.0]
    minimal_mapping["scene"]["source"]["light"]["light"]["centre"] = [0.0, 0.0]
    minimal_mapping["forecast"]["mask"] = {"kind": "annulus", "about": "grid", "inner_arcsec": 0.0, "outer_arcsec": 0.001}
    minimal_mapping["forecast"]["nuisances"] = {"background_offset": False,
        "fixed": ["lens.*", "source.*.centre_x", "source.*.ell_*", "source.*.intensity", "source.*.effective_radius"]}
    with prepare_forecast(minimal_mapping) as prepared:
        assert prepared.nuisances.names == ("source.light.light.centre_y",)
        assert prepared.workspace.nuisance_rank == 0
        assert np.isinf(prepared.workspace.condition_number)
        result = forecast(prepared, masses_msun=[1.0e8])
        assert result.provenance["nuisance_rank"] == 0
        assert result.provenance["gram_condition_number"] is None


@pytest.mark.parametrize("publication", ["after_capture", "after_read"])
def test_input_publication_during_preparation_is_refused(minimal_mapping, publication):
    import sys
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.identity import file_digest, read_file_snapshot

    path = Path(minimal_mapping["psf"]["truth"]["path"])
    original = path.read_bytes()
    trigger = file_digest if publication == "after_capture" else read_file_snapshot
    published = []
    previous_profile = sys.getprofile()
    def publish(frame, event, returned):
        if event == "return" and frame.f_code is trigger.__code__ and Path(frame.f_locals["path"]) == path:
            sys.setprofile(previous_profile)
            kernel = np.load(path, allow_pickle=False)
            kernel[3, 3] *= 1.4
            np.save(path, kernel / kernel.sum())
            published.append(True)
    sys.setprofile(publish)
    try:
        with pytest.raises(ValueError, match=str(path)):
            prepare_forecast(minimal_mapping)
        assert published == [True]
    finally:
        sys.setprofile(previous_profile)
        path.write_bytes(original)
