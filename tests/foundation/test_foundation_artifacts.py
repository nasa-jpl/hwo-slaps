"""Current artifact formats, exact scientific values and atomic no-overwrite publishing."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from hwoslaps.artifacts import (load_case, load_case_snapshot, load_forecast, load_observation, save_case, save_forecast,
                               save_image_asset, save_observation, write_json, write_yaml)


def forecast_value(kind="grid", mismatch=True):
    from hwoslaps.fisher.positions import explicit_positions, grid_positions
    from hwoslaps.fisher.result import ForecastResult
    positions = (grid_positions((0.1, -0.2), spacing_arcsec=0.4, half_width_arcsec=0.4, annulus=None)
                 if kind == "grid" else explicit_positions([[0.1, 0.2], [-0.3, 0.4]], (0.1, -0.2)))
    count = len(positions)
    raw = np.arange(1, count + 1, dtype=float)[None, :]
    profiled = raw * 0.5
    profiled[0, 0] = 0.0
    amplitude = np.full(raw.shape, -0.5) if mismatch else None
    if mismatch:
        amplitude[0, 0] = np.nan
    return ForecastResult(np.array([1e8]), positions, raw, profiled, amplitude,
                          None if amplitude is None else amplitude.copy(),
                          "kernel" if mismatch else "matched", {"scene": {"lens": {"mass": {"zeta": {"type": "Isothermal"}, "alpha": {"type": "Isothermal"}}}}}, {"threshold": 2.0})


def observation_value(noisy=False, include_lens=False):
    from hwoslaps.instrument import Detector
    from hwoslaps.observation.expected import Exposure
    from hwoslaps.observation.observation import Observation
    from hwoslaps.optics.kernels import DetectorPSF, KernelBinding
    from hwoslaps.scene.spec import GridSpec
    kernel = DetectorPSF.from_array(np.ones((3, 3)) / 9.0, 0.1, normalize=False)
    other = DetectorPSF.from_array(np.array([[0.0, 0.1, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.0]]), 0.1,
                                   normalize=False)
    psfs = KernelBinding((kernel, other), {"source:a": 0, "source:b": 0, "source:c": 1})
    rate = np.arange(6, dtype=float).reshape(2, 3)
    by_plane = {"source": rate}
    sampling = {"source:a": 0.01, "source:b": 0.02, "source:c": 0.03}
    if include_lens:
        by_plane["lens"] = np.full(rate.shape, 0.25)
        rate = by_plane["lens"] + by_plane["source"]
        psfs = KernelBinding(psfs.kernels, {**dict(psfs.group_index), "lens": 0})
        sampling["lens"] = 0.01
    exposure = Exposure(Detector(2.0, 0.5, 0.1), 10.0, 1.0, 2)
    expected = exposure.mean_adu(rate)
    observed = Observation("expected", expected, expected, exposure.noise_map_adu(rate), rate, by_plane,
                           GridSpec((2, 3), 0.1, 4), exposure, psfs, None, None, "c" * 64, None,
                           sampling)
    return observed.draw(11) if noisy else observed


def case_value(custom_mask=False):
    from hwoslaps.inference.result import CaseResult, ObservationRecord, RoleFit
    from hwoslaps.inference.settings import FitSpec, PixelMask, SamplerSettings
    from hwoslaps.scene.cosmology import Cosmology, parse_cosmology
    from hwoslaps.scene.halos import HaloModel, make_halo
    halo = make_halo(HaloModel("PointMass", None, None), 1e8, (0.1, -0.2), redshift=0.2,
                     source_redshift=0.6, cosmology=Cosmology(parse_cosmology({"name": "Planck15"})))
    smooth = RoleFit("smooth", "search", "success", -3.0, -3.0, ("x",), 1, None, None, {"x": 0.25}, None, None, 0.1)
    subhalo = RoleFit("subhalo", "truth_anchor", "success", -1.0, -1.0, (), 0, None, None, None, 0.0, None, 0.1)
    fit = (FitSpec("fixed_template", mask=PixelMask(np.array([[True, False, True], [False, True, False]])))
           if custom_mask else FitSpec("fixed_template"))
    return CaseResult("case", halo, ObservationRecord("expected", None, "c" * 64, "d" * 64, halo),
                      {"kind": "expected", "shape": [2, 3]}, fit, SamplerSettings(), 19, None,
                      {"smooth": "s", "subhalo": "h"}, smooth, subhalo, 4.0, 4.0, None, None, None,
                      ("x",), "e" * 64, {"version": "synthetic"})


@pytest.mark.parametrize("kind,mismatch", [("grid", True), ("grid", False), ("explicit", True)])
def test_forecast_round_trip_preserves_arrays_layout_and_provenance(tmp_path, kind, mismatch):
    original = forecast_value(kind, mismatch)
    first = save_forecast(original, tmp_path / "first.npz")
    second = save_forecast(original, tmp_path / "second.npz")
    assert first.read_bytes() == second.read_bytes()
    loaded = load_forecast(first)
    for name in ("masses_msun", "fisher_raw", "fisher_profiled", "sigma_amplitude", "q_asimov", "degradation",
                 "amplitude_hat", "q_mismatch", "z_mismatch", "amplitude_spurious", "q_spurious", "z_spurious"):
        left, right = getattr(original, name), getattr(loaded, name)
        if left is None:
            assert right is None
        else:
            assert left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
    for name in ("positions_yx", "cell_areas_arcsec2", "boundary"):
        left, right = getattr(original.positions, name), getattr(loaded.positions, name)
        assert right is None if left is None else left.tobytes() == right.tobytes()
    assert original.positions.kind == loaded.positions.kind
    assert original.positions.domain_radius_arcsec == loaded.positions.domain_radius_arcsec
    assert original.config == loaded.config and original.provenance == loaded.provenance
    assert tuple(original.config["scene"]["lens"]["mass"]) == tuple(loaded.config["scene"]["lens"]["mass"])
    if original.positions.grid is not None:
        assert original.positions.grid.indices.tobytes() == loaded.positions.grid.indices.tobytes()
        assert original.positions.grid.spacing_arcsec == loaded.positions.grid.spacing_arcsec


@pytest.mark.parametrize("noisy", [False, True])
def test_observation_round_trip_preserves_bytes_sampling_and_kernel_sharing(tmp_path, noisy):
    original = observation_value(noisy)
    loaded = load_observation(save_observation(original, tmp_path / "observation.npz"))
    for name in ("data_adu", "expected_adu", "noise_map_adu", "light_rate_e_per_s"):
        assert getattr(original, name).tobytes() == getattr(loaded, name).tobytes()
        assert not getattr(loaded, name).flags.writeable
    assert (loaded.data_adu is loaded.expected_adu) == (not noisy)
    assert loaded.light_rate_by_plane_e_per_s["source"] is loaded.light_rate_e_per_s
    assert loaded.psfs.for_group("source:a") is loaded.psfs.for_group("source:b")
    assert loaded.psfs.for_group("source:a") is not loaded.psfs.for_group("source:c")
    assert original.to_mapping() == loaded.to_mapping()


@pytest.mark.backend
@pytest.mark.parametrize("custom_mask", [False, True])
@pytest.mark.parametrize("replace_after_read", [False, True])
def test_case_round_trip_preserves_typed_result(tmp_path, monkeypatch, custom_mask, replace_after_read):
    import hwoslaps.artifacts as artifacts
    original = case_value(custom_mask)
    path = save_case(original, tmp_path / "case.json")
    original_bytes = path.read_bytes()
    replacement = replace(original, smooth=replace(original.smooth, log_likelihood=-4., truth_log_likelihood=-4.),
                          q_signed=6., q_clipped=6.)
    replacement_bytes = save_case(replacement, tmp_path / "replacement.json").read_bytes()
    real_snapshot = artifacts.read_file_snapshot
    def replace_file_after_actual_read(filename):
        content, digest = real_snapshot(filename)
        if replace_after_read:
            path.write_bytes(replacement_bytes)
        return content, digest
    monkeypatch.setattr(artifacts, "read_file_snapshot", replace_file_after_actual_read)
    loaded, digest = load_case_snapshot(path)
    assert original.to_mapping() == loaded.to_mapping()
    assert digest == hashlib.sha256(original_bytes).hexdigest(), "case digest must identify the bytes actually validated"
    assert load_case(path).q_signed == (6. if replace_after_read else 4.)
    assert path.read_bytes() == (replacement_bytes if replace_after_read else original_bytes)


@pytest.mark.parametrize("writer,payload", [(save_forecast, forecast_value()), (save_observation, observation_value()),
                                          pytest.param(save_case, case_value(), marks=pytest.mark.backend), (write_json, {"x": 1}),
                                          (write_yaml, {"x": 1})])
def test_publish_refuses_existing_destination_and_preserves_original_bytes(tmp_path, writer, payload):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        writer(payload, path) if writer in (save_forecast, save_observation, save_case) else writer(path, payload)
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("defect", ["old_version", "unknown_member", "object_array", "partial_grid", "bad_statistic",
                                    "nonmapping_provenance", "missing_q_asimov", "reshaped_q_same_bytes"])
def test_forecast_loader_refuses_other_schemas_members_and_pickle(tmp_path, defect):
    """Preserve archive key/metadata domains and statistic axes independently of bytes.

    The typed roundtrip keeper protects valid transport; it cannot reject a list
    provenance record, an omitted derived statistic or transposed statistic axes.
    Omitting the respective mapping/member/shape producer guard must fail this
    existing primary loader keeper. Real save/load boundaries need no new seam.
    """
    path = save_forecast(forecast_value(), tmp_path / "valid.npz")
    with np.load(path, allow_pickle=False) as stored:
        members = {name: stored[name] for name in stored.files}
    if defect == "old_version":
        members["schema_version"] = np.asarray(1)
    elif defect == "unknown_member":
        members["unknown"] = np.asarray(0)
    elif defect == "object_array":
        members["masses_msun"] = np.array([{}], dtype=object)
    elif defect == "partial_grid":
        del members["grid_indices"]
    elif defect == "nonmapping_provenance":
        members["provenance_json"] = np.asarray("[]")
    elif defect == "missing_q_asimov":
        del members["q_asimov"]
    elif defect == "reshaped_q_same_bytes":
        original = members["q_asimov"]
        members["q_asimov"] = original.T
        assert original.shape[0] == 1 and original.shape[1] > 1
        assert members["q_asimov"].shape != original.shape
        assert members["q_asimov"].tobytes() == original.tobytes()
    else:
        members["q_asimov"] = members["q_asimov"] + 1.0
    np.savez(tmp_path / "invalid.npz", **members)
    diagnostics = {
        "nonmapping_provenance": "provenance must be a mapping",
        "missing_q_asimov": "invalid forecast artifact member set",
        "reshaped_q_same_bytes": "forecast statistic q_asimov is inconsistent with the stored information",
    }
    with pytest.raises(ValueError, match=diagnostics.get(defect)):
        load_forecast(tmp_path / "invalid.npz")


def test_text_writers_use_finite_canonical_values_and_refuse_duplicate_json_keys(tmp_path):
    path = write_json(tmp_path / "values.json", {"x": np.int64(3), "pair": (1.0, 2.0)})
    assert json.loads(path.read_text()) == {"pair": [1.0, 2.0], "x": 3}
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"x": np.nan})
    assert not (tmp_path / "bad.json").exists()
    malformed = tmp_path / "duplicate.json"
    malformed.write_text('{"schema":"hwoslaps.case","version":1,"version":2,"result":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        load_case(malformed)


@pytest.mark.parametrize("defect,match", [
    ("data_inf", "data_adu.*non-finite"), ("expected_nan", "expected_adu.*non-finite"),
    ("rate_nan", "light_rate_e_per_s.*non-finite"), ("plane_nan", "light_rate_by_plane.*non-finite"),
    ("sigma_zero", "strictly positive"), ("sigma_negative", "strictly positive"),
    ("grid_scale", "pixel_scale_arcsec"), ("grid_oversampling", "over_sample_size"),
    ("grid_shape_bool", "grid.shape"), ("kernel_sampling", "angular sampling"),
    ("seed_text", "noise_seed"), ("seed_bool", "noise_seed"), ("digest_bool", "config_digest"),
])
def test_observation_loader_rejects_single_field_domain_defects(tmp_path, defect, match):
    path = save_observation(observation_value(noisy=True, include_lens=True), tmp_path / "valid.npz")
    with np.load(path, allow_pickle=False) as stored:
        members = {name: stored[name] for name in stored.files}
    metadata = json.loads(str(members["metadata_json"]))
    field = {"data_inf": "data_adu", "expected_nan": "expected_adu", "rate_nan": "light_rate_e_per_s",
             "plane_nan": "light__source_e_per_s", "sigma_zero": "noise_map_adu", "sigma_negative": "noise_map_adu"}.get(defect)
    if field is not None:
        members[field] = members[field].copy()
        members[field][0, 0] = {"data_inf": np.inf, "sigma_zero": 0.0, "sigma_negative": -1.0}.get(defect, np.nan)
    elif defect == "grid_scale":
        metadata["grid"]["pixel_scale_arcsec"] = -0.1
    elif defect == "grid_oversampling":
        metadata["grid"]["over_sample_size"] = 0
    elif defect == "grid_shape_bool":
        metadata["grid"]["shape"] = [True, 3]
    elif defect == "kernel_sampling":
        metadata["grid"]["pixel_scale_arcsec"] = 0.2
    elif defect == "seed_text":
        metadata["noise_seed"] = "eleven"
    elif defect == "seed_bool":
        metadata["noise_seed"] = True
    else:
        metadata["config_digest"] = False
    members["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(tmp_path / "invalid.npz", **members)
    with pytest.raises(ValueError, match=match):
        load_observation(tmp_path / "invalid.npz")


def test_observation_transport_keeps_valid_negative_noisy_data(tmp_path):
    from dataclasses import replace
    original = observation_value(noisy=True)
    data = original.data_adu.copy()
    data[0, 0] = -5.0
    original = replace(original, data_adu=data)
    loaded = load_observation(save_observation(original, tmp_path / "negative.npz"))
    assert loaded.data_adu[0, 0] == -5.0
    assert loaded.data_adu.tobytes() == original.data_adu.tobytes()
