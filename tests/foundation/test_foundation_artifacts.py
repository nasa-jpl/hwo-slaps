"""Current artifact formats, exact scientific values and atomic no-overwrite publishing."""
import json

import numpy as np
import pytest

from hwoslaps.artifacts import (load_case, load_forecast, load_observation, save_case, save_forecast,
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


def observation_value(noisy=False):
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
    exposure = Exposure(Detector(2.0, 0.5, 0.1), 10.0, 1.0, 2)
    expected = exposure.mean_adu(rate)
    observed = Observation("expected", expected, expected, exposure.noise_map_adu(rate), rate, {"source": rate},
                           GridSpec((2, 3), 0.1, 4), exposure, psfs, None, None, "c" * 64, None,
                           {"source:a": 0.01, "source:b": 0.02, "source:c": 0.03})
    return observed.draw(11) if noisy else observed


def case_value():
    from hwoslaps.inference.result import CaseResult, ObservationRecord, RoleFit
    from hwoslaps.inference.settings import FitSpec, SamplerSettings
    from hwoslaps.scene.cosmology import Cosmology, parse_cosmology
    from hwoslaps.scene.halos import HaloModel, make_halo
    halo = make_halo(HaloModel("PointMass", None, None), 1e8, (0.1, -0.2), redshift=0.2,
                     source_redshift=0.6, cosmology=Cosmology(parse_cosmology({"name": "Planck15"})))
    smooth = RoleFit("smooth", "search", "success", -3.0, -3.0, ("x",), 1, None, None, {"x": 0.25}, None, None, 0.1)
    subhalo = RoleFit("subhalo", "truth_anchor", "success", -1.0, -1.0, (), 0, None, None, None, 0.0, None, 0.1)
    return CaseResult("case", halo, ObservationRecord("expected", None, "c" * 64, "d" * 64, halo),
                      {"kind": "expected"}, FitSpec("fixed_template"), SamplerSettings(), 19, None,
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
def test_case_round_trip_preserves_typed_result(tmp_path):
    original = case_value()
    loaded = load_case(save_case(original, tmp_path / "case.json"))
    assert original.to_mapping() == loaded.to_mapping()


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


@pytest.mark.parametrize("defect", ["old_version", "unknown_member", "object_array", "partial_grid", "bad_statistic"])
def test_forecast_loader_refuses_other_schemas_members_and_pickle(tmp_path, defect):
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
    else:
        members["q_asimov"] = members["q_asimov"] + 1.0
    np.savez(tmp_path / "invalid.npz", **members)
    with pytest.raises(ValueError):
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
